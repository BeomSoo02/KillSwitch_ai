# app.py — KillSwitch AI (부스 전시용, GPT 항상 표시)
# - 전시용: 큰 판정 카드 + 간단 이유
# - GPT 응답: 전문가 모드와 무관하게 항상 섹션 표시 (Key/허용 시 즉시 호출)
# - 전문가 모드: 세부 지표/JSON만 토글
# --------------------------------------------------------------------------------------------
import os, re, time, unicodedata
import streamlit as st

st.set_page_config(page_title="KillSwitch AI", layout="wide")

# ===== 1) 환경/시크릿 =====
DEFAULT_REPO_ID   = "cookiechips/KillSwitch_ai"   # 기본값: 공개 리포
DEFAULT_REPO_TYPE = "model"
DEFAULT_FILENAME  = "prompt_guard_best.pt"

REPO_ID   = st.secrets.get("HF_REPO_ID")   or os.getenv("HF_REPO_ID")   or DEFAULT_REPO_ID
REPO_TYPE = st.secrets.get("HF_REPO_TYPE") or os.getenv("HF_REPO_TYPE") or DEFAULT_REPO_TYPE
FILENAME  = st.secrets.get("HF_FILENAME")  or os.getenv("HF_FILENAME")  or DEFAULT_FILENAME
HF_TOKEN  = st.secrets.get("HF_TOKEN")     or os.getenv("HF_TOKEN")     # 비공개 리포면 필요
HF_DIR    = st.secrets.get("HF_DIR")       or os.getenv("HF_DIR")       # 완전 모델 디렉토리(옵션)

BASE_MODEL = os.getenv("BASE_MODEL") or "microsoft/deberta-v3-base"
NUM_LABELS = int(os.getenv("NUM_LABELS") or 2)

# 메타 가중치
W_ACTION   = 0.15   # 실행형(강한 action)
W_INFO     = -0.15  # 설명형(info) — 순수 정보성일수록 감산
W_DOMAIN   = 0.25   # 위험 단어(domain risk)
GAP_THR    = 0.10   # softmax 확신 임계값
W_UNCERT   = -0.10  # gap < 0.10이면 불확실 → 감쇠

# ===== 2) 모델/허브 로딩 =====
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def get_ckpt_path() -> str:
    """허브에서 .pt 체크포인트 다운로드 (기본 repo_type 실패 시 반대 타입도 시도)."""
    local = os.path.join("model", FILENAME)
    if os.path.exists(local):
        return local
    try:
        return hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type=REPO_TYPE, token=HF_TOKEN)
    except Exception:
        alt = "dataset" if REPO_TYPE == "model" else "model"
        return hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type=alt, token=HF_TOKEN)

@st.cache_resource(show_spinner=False)
def load_model_tokenizer():
    """
    1) HF_DIR 있으면 완전 모델 디렉토리에서 from_pretrained
    2) 아니면 BASE_MODEL 로드 후 .pt state_dict 주입
    - PyTorch 2.6+ 호환: torch.load(..., weights_only=False) 명시
    """
    # 토크나이저: slow 우선 → 실패 시 fast
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
        tok_info = "slow"
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        tok_info = "fast"

    # 1) 완전 모델 디렉토리
    if HF_DIR and os.path.isdir(HF_DIR):
        mdl = AutoModelForSequenceClassification.from_pretrained(HF_DIR)
        mdl.to(DEVICE).eval()
        return mdl, tok, 0.5, True, tok_info

    # 2) 베이스 + state_dict
    mdl = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)
    thr = 0.5
    torch_loaded = False
    try:
        ckpt_path = get_ckpt_path()
        # ⚠ PyTorch 2.6 기본값 변경 대응: weights_only=False 로 명시
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        missing, unexpected = mdl.load_state_dict(state, strict=False)
        if missing or unexpected:
            st.caption(f"state_dict mismatch → missing:{len(missing)}, unexpected:{len(unexpected)}")

        # 체크포인트에 임계값이 저장돼 있으면 사용 (선택)
        if isinstance(ckpt, dict) and "val_thr" in ckpt:
            try:
                thr = float(ckpt["val_thr"])
            except Exception:
                pass

        torch_loaded = True
    except Exception as e:
        st.error("체크포인트 로드 실패 — 모델 미로딩")
    mdl.to(DEVICE).eval()
    return mdl, tok, thr, torch_loaded, tok_info

_cached_model = load_model_tokenizer

# ===== 3) 전처리 & 스코어링 =====
def preprocess(s: str) -> str:
    """NFKC 정규화 + 공백 정리 + 끝마침표 강제(이미 있으면 추가하지 않음)."""
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s.endswith(".") else (s + ".")

def _softmax_two(logits: torch.Tensor):
    """이진 분류 가정: p0, p1과 gap(|p1 - p0|) 반환."""
    if logits.size(-1) == 1:
        p1 = torch.sigmoid(logits)[0, 0]
        p0 = 1 - p1
    else:
        probs = torch.softmax(logits, dim=-1)[0]
        p0, p1 = probs[0], probs[1]
    gap = torch.abs(p1 - p0)
    return p0.item(), p1.item(), gap.item()

@torch.no_grad()
def model_forward(mdl, tok, text: str):
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    for k in enc:
        enc[k] = enc[k].to(DEVICE)
    out = mdl(**enc)
    logits = out.logits
    p0, p1, gap = _softmax_two(logits)
    return p0, p1, gap  # p1 = "악성" 확률

# ===== 4) 메타 가중치 탐지 =====
ACTION_PAT = re.compile(
    r"(해줘|만들어|구현(해|하)|코드(를|로)|스크립트|자동화|우회|우회해|실행해|실행법|공격해|뚫(어|는)|우회하는|차단 해제|크랙|패치|익스플로잇|exploit|payload|실행 가능한|download|다운로드|설치해|주입해|inject)",
    re.IGNORECASE,
)
INFO_PAT = re.compile(
    r"(의미|정의|원리|이유|개념|동작 원리|왜|무엇|어떻게( 되는| 작동)|설명해|설명|정리해|알려줘|요약|가이드|가르쳐)",
    re.IGNORECASE,
)
DANGER_WORDS = [
    "폭탄", "ddos", "디도스", "살해", "살인", "무기", "총기", "폭발물",
    "랜섬웨어", "백도어", "malware", "악성코드", "키로거", "익스플로잇", "exploit",
    "해킹", "크랙", "크래킹", "피싱", "스미싱",
    "신용카드 번호", "카드 번호", "cvv", "비밀번호 탈취", "은행 정보", "개인정보 탈취",
]

def detect_meta_flags(text: str):
    t = text.lower()
    is_action = bool(ACTION_PAT.search(t))
    is_info   = bool(INFO_PAT.search(t))
    is_danger = any(w.lower() in t for w in DANGER_WORDS)
    return is_action, is_info, is_danger

def apply_meta_weights(p1: float, gap: float, is_action: bool, is_info: bool, is_danger: bool):
    """원점수 p1에 메타 가중치 합산, 불확실(gap<GAP_THR)이면 감쇠. 결과는 [0,1]로 클램프."""
    score = p1
    if is_action: score += W_ACTION
    if is_info:   score += W_INFO
    if is_danger: score += W_DOMAIN
    if gap < GAP_THR: score += W_UNCERT
    score = max(0.0, min(1.0, score))
    return score

# ===== 5) 예측 파이프라인 =====
def predict(text: str, thr_ui: float):
    text = preprocess(text)
    mdl, tok, thr_ckpt, torch_loaded, tok_info = _cached_model()
    t0 = time.time()

    if torch_loaded:
        _, p1, gap = model_forward(mdl, tok, text)    # p1 = 악성 확률(원점수)
    else:
        p1, gap = 0.0, 0.0  # 모델 미로딩 실패 시 안전 쪽으로

    is_action, is_info, is_danger = detect_meta_flags(text)
    adj_score = apply_meta_weights(p1, gap, is_action, is_info, is_danger)

    thr = float(thr_ui if thr_ui is not None else thr_ckpt)
    label = "악성" if adj_score >= thr else "안전"

    return {
        "원점수(p1)": round(p1, 3),
        "조정점수(p1+가중치)": round(adj_score, 3),
        "임계값": round(thr, 3),
        "판정": label,
        "근거": {
            "softmax_gap(|p1-p0|)": round(gap, 3),
            "플래그": {
                "실행형_action": is_action,
                "설명형_info": is_info,
                "위험단어_domain": is_danger,
            },
        },
        "_elapsed_s": round(time.time() - t0, 2),
        "세부": {"device": str(DEVICE), "tokenizer": tok_info, "torch_loaded": bool(torch_loaded)},
    }

# ===== 6) UI =====
st.title("🛡️ KillSwitch AI")

# ── 사이드바 ──────────────────────────────────────────────────────────────
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""

st.sidebar.header("설정")
OPENAI_API_KEY = st.sidebar.text_input("OPENAI_API_KEY", value=st.session_state.OPENAI_API_KEY, type="password")
if OPENAI_API_KEY != st.session_state.OPENAI_API_KEY:
    st.session_state.OPENAI_API_KEY = OPENAI_API_KEY

openai_model = st.sidebar.text_input("OpenAI 모델", value="gpt-4o-mini")
thr_ui       = st.sidebar.slider("임계값(차단 기준)", 0.05, 0.95, 0.70, step=0.05)
force_call   = st.sidebar.checkbox("위험해도 GPT 호출 강행", value=False)
expert_mode  = st.sidebar.toggle("🛠️ 전문가 모드 (세부 지표/JSON만)", value=False)

with st.sidebar.expander("HF 연결 점검"):
    st.caption(f"HF: {REPO_ID} ({REPO_TYPE}) / {FILENAME}")
    if st.button("체크"):
        try:
            p = get_ckpt_path()
            size_mb = os.path.getsize(p) / 1_048_576
            st.success(f"OK: {os.path.basename(p)} · {size_mb:.1f} MB")
        except Exception as e:
            st.error("다운로드 실패")
            st.exception(e)

# ── 메인 입력 ─────────────────────────────────────────────────────────────
txt = st.text_area("프롬프트", height=140, placeholder="예) 인천 맛집 알려줘")

def summarize_reason(result: dict) -> str:
    flags = result["근거"]["플래그"]
    reasons = []
    if flags["위험단어_domain"]: reasons.append("위험 키워드 포함")
    if flags["실행형_action"]:   reasons.append("실행 지시어 탐지")
    if flags["설명형_info"]:     reasons.append("정보 요청 위주")
    g = result["근거"]["softmax_gap(|p1-p0|)"]
    conf = "높음" if g >= 0.40 else ("보통" if g >= 0.20 else "낮음")
    msg = " · ".join(reasons) if reasons else "정상적인 안내 요청"
    return f"{msg} · 확신도: {conf}"

@st.cache_resource(show_spinner=False)
def get_openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"openai 라이브러리 불러오기 실패: {e}")
    api_key = (
        st.secrets.get("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or st.session_state.OPENAI_API_KEY
    )
    return OpenAI(api_key=api_key) if api_key else None

# ── 버튼 & 출력 ───────────────────────────────────────────────────────────
if st.button("분석"):
    if not (txt and txt.strip()):
        st.warning("텍스트를 입력하세요.")
    else:
        with st.spinner("분석 중..."):
            result = predict(txt, thr_ui)
        st.success(f"분석 완료 ({result['_elapsed_s']:.2f}s)")

        verdict = result["판정"]
        reason = summarize_reason(result)

        # ▶ 전시용 판정 카드
        if verdict == "악성":
            st.markdown(
                f"""
                <div style="padding:18px;border-radius:14px;background:#ffe8e6;border:1px solid #ffb4ac">
                  <div style="font-size:22px;font-weight:700;color:#8c1d18;">⚠️ 위험한 요청으로 판단했습니다</div>
                  <div style="margin-top:6px;font-size:16px;color:#5a1a17;">사유: {reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="padding:18px;border-radius:14px;background:#e8f5e9;border:1px solid #b7e1be">
                  <div style="font-size:22px;font-weight:700;color:#1b5e20;">✅ 안전한 요청입니다</div>
                  <div style="margin-top:6px;font-size:16px;color:#1b5e20;">사유: {reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ▶ GPT 응답: 전문가 모드와 무관하게 항상 섹션 노출
        st.markdown("### 🤖 GPT 응답")
        key_from_secrets = bool(st.secrets.get("OPENAI_API_KEY"))
        key_from_env     = bool(os.getenv("OPENAI_API_KEY"))
        key_from_session = bool(st.session_state.get("OPENAI_API_KEY"))
        key_ok = key_from_secrets or key_from_env or key_from_session

        allow_call = key_ok and (verdict == "안전" or (verdict == "악성" and force_call))

        if not key_ok:
            st.info("OPENAI_API_KEY가 설정되어 있지 않아 GPT 호출을 생략합니다.")
        elif not allow_call:
            st.warning("악성 판정으로 GPT 호출이 차단되었습니다. (사이드바 '위험해도 GPT 호출 강행'을 켜면 호출됩니다)")
        else:
            try:
                client = get_openai_client()
                if client is None:
                    st.info("API Key가 설정되지 않았습니다.")
                else:
                    rsp = client.responses.create(
                        model=openai_model,
                        input=[
                            {"role": "system",
                             "content": "You are a helpful assistant. If the user's prompt is harmful or illegal, politely refuse and guide them safely."},
                            {"role": "user", "content": txt},
                        ],
                        temperature=0.3,
                        top_p=1.0,
                    )
                    st.write(rsp.output_text)
            except Exception as e:
                st.error(f"GPT 호출 오류: {type(e).__name__}: {e}")

        # ▶ 전문가 모드: 세부 지표/JSON만
        if expert_mode:
            with st.expander("🔎 세부 지표", expanded=False):
                c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])
                with c1: st.metric("판정", verdict)
                with c2: st.metric("원점수(p1)", f"{result['원점수(p1)']:.3f}")
                with c3: st.metric("조정점수", f"{result['조정점수(p1+가중치)']:.3f}")
                with c4: st.metric("임계값", f"{result['임계값']:.2f}")
                with c5: st.metric("확신도(gap)", f"{result['근거']['softmax_gap(|p1-p0|)']:.3f}")
                st.caption(f"device={result['세부']['device']} · tokenizer={result['세부']['tokenizer']} · torch_loaded={result['세부']['torch_loaded']}")

            with st.expander("🧾 원본 결과(JSON)", expanded=False):
                st.json({k: v for k, v in result.items() if not k.startswith("_")})
