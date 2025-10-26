# app.py — KillSwitch AI (최종본, 요약 → GPT → 세부 + 속도 최적화)
# - 모델 점수 + 메타가중치(실행형/설명형/위험단어) + softmax 확신(gap) 감쇠
# - 끝마침표 강제 / 키 상태 / HF 점검 / PyTorch 2.6 대응
# - 성능 저하 없이 속도 개선: fast tokenizer, FP16(CUDA), max_length 기본 224, GPT 호출 분리
# -------------------------------------------------------------------------------------------------
import os, re, time, unicodedata
import streamlit as st

# (선택) CPU 환경에서 쓰레드 과다 방지 — 과하면 렌더링/추론 지연될 수 있음
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

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

BASE_MODEL = os.getenv("BASE_MODEL") or "microsoft/deberta-v3-base"  # ⬅ 모델 축소 없음(keep base)
NUM_LABELS = int(os.getenv("NUM_LABELS") or 2)

# 메타 가중치
W_ACTION   = 0.15   # 실행형(강한 action)
W_INFO     = -0.15  # 설명형(info)
W_DOMAIN   = 0.25   # 위험 단어(domain risk)
GAP_THR    = 0.10   # softmax 확신 임계값
W_UNCERT   = -0.10  # gap < 0.10이면 불확실 → 감쇠

# ===== 2) 모델/허브 로딩 =====
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    torch.set_num_threads(4)
except Exception:
    pass

@st.cache_resource(show_spinner=False)
def get_ckpt_path() -> str:
    """허브에서 .pt 체크포인트 다운로드 (기본 repo_type 실패 시 반대 타입도 시도)."""
    # 로컬 포함 시 우선 사용
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
    - 속도: 토크나이저 fast 강제, CUDA면 FP16 사용
    """
    # 토크나이저: fast 고정 (속도↑, 정확도 영향 거의 없음)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok_info = "fast"

    # 1) 완전 모델 디렉토리
    if HF_DIR and os.path.isdir(HF_DIR):
        mdl = AutoModelForSequenceClassification.from_pretrained(HF_DIR)
        mdl.to(DEVICE).eval()
        if DEVICE.type == "cuda":
            try: mdl.half()
            except Exception: pass
        return mdl, tok, 0.5, True, tok_info

    # 2) 베이스 + state_dict
    mdl = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)
    thr = 0.5
    torch_loaded = False
    try:
        ckpt_path = get_ckpt_path()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # 2.6 대응
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        missing, unexpected = mdl.load_state_dict(state, strict=False)
        if missing or unexpected:
            st.caption(f"state_dict mismatch → missing:{len(missing)}, unexpected:{len(unexpected)}")

        # 체크포인트에 임계값 저장돼 있으면 사용(선택)
        if isinstance(ckpt, dict) and "val_thr" in ckpt:
            try:
                thr = float(ckpt["val_thr"])
            except Exception:
                pass

        torch_loaded = True
    except Exception as e:
        st.error("체크포인트 로드 실패 — 모델 미로딩")
        st.caption(str(e))

    mdl.to(DEVICE).eval()
    if DEVICE.type == "cuda":
        try: mdl.half()
        except Exception: pass

    return mdl, tok, thr, torch_loaded, tok_info

# ✅ 중복 캐싱 제거: 함수 자체가 캐싱됨
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
        # 시그모이드 헤드인 경우, 로짓 하나를 [p0, p1]로 환산
        p1 = torch.sigmoid(logits)[0, 0]
        p0 = 1 - p1
    else:
        probs = torch.softmax(logits, dim=-1)[0]
        p0, p1 = probs[0], probs[1]
    gap = torch.abs(p1 - p0)
    return p0.item(), p1.item(), gap.item()

@torch.inference_mode()
def model_forward(mdl, tok, text: str, max_len: int = 224):
    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False,     # 단건 입력이면 패딩 불필요 → 속도↑
        max_length=max_len
    )
    seq_len = int(enc["input_ids"].shape[-1])
    for k in enc:
        enc[k] = enc[k].to(DEVICE)
    out = mdl(**enc)
    logits = out.logits
    p0, p1, gap = _softmax_two(logits)
    return p0, p1, gap, seq_len, max_len  # p1 = "악성" 확률

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
    adjustments = {}
    score = p1

    if is_action:
        score += W_ACTION
        adjustments["실행형(+0.15)"] = True
    else:
        adjustments["실행형(+0.15)"] = False

    if is_info:
        score += W_INFO
        adjustments["설명형(-0.15)"] = True
    else:
        adjustments["설명형(-0.15)"] = False

    if is_danger:
        score += W_DOMAIN
        adjustments["위험단어(+0.25)"] = True
    else:
        adjustments["위험단어(+0.25)"] = False

    if gap < GAP_THR:
        score += W_UNCERT
        adjustments["불확실감쇠(gap<0.10→-0.10)"] = True
    else:
        adjustments["불확실감쇠(gap<0.10→-0.10)"] = False

    score = max(0.0, min(1.0, score))
    return score, adjustments

# ===== 5) 예측 파이프라인 =====
def predict(text: str, thr_ui: float, max_len_ui: int):
    text = preprocess(text)
    mdl, tok, thr_ckpt, torch_loaded, tok_info = _cached_model()
    t0 = time.time()

    if torch_loaded:
        p0, p1, gap, seq_len, max_len = model_forward(mdl, tok, text, max_len=max_len_ui)  # p1 = 악성 확률
    else:
        p0, p1, gap, seq_len, max_len = 1.0, 0.0, 0.0, 0, max_len_ui  # 모델 미로딩 시 안전 쪽으로
    is_action, is_info, is_danger = detect_meta_flags(text)
    adj_score, adj_map = apply_meta_weights(p1, gap, is_action, is_info, is_danger)

    thr = float(thr_ui if thr_ui is not None else thr_ckpt)
    label = "악성" if adj_score >= thr else "안전"

    return {
        "원점수(p1)": round(p1, 3),
        "조정점수(p1+가중치)": round(adj_score, 3),
        "임계값": round(thr, 3),
        "판정": label,
        "근거": {
            "softmax_gap(|p1-p0|)": round(abs(p1 - p0), 3),
            "가중치적용": adj_map,
            "플래그": {
                "실행형_action": is_action,
                "설명형_info": is_info,
                "위험단어_domain": is_danger,
            },
        },
        "세부": {
            "torch_loaded": bool(torch_loaded),
            "device": str(DEVICE),
            "tokenizer": tok_info,
            "seq_len": int(seq_len),
            "max_len": int(max_len),
        },
        "_elapsed_s": round(time.time() - t0, 2),
    }

# ===== 6) UI =====
st.title("🛡️ KillSwitch AI")

# 세션 상태(키)
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""

# 🔐 키 상태
with st.sidebar.expander("🔐 키 상태"):
    key_from_secrets = bool(st.secrets.get("OPENAI_API_KEY"))
    key_from_env     = bool(os.getenv("OPENAI_API_KEY"))
    key_from_session = bool(st.session_state.get("OPENAI_API_KEY"))
    key_ok = key_from_secrets or key_from_env or key_from_session
    st.write("OpenAI Key:", "✅ 감지됨" if key_ok else "❌ 없음")
    st.caption(
        f"• Secrets: {'✅' if key_from_secrets else '—'}   "
        f"• Env: {'✅' if key_from_env else '—'}   "
        f"• Session: {'✅' if key_from_session else '—'}"
    )

# 사이드바 옵션
OPENAI_API_KEY = st.sidebar.text_input("OPENAI_API_KEY", value=st.session_state.OPENAI_API_KEY, type="password")
if OPENAI_API_KEY != st.session_state.OPENAI_API_KEY:
    st.session_state.OPENAI_API_KEY = OPENAI_API_KEY

openai_model = st.sidebar.text_input("OpenAI 모델", value="gpt-4o-mini")
thr_ui       = st.sidebar.slider("임계값(차단 기준)", 0.05, 0.95, 0.70, step=0.05)
max_len_ui   = st.sidebar.slider("토큰 최대길이(max_length)", 128, 256, 224, step=32)  # 기본 224 (성능↓ 없이 속도↑)
force_call   = st.sidebar.checkbox("위험해도 GPT 호출 강행", value=False)

# 키워드 확인용
with st.sidebar.expander("메타 가중치 키워드(확인용)"):
    st.markdown("**실행형 패턴 예시**: 해줘, 만들어, 구현해, 코드, 스크립트, 우회, 실행해, 공격해, exploit, payload, 다운로드, 설치, inject…")
    st.markdown("**설명형 패턴 예시**: 의미, 정의, 원리, 이유, 개념, 왜, 무엇, 설명해, 정리해, 알려줘, 요약, 가이드…")
    st.markdown("**위험 단어 예시**: 폭탄, DDoS, 살해, 무기, 총기, 랜섬웨어, 백도어, 악성코드, 익스플로잇, 해킹, 피싱, 신용카드 번호, 비밀번호 탈취…")
    st.caption("문맥에 따른 세밀한 분류가 필요하면 여기에 키워드를 추가/조정하세요.")

# HF 연결 점검
st.sidebar.caption(f"HF: {REPO_ID} ({REPO_TYPE}) / {FILENAME}")
if st.sidebar.button("HF 연결 점검"):
    try:
        p = get_ckpt_path()
        size_mb = os.path.getsize(p) / 1_048_576
        st.sidebar.success(f"OK: {os.path.basename(p)} · {size_mb:.1f} MB")
    except Exception as e:
        st.sidebar.error("다운로드 실패")
        st.sidebar.exception(e)

# ===== 7) GPT 클라이언트 캐시 =====
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
    return OpenAI(api_key=api_key)

# ===== 8) 메인 입력 =====
txt = st.text_area("프롬프트", height=140, placeholder="예) 인천 맛집 알려줘")

# ===== 9) 버튼: 빠른 분석(모델만) / 분석+GPT =====
colA, colB = st.columns([1, 1])
run_only = colA.button("⚡ 빠른 분석 (모델만)")
run_with_gpt = colB.button("분석 + GPT 호출")

if run_only or run_with_gpt:
    if not (txt and txt.strip()):
        st.warning("텍스트를 입력하세요.")
    else:
        with st.spinner("분석 중..."):
            result = predict(txt, thr_ui, max_len_ui)

        # ① 최상단 요약
        st.success(f"분석 완료 ({result['_elapsed_s']:.2f}s)")
        st.markdown("### ✅ 요약")
        p1_raw   = result["원점수(p1)"]
        p1_adj   = result["조정점수(p1+가중치)"]
        thr_val  = result["임계값"]
        label    = result["판정"]
        gap_val  = result["근거"]["softmax_gap(|p1-p0|)"]

        c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])
        with c1: st.metric("판정", label)
        with c2: st.metric("원점수(p1)", f"{p1_raw:.3f}")
        with c3: st.metric("조정점수", f"{p1_adj:.3f}")
        with c4: st.metric("임계값", f"{thr_val:.2f}")
        with c5: st.metric("확신도(gap)", f"{gap_val:.3f}")

        # 길이 자름 경고
        seq_len = result["세부"]["seq_len"]; max_len = result["세부"]["max_len"]
        if seq_len >= max_len:
            st.caption(f"⚠️ 입력이 길어 일부가 잘렸을 수 있어요. (seq_len={seq_len} / max_length={max_len})")

        # ② GPT 응답 (요약 바로 아래, 버튼에 따라 호출)
        st.markdown("### 🤖 GPT 응답")
        key_from_secrets = bool(st.secrets.get("OPENAI_API_KEY"))
        key_from_env     = bool(os.getenv("OPENAI_API_KEY"))
        key_from_session = bool(st.session_state.get("OPENAI_API_KEY"))
        key_ok = key_from_secrets or key_from_env or key_from_session

        if not run_with_gpt:
            st.info("빠른 분석 모드: GPT 호출을 생략했습니다. (위의 '분석 + GPT 호출' 버튼으로 실행 가능)")
        else:
            if not key_ok:
                st.info("OPENAI_API_KEY가 없어 GPT 호출을 생략했습니다.")
            elif label == "악성" and not force_call:
                st.warning("악성으로 판정되어 GPT 호출이 차단되었습니다. (사이드바 '강행'을 체크하면 호출)")
            else:
                try:
                    client = get_openai_client()
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

        # ③ 세부/근거 (아래로 내림)
        with st.expander("🔍 근거 / 메타 세부 보기"):
            st.write({
                "softmax_gap(|p1-p0|)": gap_val,
                "가중치적용": result["근거"]["가중치적용"],
                "플래그": result["근거"]["플래그"],
                "세부": {
                    "torch_loaded": result["세부"]["torch_loaded"],
                    "device": result["세부"]["device"],
                    "tokenizer": result["세부"]["tokenizer"],
                    "seq_len": seq_len,
                    "max_len": max_len,
                },
            })

        with st.expander("🧾 원본 결과(JSON)"):
            st.json({k: v for k, v in result.items() if not k.startswith("_")})
