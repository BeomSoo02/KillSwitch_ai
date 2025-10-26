# app.py — KillSwitch AI (모델 점수만 + 끝마침표 강제 + 키 상태 + HF 점검 + PyTorch 2.6 패치)
# --------------------------------------------------------------------------------------------

import os, re, time, unicodedata
import streamlit as st

st.set_page_config(page_title="KillSwitch AI", layout="wide")

# ===== 1) 환경/시크릿 =====
DEFAULT_REPO_ID   = "cookiechips/KillSwitch_ai"   # 기본값: 사용 중이던 공개 리포
DEFAULT_REPO_TYPE = "model"
DEFAULT_FILENAME  = "prompt_guard_best.pt"

REPO_ID   = st.secrets.get("HF_REPO_ID")   or os.getenv("HF_REPO_ID")   or DEFAULT_REPO_ID
REPO_TYPE = st.secrets.get("HF_REPO_TYPE") or os.getenv("HF_REPO_TYPE") or DEFAULT_REPO_TYPE
FILENAME  = st.secrets.get("HF_FILENAME")  or os.getenv("HF_FILENAME")  or DEFAULT_FILENAME
HF_TOKEN  = st.secrets.get("HF_TOKEN")     or os.getenv("HF_TOKEN")     # 비공개 리포면 필요
HF_DIR    = st.secrets.get("HF_DIR")       or os.getenv("HF_DIR")       # 완전 모델 디렉토리(옵션)

BASE_MODEL = os.getenv("BASE_MODEL") or "microsoft/deberta-v3-base"
NUM_LABELS = int(os.getenv("NUM_LABELS") or 2)

# ===== 2) 모델/허브 로딩 =====
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def get_ckpt_path() -> str:
    """
    허브에서 .pt 체크포인트 다운로드 (기본 repo_type 실패 시 반대 타입도 시도).
    """
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

        # 체크포인트에 임계값이 저장돼 있으면 사용
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
    return mdl, tok, thr, torch_loaded, tok_info

_cached_model = st.cache_resource(show_spinner=False)(load_model_tokenizer)

# ===== 3) 전처리 & 끝마침표 강제 =====
def preprocess(s: str) -> str:
    """NFKC 정규화 + 공백 정리 + 끝마침표 강제(이미 있으면 추가하지 않음)."""
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s.endswith(".") else (s + ".")

@torch.no_grad()
def score_once(mdl, tok, text: str) -> float:
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    for k in enc:
        enc[k] = enc[k].to(DEVICE)
    logits = mdl(**enc).logits
    if logits.size(-1) == 1:   # 시그모이드 헤드
        return torch.sigmoid(logits)[0, 0].item()
    return torch.softmax(logits, dim=-1)[0, 1].item()  # 소프트맥스 1(악성) 확률

def predict(text: str, thr_ui: float):
    text = preprocess(text)
    mdl, tok, thr_ckpt, torch_loaded, tok_info = _cached_model()
    t0 = time.time()
    m_score = score_once(mdl, tok, text) if torch_loaded else 0.0
    thr = float(thr_ui if thr_ui is not None else thr_ckpt)
    label = "악성" if m_score >= thr else "안전"
    return {
        "점수": round(m_score, 3),
        "임계값": round(thr, 3),
        "판정": label,
        "세부": {
            "torch_loaded": bool(torch_loaded),
            "device": str(DEVICE),
            "tokenizer": tok_info,
        },
        "_elapsed_s": round(time.time() - t0, 2),
    }

# ===== 4) UI =====
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
OPENAI_API_KEY = st.sidebar.text_input(
    "OPENAI_API_KEY",
    value=st.session_state.OPENAI_API_KEY,
    type="password"
)
if OPENAI_API_KEY != st.session_state.OPENAI_API_KEY:
    st.session_state.OPENAI_API_KEY = OPENAI_API_KEY

openai_model = st.sidebar.text_input("OpenAI 모델", value="gpt-4o-mini")
thr_ui       = st.sidebar.slider("임계값(차단 기준)", 0.05, 0.95, 0.50, step=0.05)
force_call   = st.sidebar.checkbox("위험해도 GPT 호출 강행", value=False)

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

# 메인 입력
txt = st.text_area("프롬프트", height=140, placeholder="예) 인천 맛집 알려줘")
if st.button("분석 (GPT 호출)"):
    if not (txt and txt.strip()):
        st.warning("텍스트를 입력하세요.")
    else:
        with st.spinner("분석 중..."):
            result = predict(txt, thr_ui)
        st.success(f"분석 완료 ({result['_elapsed_s']:.2f}s)")
        st.subheader("분석 결과  ↪️")
        st.json({k: v for k, v in result.items() if not k.startswith("_")})

        # GPT 호출
        st.subheader("GPT 응답")
        if not key_ok:
            st.info("OPENAI_API_KEY가 없어 GPT 호출을 생략했습니다.")
        elif result["판정"] == "악성" and not force_call:
            st.warning("악성으로 판정되어 GPT 호출이 차단되었습니다. (사이드바 '강행'을 체크하면 호출)")
        else:
            try:
                from openai import OpenAI
                api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or st.session_state.OPENAI_API_KEY
                client = OpenAI(api_key=api_key)
                rsp = client.responses.create(
                    model=openai_model,
                    input=[
                        {"role": "system",
                         "content": "You are a helpful assistant. If the user's prompt is clearly harmful or illegal, politely refuse and provide safe alternatives."},
                        {"role": "user", "content": txt},
                    ],
                    temperature=0.3,
                    top_p=1.0,
                )
                st.write(rsp.output_text)
            except Exception as e:
                st.error(f"GPT 호출 오류: {type(e).__name__}: {e}")
