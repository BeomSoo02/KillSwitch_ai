# app.py — KillSwitch AI · Streamlit 데모 (모델 점수만 + 마침표 강건화 + 키 상태)
# ----------------------------------------------------------------------
# ✔ HF 체크포인트 로딩 (state_dict)
# ✔ 모델 점수만 사용(규칙/키워드 전부 제거)
# ✔ 마침표 강건화(원본/제거/추가) + mean/max 선택
# ✔ 사이드바 OPENAI_API_KEY 입력(세션 유지, password)
# ✔ 🔐 키 상태 표시 (Secrets/Env/Session)
# ✔ 필요 시 GPT 호출 (Responses API)
# ----------------------------------------------------------------------

import os, re, unicodedata
import streamlit as st

st.set_page_config(page_title="KillSwitch AI", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 1) 환경/시크릿
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REPO_ID   = "your-username/killswitch-ai-checkpoints"
DEFAULT_REPO_TYPE = "model"
DEFAULT_FILENAME  = "pt/prompt_guard_best.pt"

REPO_ID   = st.secrets.get("HF_REPO_ID")   or os.getenv("HF_REPO_ID")   or DEFAULT_REPO_ID
REPO_TYPE = st.secrets.get("HF_REPO_TYPE") or os.getenv("HF_REPO_TYPE") or DEFAULT_REPO_TYPE
FILENAME  = st.secrets.get("HF_FILENAME")  or os.getenv("HF_FILENAME")  or DEFAULT_FILENAME
HF_TOKEN  = st.secrets.get("HF_TOKEN")     or os.getenv("HF_TOKEN")
HF_DIR    = st.secrets.get("HF_DIR")       or os.getenv("HF_DIR")

BASE_MODEL = os.getenv("BASE_MODEL") or "microsoft/deberta-v3-base"
NUM_LABELS = int(os.getenv("NUM_LABELS") or 2)

# ─────────────────────────────────────────────────────────────────────────────
# 2) HF 체크포인트 & 모델 로드
# ─────────────────────────────────────────────────────────────────────────────
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource(show_spinner=False)
def get_ckpt_path() -> str:
    local = os.path.join("model", "prompt_guard_best.pt")
    if os.path.exists(local):
        return local
    try:
        return hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                               repo_type=REPO_TYPE, token=HF_TOKEN)
    except Exception:
        alt = "dataset" if REPO_TYPE == "model" else "model"
        return hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                               repo_type=alt, token=HF_TOKEN)

@st.cache_resource(show_spinner=False)
def load_model_tokenizer():
    # 토크나이저 (SentencePiece 필요 시 slow, 실패하면 fast)
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    except Exception:
        st.info("슬로우 토크나이저 로드 실패 → fast 토크나이저로 재시도합니다.")
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    # 완전 모델 디렉토리 사용
    if HF_DIR and os.path.isdir(HF_DIR):
        mdl = AutoModelForSequenceClassification.from_pretrained(HF_DIR)
        mdl.eval()
        return mdl, tok, 0.5, True

    # 베이스 + state_dict 주입
    mdl = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)
    thr = 0.5
    torch_loaded = False
    try:
        ckpt_path = get_ckpt_path()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        mdl.load_state_dict(state, strict=False)
        thr = float(ckpt.get("val_thr", 0.5)) if isinstance(ckpt, dict) else 0.5
        torch_loaded = True
    except Exception as e:
        st.info("체크포인트를 불러오지 못했습니다(모델 미로딩).")
        st.caption(str(e))

    mdl.eval()
    return mdl, tok, thr, torch_loaded

_cached_model = st.cache_resource(show_spinner=False)(load_model_tokenizer)

# ─────────────────────────────────────────────────────────────────────────────
# 3) 전처리 & 마침표 강건화 스코어
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(s: str) -> str:
    """NFKC 정규화 + 공백 정리"""
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

@torch.no_grad()
def score_once(mdl, tok, text: str) -> float:
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    logits = mdl(**enc).logits
    if logits.size(-1) == 1:   # 시그모이드 헤드
        return torch.sigmoid(logits)[0, 0].item()
    return torch.softmax(logits, dim=-1)[0, 1].item()  # 소프트맥스 1(악성) 확률

def robust_score(mdl, tok, text: str, method: str = "mean") -> float:
    """문장 끝의 마침표 유무에 강건: 원문 / 마침표 제거 / 마침표 강제 추가 3회 평가."""
    v1 = text
    v2 = text.rstrip(". ")
    v3 = (text.rstrip() + ".")
    scores = [score_once(mdl, tok, v) for v in (v1, v2, v3)]
    return max(scores) if method == "max" else sum(scores) / len(scores)

# ─────────────────────────────────────────────────────────────────────────────
# 4) 추론 (모델 점수만)
# ─────────────────────────────────────────────────────────────────────────────
def predict(text: str, thr_ui: float, dot_robust: bool, robust_method: str):
    text = preprocess(text)
    mdl, tok, thr_ckpt, torch_loaded = _cached_model()

    m_score = 0.0
    if torch_loaded:
        m_score = robust_score(mdl, tok, text, method=robust_method) if dot_robust else score_once(mdl, tok, text)

    thr = float(thr_ui if thr_ui is not None else thr_ckpt)
    label = "악성" if m_score >= thr else "안전"

    return {
        "점수": round(m_score, 3),
        "임계값": round(thr, 3),
        "판정": label,
        "세부": {
            "model_score": round(m_score, 3),
            "torch_loaded": bool(torch_loaded),
            "dot_robust": bool(dot_robust),
            "robust_method": robust_method,
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5) UI (OPENAI_API_KEY 세션 유지 + 🔐키 상태)
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛡️ KillSwitch AI")

# 세션 상태 초기화
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""

# 🔐 키 상태 표시
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

# 사이드바 입력/옵션
OPENAI_API_KEY = st.sidebar.text_input(
    "OPENAI_API_KEY",
    value=st.session_state.OPENAI_API_KEY,
    type="password"
)
if OPENAI_API_KEY != st.session_state.OPENAI_API_KEY:
    st.session_state.OPENAI_API_KEY = OPENAI_API_KEY

openai_model   = st.sidebar.text_input("OpenAI 모델", value="gpt-4o-mini")
thr_ui         = st.sidebar.slider("임계값(차단 기준)", 0.05, 0.95, 0.5, step=0.05)
dot_robust     = st.sidebar.checkbox("마침표 강건화(권장)", value=True)
robust_method  = st.sidebar.radio("강건화 방식", ["mean", "max"], index=0, horizontal=True)
force_call     = st.sidebar.checkbox("위험해도 GPT 호출 강행", value=False)

st.sidebar.caption(f"HF: {REPO_ID} ({REPO_TYPE}) / {FILENAME}")

# 입력 & 실행
txt = st.text_area("프롬프트", height=140, placeholder="예) 인천 맛집 알려줘")
run = st.button("분석 (GPT 호출)")

if run:
    result = predict(txt, thr_ui, dot_robust, robust_method)
    st.subheader("분석 결과  ↪️")
    st.json(result)

    st.subheader("GPT 응답")
    if not (st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or st.session_state.OPENAI_API_KEY):
        st.info("OPENAI_API_KEY가 없어 GPT 호출을 생략했습니다.")
    elif result["판정"] == "악성" and not force_call:
        st.warning("악성으로 판정되어 GPT 호출을 차단했습니다. (사이드바 '강행'을 체크하면 호출)")
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
