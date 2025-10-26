# app.py — KillSwitch AI · Streamlit 데모 (세션 유지 패치 적용)
# ----------------------------------------------------------------------
# ✔ HF 체크포인트 로딩
# ✔ 규칙(rule) + 모델 점수 융합
# ✔ 사이드바 OPENAI_API_KEY 입력(세션 유지)
# ✔ 필요 시 GPT 호출
# ----------------------------------------------------------------------

import os, time, re, json
import streamlit as st

# 0) 페이지 설정
st.set_page_config(page_title="KillSwitch AI", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 1) 환경/시크릿 설정
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
    except:
        alt = "dataset" if REPO_TYPE == "model" else "model"
        return hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                               repo_type=alt, token=HF_TOKEN)

@st.cache_resource(show_spinner=False)
def load_model_tokenizer():
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    except:
        st.info("fast tokenizer 로 재시도")
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    if HF_DIR and os.path.isdir(HF_DIR):
        mdl = AutoModelForSequenceClassification.from_pretrained(HF_DIR)
        mdl.eval()
        return mdl, tok, 0.5, True

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
        st.info("체크포인트 미적용(규칙만)")
        st.caption(str(e))

    mdl.eval()
    return mdl, tok, thr, torch_loaded

_cached_model = st.cache_resource(show_spinner=False)(load_model_tokenizer)

# ─────────────────────────────────────────────────────────────────────────────
# 3) 규칙 기반 탐지
# ─────────────────────────────────────────────────────────────────────────────
BAD_PATTERNS = [
    r"자살|자해|극단적\s*선택", r"폭탄|총기|살인|테러",
    r"마약|필로폰|코카인", r"해킹|디도스|랜섬웨어|취약점",
    r"비밀번호|OTP|백도어|주민등록번호|신용카드",
    r"피싱|phishing",
    r"씨발|ㅅㅂ|개새끼|병신|지랄",
    r"kill|murder"
]
BAD_RE = re.compile("|".join(BAD_PATTERNS), re.I)

def rule_detect(text: str):
    found = sorted(set([m.group(0) for m in BAD_RE.finditer(text)]))
    score = min(1.0, len(found)*0.4) if found else 0.0
    return score, found

# ─────────────────────────────────────────────────────────────────────────────
# 4) 추론 & 융합
# ─────────────────────────────────────────────────────────────────────────────
def predict(text: str, thr_ui: float):
    mdl, tok, thr_ckpt, torch_loaded = _cached_model()
    r_score, r_keys = rule_detect(text)

    m_score = 0.0
    if torch_loaded:
        enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = mdl(**enc).logits
            if logits.size(-1)==1:
                m_score = torch.sigmoid(logits)[0,0].item()
            else:
                m_score = torch.softmax(logits, dim=-1)[0,1].item()

    score = max(r_score, m_score)
    thr = float(thr_ui if thr_ui is not None else thr_ckpt)
    label = "악성" if score >= thr else "안전"

    return {
        "점수": round(score,3),
        "임계값": round(thr,3),
        "판정": label,
        "키워드": r_keys or ["-"],
        "세부": {
            "rule_score": round(r_score,3),
            "model_score": round(m_score,3),
            "torch_loaded": torch_loaded,
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5) UI (OPENAI_API_KEY 세션 유지)
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛡️ KillSwitch AI")

# 세션 상태 초기화
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""

OPENAI_API_KEY = st.sidebar.text_input(
    "OPENAI_API_KEY", 
    value=st.session_state.OPENAI_API_KEY,
    type="password"
)
if OPENAI_API_KEY != st.session_state.OPENAI_API_KEY:
    st.session_state.OPENAI_API_KEY = OPENAI_API_KEY

openai_model   = st.sidebar.text_input("OpenAI 모델", value="gpt-4o-mini")
thr_ui         = st.sidebar.slider("임계값(차단 기준)", 0.05, 0.95, 0.35, step=0.05)
force_call     = st.sidebar.checkbox("위험해도 GPT 호출 강행", value=False)

st.sidebar.caption(f"HF: {REPO_ID} ({REPO_TYPE}) / {FILENAME}")

txt = st.text_area("프롬프트", height=140, placeholder="예) 인천 맛집 알려줘")
run = st.button("분석 (GPT 호출)")

if run:
    result = predict(txt, thr_ui)
    st.subheader("분석 결과  ↪️")
    st.json(result)

    st.subheader("GPT 응답")
    if not st.session_state.OPENAI_API_KEY:
        st.info("OPENAI_API_KEY가 없어 GPT 호출 생략")
    elif result["판정"] == "악성" and not force_call:
        st.warning("악성 판정 → GPT 호출 차단 (강행 ON 시 가능)")
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)
            rsp = client.responses.create(
                model=openai_model,
                input=[
                    {"role": "system", "content": "If harmful, refuse."},
                    {"role": "user", "content": txt}
                ],
            )
            st.write(rsp.output_text)
        except Exception as e:
            st.error(f"GPT 호출 오류: {type(e).__name__}: {e}")
