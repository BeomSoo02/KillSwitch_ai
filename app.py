# app.py — KillSwitch AI (강제 trailing-dot + 모델 점수만 + 키 상태)
# -------------------------------------------------------------------

import os, re, unicodedata, time
import streamlit as st

st.set_page_config(page_title="KillSwitch AI", layout="wide")

# ✅ 환경/시크릿 설정
DEFAULT_REPO_ID   = "cookiechips/KillSwitch_ai"
DEFAULT_REPO_TYPE = "model"
DEFAULT_FILENAME  = "prompt_guard_best.pt"

REPO_ID   = st.secrets.get("HF_REPO_ID")   or os.getenv("HF_REPO_ID")   or DEFAULT_REPO_ID
REPO_TYPE = st.secrets.get("HF_REPO_TYPE") or os.getenv("HF_REPO_TYPE") or DEFAULT_REPO_TYPE
FILENAME  = "prompt_guard_best.pt"
HF_TOKEN  = st.secrets.get("HF_TOKEN")     or os.getenv("HF_TOKEN")
HF_DIR    = st.secrets.get("HF_DIR")

BASE_MODEL = "microsoft/deberta-v3-base"
NUM_LABELS = 2

# ✅ 모델 로드
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def get_ckpt_path():
    try:
        return hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                               repo_type=REPO_TYPE, token=HF_TOKEN)
    except:
        return hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                               repo_type="dataset", token=HF_TOKEN)

@st.cache_resource
def load_model_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

    mdl = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)
    try:
        ckpt = torch.load(get_ckpt_path(), map_location="cpu")
        mdl.load_state_dict(ckpt.get("model", ckpt), strict=False)
    except Exception as e:
        st.error("체크포인트 로드 실패 — 모델 미로딩")
        st.caption(str(e))

    return mdl.to(DEVICE).eval(), tok

mdl, tok = load_model_tokenizer()

# ✅ 전처리 + 끝마침표 강제 only
def preprocess(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s.endswith(".") else s + "."

@torch.no_grad()
def score(text):
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    logits = mdl(**{k: v.to(DEVICE) for k, v in enc.items()}).logits
    if logits.size(-1) == 1:
        return torch.sigmoid(logits)[0,0].item()
    return torch.softmax(logits, dim=-1)[0,1].item()

def predict(text, thr):
    text = preprocess(text)
    t0 = time.time()
    s = score(text)
    label = "악성" if s >= thr else "안전"
    return {
        "점수": round(s,3),
        "임계값": round(thr,3),
        "판정": label,
        "_elapsed_s": round(time.time() - t0, 2),
    }

# ✅ UI — 키 상태 / 옵션
st.title("🛡️ KillSwitch AI")

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""

with st.sidebar.expander("🔐 키 상태"):
    key_ok = any([
        bool(st.secrets.get("OPENAI_API_KEY")),
        bool(os.getenv("OPENAI_API_KEY")),
        bool(st.session_state.get("OPENAI_API_KEY"))
    ])
    st.write("OpenAI Key:", "✅ 감지됨" if key_ok else "❌ 없음")

OPENAI_API_KEY = st.sidebar.text_input(
    "OPENAI_API_KEY", value=st.session_state.OPENAI_API_KEY, type="password"
)
st.session_state.OPENAI_API_KEY = OPENAI_API_KEY

openai_model = st.sidebar.text_input("OpenAI 모델", value="gpt-4o-mini")
thr_ui = st.sidebar.slider("임계값(차단 기준)", 0.05, 0.95, 0.5, step=0.05)
force_call = st.sidebar.checkbox("위험해도 GPT 호출 강행")

st.sidebar.caption(f"HF: {REPO_ID} / {FILENAME}")
if st.sidebar.button("HF 연결 점검"):
    try:
        p = get_ckpt_path()
        st.sidebar.success(f"OK {os.path.basename(p)}")
    except Exception as e:
        st.sidebar.error("실패")
        st.sidebar.caption(str(e))

# ✅ 입력/실행
txt = st.text_area("프롬프트", height=140)
if st.button("분석 (GPT 호출)"):
    result = predict(txt, thr_ui)
    st.subheader("분석 결과")
    st.json(result)

    st.subheader("GPT 응답")
    if not st.session_state.OPENAI_API_KEY:
        st.info("API KEY 없음 → GPT 호출 생략")
    elif result["판정"] == "악성" and not force_call:
        st.warning("악성 판정 → GPT 호출 차단됨")
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)
            rsp = client.responses.create(
                model=openai_model,
                input=[{"role":"user","content":txt}],
                temperature=0.3
            )
            st.write(rsp.output_text)
        except Exception as e:
            st.error(str(e))
