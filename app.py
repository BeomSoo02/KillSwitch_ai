# app.py — KillSwitch AI (최종본)
# - 모델 점수 + 메타가중치(실행/설명/위험단어) + softmax 확신도(gap) 감쇠
# - GPT 응답 UI 상단 배치 / 단일 버튼(분석)
# - 속도 최적화: fast tokenizer, FP16(CUDA), max_length 디폴트 224
# --------------------------------------------------------------------------------
import os, re, time, unicodedata
import streamlit as st

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

st.set_page_config(page_title="KillSwitch AI", layout="wide")

# ===== 1) 환경/시크릿 =====
DEFAULT_REPO_ID   = "cookiechips/KillSwitch_ai"
DEFAULT_REPO_TYPE = "model"
DEFAULT_FILENAME  = "prompt_guard_best.pt"

REPO_ID   = st.secrets.get("HF_REPO_ID")   or os.getenv("HF_REPO_ID")   or DEFAULT_REPO_ID
REPO_TYPE = st.secrets.get("HF_REPO_TYPE") or os.getenv("HF_REPO_TYPE") or DEFAULT_REPO_TYPE
FILENAME  = st.secrets.get("HF_FILENAME")  or os.getenv("HF_FILENAME")  or DEFAULT_FILENAME
HF_TOKEN  = st.secrets.get("HF_TOKEN")     or os.getenv("HF_TOKEN")
HF_DIR    = st.secrets.get("HF_DIR")       or os.getenv("HF_DIR")

BASE_MODEL = "microsoft/deberta-v3-base"
NUM_LABELS = 2

W_ACTION = 0.15
W_INFO   = -0.15
W_DOMAIN = 0.25
GAP_THR  = 0.10
W_UNCERT = -0.10

# ===== 2) 모델 로딩 =====
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)

@st.cache_resource
def get_ckpt_path():
    local = os.path.join("model", FILENAME)
    if os.path.exists(local):
        return local
    try:
        return hf_hub_download(REPO_ID, FILENAME, REPO_TYPE, HF_TOKEN)
    except:
        alt = "dataset" if REPO_TYPE=="model" else "model"
        return hf_hub_download(REPO_ID, FILENAME, alt, HF_TOKEN)

@st.cache_resource
def load_model_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)
    thr = 0.5

    try:
        ckpt = torch.load(get_ckpt_path(), map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        mdl.load_state_dict(state, strict=False)
        if isinstance(ckpt, dict) and "val_thr" in ckpt:
            thr = float(ckpt["val_thr"])
    except Exception as e:
        st.error(f"체크포인트 로드 오류: {e}")

    mdl.to(DEVICE).eval()
    if DEVICE.type == "cuda":
        mdl.half()

    return mdl, tok, thr

_cached_model = load_model_tokenizer

# ===== 3) 전처리 & 모델 추론 =====
def preprocess(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s.endswith(".") else s+"."

def _softmax_two(logits):
    probs = torch.softmax(logits, dim=-1)[0]
    p0, p1 = probs[0], probs[1]
    gap = abs(p1 - p0)
    return p0.item(), p1.item(), gap.item()

@torch.inference_mode()
def model_forward(mdl, tok, text, max_len):
    enc = tok(text, return_tensors="pt", truncation=True, padding=False, max_length=max_len)
    seq_len = enc["input_ids"].shape[-1]
    for k in enc: enc[k] = enc[k].to(DEVICE)
    out = mdl(**enc)
    p0, p1, gap = _softmax_two(out.logits)
    return p1, gap, seq_len

# ===== 4) 메타 가중치 =====
ACTION_PAT = re.compile(r"(해줘|만들어|구현|코드|스크립트|자동화|실행|공격|다운로드|install|inject|exploit)", re.IGNORECASE)
INFO_PAT = re.compile(r"(의미|정의|원리|이유|설명|알려줘|요약|가이드|무엇)", re.IGNORECASE)
DANGER_WORDS = ["폭탄","ddos","해킹","무기","랜섬웨어","백도어","악성코드","피싱","cvv","비밀번호","탈취"]

def detect_meta(t):
    t=t.lower()
    return bool(ACTION_PAT.search(t)), bool(INFO_PAT.search(t)), any(w in t for w in DANGER_WORDS)

def apply_weights(p1,g,a,i,d):
    score=p1
    if a: score+=W_ACTION
    if i: score+=W_INFO
    if d: score+=W_DOMAIN
    if g<GAP_THR: score+=W_UNCERT
    return max(0,min(1,score))

# ===== 5) 예측 파이프라인 =====
def predict(text, thr_ui, max_len):
    text = preprocess(text)
    mdl,tok,thr = _cached_model()
    t0=time.time()

    p1,gap,seq_len = model_forward(mdl,tok,text,max_len)
    act,info,danger = detect_meta(text)
    adj = apply_weights(p1,gap,act,info,danger)
    label = "악성" if adj>=thr else "안전"

    return {
        "판정":label, "원점수":round(p1,3),"조정":round(adj,3),
        "임계":thr, "gap":round(gap,3),
        "act":act,"info":info,"danger":danger,
        "seq_len":seq_len,"max_len":max_len,
        "_elapsed":round(time.time()-t0,2)
    }

# ===== 6) UI =====
st.title("🛡️ KillSwitch AI")

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY=""

OPENAI_API_KEY = st.sidebar.text_input("OPENAI_API_KEY",st.session_state.OPENAI_API_KEY,type="password")
st.session_state.OPENAI_API_KEY=OPENAI_API_KEY

thr_ui = st.sidebar.slider("임계값",0.05,0.95,0.70,0.05)
max_len_ui = st.sidebar.slider("max_length",128,256,224,32)
force_call = st.sidebar.checkbox("위험해도 GPT 강행",False)
openai_model = st.sidebar.text_input("GPT 모델","gpt-4o-mini")

txt = st.text_area("프롬프트",height=140,placeholder="예) 인천 맛집 알려줘")

@st.cache_resource
def get_openai_client():
    from openai import OpenAI
    k=st.session_state.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=k) if k else None

if st.button("분석"):
    if not txt.strip():
        st.warning("텍스트를 입력하세요.")
    else:
        with st.spinner("분석 중..."):
            r=predict(txt,thr_ui,max_len_ui)

        st.success(f"분석 완료 ({r['_elapsed']}s)")
        st.markdown("### ✅ 요약")

        c1,c2,c3,c4,c5=st.columns([1.2,1,1,1,1])
        with c1: st.metric("판정",r["판정"])
        with c2: st.metric("원점수",r["원점수"])
        with c3: st.metric("조정",r["조정"])
        with c4: st.metric("임계",r["임계"])
        with c5: st.metric("확신도",r["gap"])

        if r["seq_len"]>=r["max_len"]:
            st.caption("⚠️ 입력이 너무 길어 일부가 잘렸을 수 있어요.")

        st.markdown("### 🤖 GPT 응답")
        client=get_openai_client()

        if client is None:
            st.info("🔑 API Key 없음 → GPT 미호출")
        elif r["판정"]=="악성" and not force_call:
            st.warning("🛑 악성 → GPT 차단(강행 옵션 활성 시 가능)")
        else:
            try:
                rsp=client.responses.create(
                    model=openai_model,
                    input=[
                        {"role":"system","content":"불법/유해 요청은 정중히 거절하고 안전한 방향 안내"},
                        {"role":"user","content":txt},
                    ],
                    temperature=0.3
                )
                st.write(rsp.output_text)
            except Exception as e:
                st.error(f"GPT 오류: {e}")

        with st.expander("🔍 세부/근거"):
            st.write({
                "실행형":r["act"],"설명형":r["info"],
                "위험단어":r["danger"],
                "seq_len":r["seq_len"],"max_len":r["max_len"]
            })
