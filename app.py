# ===== [ADD] 허깅페이스 체크포인트 로더 =====
import os, torch
import streamlit as st
from huggingface_hub import hf_hub_download  # 여러 파일이면 snapshot_download도 가능

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

# ★ 본인 허브 리포 / 파일 경로로 바꾸세요
REPO_ID   = "your-username/killswitch-ai-checkpoints"  # 예: "beomsu/killswitch-ckpt"
REPO_TYPE = "model"      # 모델 리포면 "model", 데이터셋 리포면 "dataset"
FILENAME  = "pt/prompt_guard_best.pt"  # 허브 리포 내부 경로

@st.cache_resource(show_spinner=False)
def get_ckpt_path():
    # 1) 레포에 체크포인트를 같이 넣었으면 로컬 우선
    local_path = os.path.join("model", "prompt_guard_best.pt")
    if os.path.exists(local_path):
        return local_path
    # 2) 허브에서 단일 파일 다운로드
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type=REPO_TYPE,
        token=HF_TOKEN
    )
# ===== [USE] 체크포인트 로드 + 모델/토크나이저 준비 =====
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_MODEL = "microsoft/deberta-v3-base"   # ← 학습에 쓴 베이스로 바꾸세요 (예: xlm-roberta-base)
NUM_LABELS = 2                              # ← 클래스 수에 맞게

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    # tokenizers 빌드 에러 회피용: use_fast=False 권장
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS
    )

    ckpt_path = get_ckpt_path()  # ← 방금 만든 로더 사용
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # 키 안 맞아도 동작하게 strict=False
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        st.warning(f"state_dict mismatch → missing:{len(missing)}, unexpected:{len(unexpected)}")

    thr = float(ckpt.get("val_thr", 0.5)) if isinstance(ckpt, dict) else 0.5
    model.eval()
    return model, tokenizer, thr


def predict(text: str, max_len: int = 256):
    model, tok, thr = load_model_and_tokenizer()
    if not text.strip():
        return None

    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    with torch.no_grad():
        logits = model(**enc).logits
        # 이진로지스틱(head=1)과 다중분류(head=2 이상) 모두 대응
        if logits.size(-1) == 1:
            prob_mal = torch.sigmoid(logits)[0, 0].item()
        else:
            prob_mal = torch.softmax(logits, dim=-1)[0, 1].item()

    label = "악성" if prob_mal >= thr else "안전"
    return {"prob_mal": prob_mal, "thr": thr, "label": label}


# ===== [UI] 최소 동작 데모 =====
st.title("KillSwitch.AI Prompt Guard (Demo)")
txt = st.text_area("프롬프트 입력", height=140, placeholder="여기에 프롬프트를 붙여넣고 '분석'을 누르세요.")
col1, col2 = st.columns(2)
with col1:
    run = st.button("분석")
with col2:
    st.caption(f"허브: {REPO_ID} / 파일: {FILENAME}")

if run:
    out = predict(txt)
    if out is None:
        st.error("텍스트를 입력하세요.")
    else:
        st.success(f"결과: {out['label']}  |  p(악성)={out['prob_mal']:.3f}  /  thr={out['thr']:.3f}")

