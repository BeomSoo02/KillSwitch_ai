import os, torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download, HfApi
from config import REPO_ID, REVISION, MAX_LEN, BASE_MODEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _hf_token():
    # Streamlit Cloud에서는 st.secrets에 HF_TOKEN을 넣어두면 app.py에서 env로 주입됨
    return os.getenv("HF_TOKEN")

def _exists_revision(repo_id: str, revision: str, token: str | None):
    try:
        info = HfApi().model_info(repo_id, revision=revision, token=token)
        return True, f"ok ({info.sha[:7]})"
    except Exception as e:
        return False, str(e)

def load_model():
    token = _hf_token()

    # 0) 리비전 확인 (원인 즉시 노출)
    ok, msg = _exists_revision(REPO_ID, REVISION, token)
    if not ok:
        st.error(f"[HF] REVISION 확인 실패: {msg}")
        st.info("config.py의 REVISION을 'main'으로 바꾸거나, 실제 태그/커밋으로 설정하세요.")
        # 그래도 진행 시도 (main으로 폴백)
        rev = "main"
    else:
        rev = REVISION

    # 1) 토크나이저 로딩 (리포에 없으면 BASE_MODEL로 폴백)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            REPO_ID, revision=rev, use_auth_token=token, trust_remote_code=False
        )
    except Exception as e:
        st.warning(f"[HF] 리포에서 토크나이저 로드 실패 → BASE_MODEL로 폴백: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, use_auth_token=token, trust_remote_code=False
        )

    # 2) 모델 로딩 (Transformers 형식 가정)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            REPO_ID, revision=rev, use_auth_token=token, trust_remote_code=False
        )
    except Exception as e:
        # 만약 repo에 safetensors만 있고 포맷이 어긋나면, 직접 파일 받아 로드하는 경로를 추가
        st.error(f"[HF] 모델 로드 실패: {e}")
        raise

    model.to(DEVICE).eval()
    return tokenizer, model

@torch.no_grad()
def predict_proba(texts, tokenizer, model):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
    return probs.detach().cpu().tolist()

def predict_one(text, tokenizer, model):
    return predict_proba([text], tokenizer, model)[0]
