# safety_core.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
from config import REPO_ID, REVISION, MAX_LEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_hf_token() -> str | None:
    # Streamlit Cloud에서는 st.secrets를 사용하는 편이 좋지만,
    # lib 독립을 위해 여기선 환경변수만 본다. (app.py에서 secrets를 env로 주입 가능)
    return os.getenv("HF_TOKEN", None)

# Streamlit에서 캐시할 것이므로 함수 자체는 순수하게 유지
def load_model():
    hf_token = _get_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(
        REPO_ID, revision=REVISION, use_auth_token=hf_token, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        REPO_ID, revision=REVISION, use_auth_token=hf_token, trust_remote_code=False
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

@torch.no_grad()
def predict_proba(texts: List[str], tokenizer, model) -> List[float]:
    """
    returns: 악성 클래스(1)의 확률 리스트
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[:, 1]  # class-1 = 악성
    return probs.detach().cpu().tolist()

def predict_one(text: str, tokenizer, model) -> float:
    return predict_proba([text], tokenizer, model)[0]
