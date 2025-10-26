import os, io, torch
import streamlit as st
from typing import Optional
from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import REPO_ID, REVISION, BASE_MODEL, MAX_LEN, HF_CKPT_FILENAME

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN")

def _pick_ckpt_file(repo_id: str, revision: str, token: Optional[str]) -> str:
    """리포 안에서 .safetensors/.bin/.pt 중 하나를 자동 선택."""
    api = HfApi()
    files = [f.rfilename for f in api.list_repo_files(repo_id=repo_id, revision=revision, token=token)]
    # 우선순위: safetensors > bin > pt
    for ext in (".safetensors", ".bin", ".pt"):
        for f in files:
            if f.lower().endswith(ext):
                return f
    raise FileNotFoundError("모델 가중치 파일(.safetensors/.bin/.pt)을 리포에서 찾지 못했습니다.")

def load_model():
    token = _hf_token()

    # 1) 토크나이저: HF 리포에 없다고 가정하고 BASE_MODEL에서 로드
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_auth_token=token, trust_remote_code=False)

    # 2) 베이스 모델 골격 준비 (num_labels=2 가정)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, use_auth_token=token, trust_remote_code=False
    )

    # 3) 리포에서 체크포인트 파일 선택/다운로드
    ckpt_name = HF_CKPT_FILENAME or _pick_ckpt_file(REPO_ID, REVISION, token)
    ckpt_path = hf_hub_download(
        repo_id=REPO_ID, filename=ckpt_name, revision=REVISION, use_auth_token=token
    )

    # 4) state_dict 로드 & 주입
    state = None
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(ckpt_path)
    else:
        state = torch.load(ckpt_path, map_location="cpu")

    # 키가 100% 일치하지 않을 수 있으므로 strict=False 권장(분류 헤드 등 차이 허용)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        st.warning(f"[state_dict] missing={len(missing)}, unexpected={len(unexpected)}")

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
