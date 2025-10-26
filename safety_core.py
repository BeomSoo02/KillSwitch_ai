# safety_core.py
import os, torch
import streamlit as st
from typing import Optional
from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_MODEL = "microsoft/deberta-v3-base"   # 우리 베이스
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key, default)
    return v if (v is not None and str(v).strip() != "") else default

def _pick_ckpt(repo_id: str, revision: str) -> str:
    """리포 안에서 가중치 파일명 자동 탐색 (.safetensors > .bin > .pt)"""
    api = HfApi()
    files = [f.rfilename for f in api.list_repo_files(repo_id=repo_id, revision=revision)]
    for ext in (".safetensors", ".bin", ".pt"):
        for f in files:
            if f.lower().endswith(ext):
                return f
    raise FileNotFoundError("리포에서 모델 가중치(.safetensors/.bin/.pt)를 찾지 못했습니다.")

@st.cache_resource(show_spinner=True)
def load_model():
    repo_id  = _env("HF_REPO_ID")
    ckpt_nm  = _env("HF_FILENAME")   # 없으면 자동 탐색
    revision = _env("HF_REVISION", "main")

    if not repo_id:
        st.error("환경변수 HF_REPO_ID 가 비어있습니다 (Streamlit Secrets 확인).")
        raise RuntimeError("HF_REPO_ID missing")

    # 1) 토크나이저/베이스 모델 로드 (Transformers 포맷이 아니므로 base에서)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    # 2) 체크포인트 파일명 결정 및 다운로드 (공개 리포라 token 불필요)
    if not ckpt_nm:
        ckpt_nm = _pick_ckpt(repo_id, revision)

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_nm, revision=revision)

    # 3) state_dict 로드 & 주입
    state = None
    try:
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(ckpt_path)
        else:
            state = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        st.error(f"체크포인트 로드 실패: {e}")
        raise

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        st.warning(f"[state_dict] missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(DEVICE).eval()
    return tokenizer, model

@torch.no_grad()
def predict_proba(texts, tokenizer, model, max_len: int = 256):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
    return probs.cpu().tolist()

def predict_one(text, tokenizer, model, max_len: int = 256):
    return predict_proba([text], tokenizer, model, max_len=max_len)[0]
