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
# ===== [ADD END] =====
