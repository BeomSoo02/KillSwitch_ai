
import os, re, math, glob
from typing import Dict, Any
import numpy as np
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME   = os.getenv("PAIR_MODEL_NAME", "microsoft/deberta-v3-small")
MAX_LEN      = int(os.getenv("MAX_LEN", "192"))
RANK_TAU     = float(os.getenv("RANK_TAU", "0.75"))
THRESHOLD    = float(os.getenv("THRESHOLD", "0.60"))
USE_TRANSLATION = os.getenv("USE_TRANSLATION", "true").lower() == "true"

# 체크포인트 경로: 우선 환경변수, 없으면 HF Hub에서 받아오기
CKPT_PATH = os.getenv("PAIR_CKPT_PATH")
if not CKPT_PATH:
    try:
        from huggingface_hub import hf_hub_download
        repo = os.getenv("HF_REPO_ID")
        fname = os.getenv("HF_CKPT_FILENAME", "killswitch_ai_demo_zero_1.pt")
        tok = os.getenv("HF_TOKEN")
        if repo:
            CKPT_PATH = hf_hub_download(repo_id=repo, filename=fname, repo_type="model", token=tok)
    except Exception:
        CKPT_PATH = None
if not CKPT_PATH:
    hits = glob.glob("**/killswitch_ai_demo_zero_*.pt", recursive=True)
    if hits: CKPT_PATH = sorted(hits)[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

class MeanPooler(nn.Module):
    def forward(self, h, m):
        m = m.unsqueeze(-1).float()
        return (h*m).sum(1) / m.sum(1).clamp_min(1.0)

class PairScorer(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.bb = AutoModel.from_pretrained(base)
        if hasattr(self.bb.config, "use_cache"):
            try: self.bb.config.use_cache = False
            except: pass
        self.pool = MeanPooler(); self.drop = nn.Dropout(0.10)
        self.head = nn.Linear(self.bb.config.hidden_size, 1)
    def score(self, ids, msk):
        out = self.bb(input_ids=ids, attention_mask=msk)
        x = self.pool(out.last_hidden_state, msk)
        x = self.drop(x)
        return self.head(x).squeeze(-1)
