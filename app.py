# app.py — KillSwitch AI (모델 점수 + 끝마침표 + 의도/위험/확신도 보정, GPT 미사용)
# -----------------------------------------------------------------------------------
# ✔ 모델 점수만 사용 (규칙 최소: 점수 보정용 가벼운 패턴만)
# ✔ 끝마침표 강제 (원문이 '.'로 끝나면 추가 안 함)
# ✔ 의도(설명/실행) + 위험 단어 + 확신도(불확실 시 감쇠) 보정
# ✔ '검토필요' 중간 밴드 제공 (운영/시연 신뢰도 ↑)
# ✔ HF 체크포인트 로드(Pytorch 2.6 호환: weights_only=False)
# ✔ 토크나이저 slow 우선→실패 시 fast
# ✔ HF 연결 점검 버튼
# -----------------------------------------------------------------------------------

import os, re, time, unicodedata
from typing import List, Dict, Any
import streamlit as st

st.set_page_config(page_title="KillSwitch AI", layout="wide")

# ===== 1) 환경/시크릿 =====
DEFAULT_REPO_ID   = "cookiechips/KillSwitch_ai"   # 필요하면 교체
DEFAULT_REPO_TYPE = "model"
DEFAULT_FILENAME  = "prompt_guard_best.pt"

REPO_ID   = st.secrets.get("HF_REPO_ID")   or os.getenv("HF_REPO_ID")   or DEFAULT_REPO_ID
REPO_TYPE = st.secrets.get("HF_REPO_TYPE") or os.getenv("HF_REPO_TYPE") or DEFAULT_REPO_TYPE
FILENAME  = st.secrets.get("HF_FILENAME")  or os.getenv("HF_FILENAME")  or DEFAULT_FILENAME
HF_TOKEN  = st.secrets.get("HF_TOKEN")     or os.getenv("HF_TOKEN")
HF_DIR    = st.secrets.get("HF_DIR")       or os.getenv("HF_DIR")

BASE_MODEL = os.getenv("BASE_MODEL") or "microsoft/deberta-v3-base"
NUM_LABELS = int(os.getenv("NUM_LABELS") or 2)

# ===== 2) 모델/허브 로딩 =====
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def get_ckpt_path() -> str:
    """허브에서 .pt 체크포인트 다운로드 (repo_type 실패 시 반대 타입 재시도)."""
    local = os.path.join("model", FILENAME)
    if os.path.exists(local):
        return local
    try:
        return hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type=REPO_TYPE, token=HF_TOKEN)
    except Exception:
        alt = "dataset" if REPO_TYPE == "model" else "model"
        return hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type=alt, token=HF_TOKEN)

@st.cache_resource(show_spinner=False)
def load_model_tokenizer():
    # 토크나이저: slow 우선 → 실패 시 fast
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
        tok_info = "slow"
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        tok_info = "fast"

    # 완전 모델 디렉토리 우선
    if HF_DIR and os.path.isdir(HF_DIR):
        mdl = AutoModelForSequenceClassification.from_pretrained(HF_DIR)
        mdl.to(DEVICE).eval()
        return mdl, tok, 0.5, True, tok_info

    # 베이스 + state_dict 주입
    mdl = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)
    thr = 0.5
    torch_loaded = False
    try:
        ckpt_path = get_ckpt_path()
        # PyTorch 2.6+ 호환: weights_only=False 명시
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = mdl.load_state_dict(state, strict=False)
        if missing or unexpected:
            st.caption(f"state_dict mismatch → missing:{len(missing)}, unexpected:{len(unexpected)}")
        if isinstance(ckpt, dict) and "val_thr" in ckpt:
            try:
                thr = float(ckpt["val_thr"])
            except Exception:
                pass
        torch_loaded = True
    except Exception as e:
        st.error("체크포인트 로드 실패 — 모델 미로딩")
        st.caption(str(e))

    mdl.to(DEVICE).eval()
    return mdl, tok, thr, torch_loaded, tok_info

_cached_model = st.cache_resource(show_spinner=False)(load_model_tokenizer)

# ===== 3) 전처리 & 기본 점수 =====
def preprocess(s: str) -> str:
    """NFKC + 공백 정리 + 끝마침표 강제(이미 있으면 추가하지 않음)."""
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s.endswith(".") else (s + ".")

@torch.no_grad()
def score_once(mdl, tok, text: str) -> float:
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    for k in enc:
        enc[k] = enc[k].to(DEVICE)
    logits = mdl(**enc).logits
    if logits.size(-1) == 1:   # 시그모이드 헤드
        return torch.sigmoid(logits)[0, 0].item()
    return torch.softmax(logits, dim=-1)[0, 1].item()  # 소프트맥스 1(악성) 확률

# ===== 4) 점수 보정(의도/위험/확신도) =====
HIGH_RISK_KEYWORDS = [
    r"폭탄", r"폭발물", r"폭발", r"무기", r"테러",
    r"디도스", r"\bDDoS\b", r"랜섬웨어", r"살인", r"살해",
    r"\bbomb\b", r"explosive", r"weapon", r"kill\W"
]
ACTION_PATTERNS = [
    r"만들어줘", r"만드는\s*방법", r"실행해줘", r"작성해줘",
    r"스크립트\s*만들어", r"how to", r"\bmake\b", r"\bbuild\b"
]
INFO_PATTERNS = [
    r"설명해줘", r"무엇인지", r"개념", r"정의", r"이론적으로",
    r"what is", r"\bexplain\b", r"\bdefinition\b"
]

def any_match(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)

def uncertainty_from_prob(p: float) -> float:
    """p=0/1에서 0, p=0.5에서 1 → 0.75 이상이면 불확실로 간주."""
    return 1.0 - abs(p - 0.5) * 2.0

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def post_adjust(original_text: str, base_prob: float,
                action_boost: float = 0.15,
                info_penalty: float = -0.15,
                highrisk_boost: float = 0.25,
                uncertainty_penalty: float = -0.10) -> (float, float, list):
    """
    base_prob에 보정치 합산 → 최종 점수와 보정 합(adj), 이유 리스트 반환
    """
    reasons = []
    adj = 0.0

    if any_match(HIGH_RISK_KEYWORDS, original_text):
        adj += highrisk_boost; reasons.append("high-risk-keyword")
    if any_match(ACTION_PATTERNS, original_text):
        adj += action_boost; reasons.append("action-verb")
    if any_match(INFO_PATTERNS, original_text):
        adj += info_penalty; reasons.append("info-verb")

    unc = uncertainty_from_prob(base_prob)
    if unc > 0.75:
        adj += uncertainty_penalty; reasons.append("low-confidence")

    final = clamp01(base_prob + adj)
    return final, adj, reasons

# ===== 5) 예측 파이프라인 =====
def predict_with_postproc(text: str, ui_thr: float,
                          low_review: float = 0.40, high_review: float = 0.70) -> Dict[str, Any]:
    """
    최종 판정:
      - final_score >= ui_thr  → '악성'
      - low_review <= score < high_review → '검토필요'
      - else → '안전'
    """
    mdl, tok, thr_ckpt, torch_loaded, tok_info = _cached_model()
    t0 = time.time()
    txt_proc = preprocess(text)

    base = score_once(mdl, tok, txt_proc) if torch_loaded else 0.0
    final, adj, reasons = post_adjust(text, base)

    if final >= ui_thr:
        label, base_reason = "악성", "model-high"
    elif low_review <= final < high_review:
        label, base_reason = "검토필요", "model-mid"
    else:
        label, base_reason = "안전", "model-low"

    if not reasons:
        reasons = [base_reason]

    return {
        "점수": round(final, 3),
        "원점수": round(base, 3),
        "임계값": round(ui_thr, 3),
        "판정": label,
        "근거": reasons,
        "세부": {
            "보정합": round(adj, 3),
            "torch_loaded": bool(torch_loaded),
            "device": str(DEVICE),
            "tokenizer": tok_info,
        },
        "_elapsed_s": round(time.time() - t0, 2),
    }

# ===== 6) UI =====
st.title("🛡️ KillSwitch AI — Heuristic Post-Processing (No GPT)")

# 사이드바 옵션
thr_ui      = st.sidebar.slider("임계값(악성 판정 컷)", 0.30, 0.95, 0.50, step=0.05)
low_review  = st.sidebar.slider("검토필요(하한)",        0.10, 0.90, 0.40, step=0.05)
high_review = st.sidebar.slider("검토필요(상한)",        0.20, 0.95, 0.70, step=0.05)

# HF 연결 점검
st.sidebar.caption(f"HF: {REPO_ID} ({REPO_TYPE}) / {FILENAME}")
if st.sidebar.button("HF 연결 점검"):
    try:
        p = get_ckpt_path()
        size_mb = os.path.getsize(p) / 1_048_576
        st.sidebar.success(f"OK: {os.path.basename(p)} · {size_mb:.1f} MB")
    except Exception as e:
        st.sidebar.error("다운로드 실패")
        st.sidebar.exception(e)

# 입력 & 실행
txt = st.text_area("프롬프트", height=140, placeholder="예) 인천 맛집 알려줘")
if st.button("분석 실행"):
    if not (txt and txt.strip()):
        st.warning("텍스트를 입력하세요.")
    else:
        with st.spinner("분석 중..."):
            result = predict_with_postproc(txt, thr_ui, low_review, high_review)
        st.success(f"분석 완료 ({result['_elapsed_s']:.2f}s)")
        st.subheader("분석 결과  ↪️")
        # 판정 이모지
        emoji = "✅" if result["판정"] == "안전" else ("🟡" if result["판정"] == "검토필요" else "⛔")
        st.write(f"**판정:** {result['판정']} {emoji}")
        st.write(f"**점수:** {result['점수']} (원점수: {result['원점수']}, 임계값: {result['임계값']})")
        st.write(f"**근거:** {', '.join(result['근거'])}")
        st.json({k: v for k, v in result.items() if not k.startswith("_")})
