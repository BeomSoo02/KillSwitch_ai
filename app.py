# app.py — KillSwitch AI (모델 점수만 + robust mean + 이중 임계값 + LLM 디모션 옵션)
# ------------------------------------------------------------------------------------

import os, re, time, unicodedata
from typing import Optional
import streamlit as st

st.set_page_config(page_title="KillSwitch AI", layout="wide")

# ===== 1) 환경/시크릿 =====
DEFAULT_REPO_ID   = "cookiechips/KillSwitch_ai"   # 필요시 교체
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
    # 로컬 포함 시 우선
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

    # 완전 모델 디렉토리
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

# ===== 3) 전처리 & robust mean 점수 =====
# 가벼운 정규화(룰 아님): zero-width 제거, 구두점 평탄화, 공백 정리
ZWSP = "".join([chr(c) for c in [0x200B,0x200C,0x200D,0xFEFF]])
ZWSP_RE = re.compile(f"[{re.escape(ZWSP)}]")

def normalize_lite(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = ZWSP_RE.sub("", s)
    s = re.sub(r"[·•●]+", ".", s)        # 점 유사문자 단순화
    s = re.sub(r"[!！]{2,}", "!", s)      # 연속 느낌표 축소
    s = re.sub(r"[?？]{2,}", "?", s)      # 연속 물음표 축소
    s = re.sub(r"\s+", " ", s).strip()
    return s

@torch.no_grad()
def score_once(mdl, tok, text: str) -> float:
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    for k in enc:
        enc[k] = enc[k].to(DEVICE)
    logits = mdl(**enc).logits
    if logits.size(-1) == 1:   # 시그모이드 헤드
        return torch.sigmoid(logits)[0, 0].item()
    return torch.softmax(logits, dim=-1)[0, 1].item()  # 소프트맥스 1(악성) 확률

def robust_mean_score(mdl, tok, text: str) -> float:
    t = normalize_lite(text)
    v1 = t
    v2 = t.rstrip(". ")
    v3 = (t.rstrip() + ".")
    scores = [score_once(mdl, tok, v) for v in (v1, v2, v3)]
    return sum(scores) / len(scores)

# ===== 4) LLM 세컨드 오피니언 (중간대역만, 오탐 디모션 전용) =====
def llm_safety_check(text: str, api_key: str, model_name: str = "gpt-4o-mini") -> str:
    """
    반환: "SAFE" | "UNSAFE" | "UNKNOWN"
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Classify the user's message as SAFE or UNSAFE for safety policy.\n"
            "Return exactly one word: SAFE or UNSAFE.\n\n"
            f"User message:\n{text}"
        )
        rsp = client.responses.create(
            model=model_name,
            input=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=5,
        )
        out = (rsp.output_text or "").strip().upper()
        if "UNSAFE" in out:
            return "UNSAFE"
        if "SAFE" in out:
            return "SAFE"
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"

# ===== 5) 최종 판정: 이중 임계값 + LLM(옵션, 디모션 전용) =====
LOW_T  = 0.60
HIGH_T = 0.85

def predict_fused(text: str, ui_thr: float, use_llm_demotion: bool,
                  api_key: Optional[str], model_name: str):
    mdl, tok, thr_ckpt, torch_loaded, tok_info = _cached_model()
    t0 = time.time()

    s_local = robust_mean_score(mdl, tok, text) if torch_loaded else 0.0

    # 고득점: 즉시 악성
    if s_local >= HIGH_T:
        return {
            "점수": round(s_local,3),
            "임계값": round(ui_thr,3),
            "판정": "악성",
            "근거": "local-high",
            "세부": {"torch_loaded": bool(torch_loaded), "device": str(DEVICE), "tokenizer": tok_info},
            "_elapsed_s": round(time.time() - t0, 2),
        }

    # 저득점: 즉시 안전
    if s_local < LOW_T:
        return {
            "점수": round(s_local,3),
            "임계값": round(ui_thr,3),
            "판정": "안전",
            "근거": "local-low",
            "세부": {"torch_loaded": bool(torch_loaded), "device": str(DEVICE), "tokenizer": tok_info},
            "_elapsed_s": round(time.time() - t0, 2),
        }

    # 중간대역: LLM 디모션(오탐 줄이기) — SAFE면 안전으로 내림
    verdict = "skipped"
    if use_llm_demotion and api_key:
        v = llm_safety_check(text, api_key, model_name)
        verdict = v.lower()
        if v == "SAFE":
            return {
                "점수": round(s_local,3),
                "임계값": round(ui_thr,3),
                "판정": "안전",
                "근거": "llm-safe",
                "세부": {"torch_loaded": bool(torch_loaded), "device": str(DEVICE), "tokenizer": tok_info},
                "_elapsed_s": round(time.time() - t0, 2),
            }

    # LLM 미사용/UNKNOWN/UNSAFE → 로컬 임계값으로 최종 판정
    final = "악성" if s_local >= ui_thr else "안전"
    return {
        "점수": round(s_local,3),
        "임계값": round(ui_thr,3),
        "판정": final,
        "근거": verdict if verdict != "skipped" else "local-mid",
        "세부": {"torch_loaded": bool(torch_loaded), "device": str(DEVICE), "tokenizer": tok_info},
        "_elapsed_s": round(time.time() - t0, 2),
    }

# ===== 6) UI =====
st.title("🛡️ KillSwitch AI")

# 세션 상태(키)
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""

# 🔐 키 상태
with st.sidebar.expander("🔐 키 상태"):
    key_from_secrets = bool(st.secrets.get("OPENAI_API_KEY"))
    key_from_env     = bool(os.getenv("OPENAI_API_KEY"))
    key_from_session = bool(st.session_state.get("OPENAI_API_KEY"))
    key_ok = key_from_secrets or key_from_env or key_from_session
    st.write("OpenAI Key:", "✅ 감지됨" if key_ok else "❌ 없음")
    st.caption(
        f"• Secrets: {'✅' if key_from_secrets else '—'}   "
        f"• Env: {'✅' if key_from_env else '—'}   "
        f"• Session: {'✅' if key_from_session else '—'}"
    )

# 사이드바 옵션
OPENAI_API_KEY = st.sidebar.text_input(
    "OPENAI_API_KEY",
    value=st.session_state.OPENAI_API_KEY,
    type="password"
)
if OPENAI_API_KEY != st.session_state.OPENAI_API_KEY:
    st.session_state.OPENAI_API_KEY = OPENAI_API_KEY

thr_ui            = st.sidebar.slider("임계값(최종 판정)", 0.50, 0.95, 0.75, step=0.05)
use_llm_demotion  = st.sidebar.checkbox("오탐 줄이기: LLM 세컨드 오피니언(중간대역만)", value=True)
llm_model_name    = st.sidebar.text_input("LLM 모델(디모션용)", value="gpt-4o-mini")
gen_model_answer  = st.sidebar.checkbox("원문에 대한 GPT 답변 생성", value=False)

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

# 메인 입력 & 실행
txt = st.text_area("프롬프트", height=140, placeholder="예) 인천 맛집 알려줘")
if st.button("분석 실행"):
    if not (txt and txt.strip()):
        st.warning("텍스트를 입력하세요.")
    else:
        api_key = (st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or st.session_state.OPENAI_API_KEY)
        with st.spinner("분석 중..."):
            result = predict_fused(txt, thr_ui, use_llm_demotion, api_key if key_ok else None, llm_model_name)
        st.success(f"분석 완료 ({result['_elapsed_s']:.2f}s)")
        st.subheader("분석 결과  ↪️")
        st.json({k: v for k, v in result.items() if not k.startswith("_")})

        # (옵션) 원문에 대한 GPT 답변 생성
        if gen_model_answer:
            st.subheader("GPT 응답")
            if not key_ok:
                st.info("OPENAI_API_KEY가 없어 GPT 호출을 생략했습니다.")
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    rsp = client.responses.create(
                        model=llm_model_name,
                        input=[
                            {"role": "system",
                             "content": "You are a helpful assistant. If the user's prompt is clearly harmful or illegal, politely refuse and provide safe alternatives."},
                            {"role": "user", "content": txt},
                        ],
                        temperature=0.3,
                        top_p=1.0,
                    )
                    st.write(rsp.output_text)
                except Exception as e:
                    st.error(f"GPT 호출 오류: {type(e).__name__}: {e}")
