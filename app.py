# app.py — KillSwitch AI · Streamlit (모델 점수만, 안정화 패치 포함)
# ----------------------------------------------------------------------
# ✔ 공개 HF 허브 체크포인트(.pt) 자동 다운로드 → state_dict 주입
# ✔ 규칙(rule) 제거 — 모델 확률만으로 판정
# ✔ 안정화: 입력 정규화(NFKC 등), 토크나이저 고정(use_fast=False), strict=True 로딩 시도
# ✔ 선택: 마침표 강건화(원문/마침표 유무 앙상블)
# ✔ 사이드바: 임계값, HF 연결 점검, 마침표 강건화 on/off
# ✔ HF_DIR 제공 시 from_pretrained 디렉토리 직접 로드(완전 모델 형식)
# ----------------------------------------------------------------------

import os, time, re, unicodedata
import streamlit as st

st.set_page_config(page_title="KillSwitch AI", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 1) 환경/시크릿 설정
#    (Streamlit Secrets/Env 로 덮어쓰기 가능)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REPO_ID   = "cookiechips/KillSwitch_ai"   # 예시: 공개 HF 리포
DEFAULT_REPO_TYPE = "model"                       # "model" 또는 "dataset"
DEFAULT_FILENAME  = "prompt_guard_best.pt"        # 허브 내부 파일명/경로

REPO_ID   = st.secrets.get("HF_REPO_ID")   or os.getenv("HF_REPO_ID")   or DEFAULT_REPO_ID
REPO_TYPE = st.secrets.get("HF_REPO_TYPE") or os.getenv("HF_REPO_TYPE") or DEFAULT_REPO_TYPE
FILENAME  = st.secrets.get("HF_FILENAME")  or os.getenv("HF_FILENAME")  or DEFAULT_FILENAME
HF_TOKEN  = st.secrets.get("HF_TOKEN")     or os.getenv("HF_TOKEN")     # 공개면 비워도 됨
HF_DIR    = st.secrets.get("HF_DIR")       or os.getenv("HF_DIR")       # 완전 모델 디렉토리

BASE_MODEL = os.getenv("BASE_MODEL") or "microsoft/deberta-v3-base"
NUM_LABELS = int(os.getenv("NUM_LABELS") or 2)

# ─────────────────────────────────────────────────────────────────────────────
# 2) 모델/토크나이저 로드 (HF .pt → state_dict 주입)
# ─────────────────────────────────────────────────────────────────────────────
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def get_ckpt_path() -> str:
    """허브에서 .pt 파일을 캐시에 다운로드하고 경로 반환.
       ./model/prompt_guard_best.pt 가 있으면 그걸 우선 사용."""
    local = os.path.join("model", "prompt_guard_best.pt")
    if os.path.exists(local):
        return local

    try:
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type=REPO_TYPE,
            token=HF_TOKEN
        )
    except Exception as e1:
        # 반대 타입으로도 시도 (dataset에 올렸을 가능성)
        alt = "dataset" if REPO_TYPE == "model" else "model"
        try:
            p = hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                repo_type=alt,
                token=HF_TOKEN
            )
            st.info(f"repo_type='{REPO_TYPE}' 실패 → '{alt}'로 성공")
            return p
        except Exception as e2:
            st.error("체크포인트(.pt) 다운로드 실패 — 모델 미로딩 상태로 진행됩니다.")
            st.caption(str(e1))
            st.caption(str(e2))
            raise

@st.cache_resource(show_spinner=False)
def load_model_tokenizer():
    """토크나이저/모델 로드:
       1) HF_DIR이 지정되면 해당 디렉토리에서 from_pretrained
       2) 아니면 BASE_MODEL 로드 후 .pt state_dict 주입 (strict=True 시도 후 fallback)
    """
    # 토크나이저 고정 (SentencePiece 필요 → requirements.txt에 sentencepiece)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    st.caption("Tokenizer: slow (SentencePiece)")

    # 1) 완전 모델 디렉토리
    if HF_DIR and os.path.isdir(HF_DIR):
        mdl = AutoModelForSequenceClassification.from_pretrained(HF_DIR)
        mdl.to(DEVICE).eval()
        return mdl, tok, 0.5, True  # thr=0.5, torch_loaded=True

    # 2) 베이스 모델 + 체크포인트 주입
    mdl = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)
    thr = 0.5
    torch_loaded = False
    try:
        ckpt_path = get_ckpt_path()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # 우선 strict=True로 시도 → 분류 헤드 미주입 이슈 조기 발견
        try:
            mdl.load_state_dict(state, strict=True)
            st.caption("state_dict: strict=True (완전 일치)")
        except Exception as e_strict:
            st.warning(f"state_dict strict 로드 실패 → strict=False로 재시도: {e_strict}")
            missing, unexpected = mdl.load_state_dict(state, strict=False)
            st.caption(f"state_dict(strict=False) → missing:{len(missing)}, unexpected:{len(unexpected)}")

        # 체크포인트 내에 val_thr 저장되어 있으면 사용
        try:
            if isinstance(ckpt, dict) and "val_thr" in ckpt:
                thr = float(ckpt["val_thr"])
        except Exception:
            pass

        torch_loaded = True
    except Exception as e:
        st.info("학습 체크포인트를 불러오지 못했습니다(모델 미로딩).")
        st.caption(str(e))

    mdl.to(DEVICE).eval()
    return mdl, tok, thr, torch_loaded

@st.cache_resource(show_spinner=False)
def _cached_model():
    return load_model_tokenizer()

# ─────────────────────────────────────────────────────────────────────────────
# 3) 입력 정규화 & 강건 스코어링
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(s: str) -> str:
    """NFKC 정규화, 전각/스마트 구두점 치환, 공백 정리"""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    s = s.replace("。", ".")
    s = re.sub(r"\s+", " ", s).strip()
    return s

@torch.no_grad()
def score_once(mdl, tok, text: str) -> float:
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = mdl(**enc).logits
        if logits.size(-1) == 1:     # 시그모이드 헤드
            return torch.sigmoid(logits)[0, 0].item()
        else:                        # 소프트맥스 헤드
            return torch.softmax(logits, dim=-1)[0, 1].item()

def robust_score(mdl, tok, text: str, method: str = "mean") -> float:
    """마침표 유무에 강건한 점수: 원문 / 끝마침표 제거 / 끝마침표 강제 추가로 3회 스코어."""
    v1 = text
    v2 = text.rstrip(". ")
    v3 = (text.rstrip() + ".")
    scores = [score_once(mdl, tok, v) for v in (v1, v2, v3)]
    return sum(scores)/len(scores) if method == "mean" else max(scores)

# ─────────────────────────────────────────────────────────────────────────────
# 4) 추론
# ─────────────────────────────────────────────────────────────────────────────
def predict(text: str, thr_ui: float, dot_robust: bool, robust_method: str = "mean"):
    text = preprocess(text)
    mdl, tok, thr_ckpt, torch_loaded = _cached_model()
    t0 = time.time()

    if not torch_loaded:
        return {
            "점수": 0.0,
            "임계값": float(thr_ui if thr_ui is not None else thr_ckpt),
            "판정": "안전(모델 미로딩)",
            "세부": {"model_score": 0.0, "torch_loaded": False, "robust": dot_robust},
            "_elapsed_s": round(time.time() - t0, 2),
        }

    m_score = robust_score(mdl, tok, text, method=robust_method) if dot_robust else score_once(mdl, tok, text)
    thr = float(thr_ui if thr_ui is not None else thr_ckpt)
    label = "악성" if m_score >= thr else "안전"

    return {
        "점수": round(m_score, 3),
        "임계값": round(thr, 3),
        "판정": label,
        "세부": {"model_score": round(m_score, 3), "torch_loaded": True, "robust": dot_robust},
        "_elapsed_s": round(time.time() - t0, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5) UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛡️ KillSwitch AI")

# 사이드바
st.sidebar.header("설정")
thr_ui         = st.sidebar.slider("임계값(차단 기준)", 0.05, 0.95, 0.50, step=0.05)
dot_robust     = st.sidebar.checkbox("마침표 강건화(권장)", value=True)
robust_method  = st.sidebar.selectbox("강건화 방식", ["mean", "max"], index=0)

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

# 입력 & 버튼 (단일 버튼)
txt = st.text_area("프롬프트", height=140, placeholder="예) 인천 맛집 알려줘.")
run = st.button("분석 (GPT 호출)")

if run:
    with st.spinner("분석 중..."):
        result = predict(txt, thr_ui=thr_ui)
    st.success(f"분석 완료 ({result['_elapsed_s']:.2f}s)")

    # 분석 결과
    st.subheader("분석 결과  ↪️")
    st.json({k: v for k, v in result.items() if not k.startswith("_")})

    # GPT 응답 (한 버튼 안에서 후속 실행)
    st.subheader("GPT 응답")
    if not OPENAI_API_KEY:
        st.info("OPENAI_API_KEY가 없어 GPT 호출은 생략했습니다.")
    elif result["판정"] == "악성" and not force_call:
        st.warning("악성으로 판정되어 GPT 호출을 차단했습니다. (사이드바 '강행'을 체크하면 호출)")
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            sys_prompt = (
                "You are a helpful assistant. If the user's prompt is clearly harmful or illegal, "
                "politely refuse and provide safe alternatives."
            )
            msgs = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": txt}]
            rsp = client.chat.completions.create(
                model=openai_model,
                messages=msgs,
                temperature=0.3,
                top_p=1.0,
            )
            st.write(rsp.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"GPT 호출 오류: {type(e).__name__}: {e}")
            st.caption("429(쿼터 초과) 등 요금제/모델 이름을 확인하세요.")
    
    
