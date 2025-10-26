# app.py — KillSwitch AI · Streamlit (모델 점수만 사용, 키워드 룰 제거)
# ----------------------------------------------------------------------
# ✔ 공개 HF 허브 체크포인트(.pt) 자동 다운로드 → state_dict 주입
# ✔ 규칙(rule) 제거: 모델 확률만으로 판정
# ✔ 사이드바: 임계값, OpenAI 키(선택), HF 연결 점검
# ✔ HF_DIR 제공 시 from_pretrained 디렉토리 직접 로드(완전 모델 형식)
# ----------------------------------------------------------------------

import os, time
import streamlit as st

st.set_page_config(page_title="KillSwitch AI", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 1) 환경/시크릿 설정 (Streamlit Secrets/Env 로 덮어쓰기 가능)
#    - HF_REPO_ID, HF_REPO_TYPE, HF_FILENAME, HF_TOKEN(공개면 불필요), HF_DIR
#    - BASE_MODEL, NUM_LABELS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REPO_ID   = "your-username/killswitch-ai-checkpoints"   # 예: "cookiechips/KillSwitch_ai"
DEFAULT_REPO_TYPE = "model"                                      # "model" 또는 "dataset"
DEFAULT_FILENAME  = "pt/prompt_guard_best.pt"                    # 허브 내부 경로

REPO_ID   = st.secrets.get("HF_REPO_ID")   or os.getenv("HF_REPO_ID")   or DEFAULT_REPO_ID
REPO_TYPE = st.secrets.get("HF_REPO_TYPE") or os.getenv("HF_REPO_TYPE") or DEFAULT_REPO_TYPE
FILENAME  = st.secrets.get("HF_FILENAME")  or os.getenv("HF_FILENAME")  or DEFAULT_FILENAME
HF_TOKEN  = st.secrets.get("HF_TOKEN")     or os.getenv("HF_TOKEN")     # 공개 리포면 생략 가능
HF_DIR    = st.secrets.get("HF_DIR")       or os.getenv("HF_DIR")       # 완전 모델 디렉토리 경로

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
    """허브에서 .pt 파일을 로컬 캐시에 다운로드하고 경로 반환.
       ./model/prompt_guard_best.pt 가 있으면 그걸 우선 사용."""
    local = os.path.join("model", "prompt_guard_best.pt")
    if os.path.exists(local):
        return local

    try:
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type=REPO_TYPE,
            token=HF_TOKEN  # 공개면 None이어도 OK
        )
    except Exception as e1:
        # 리포 타입 반대로도 시도 (dataset에 올렸을 수 있음)
        alt = "dataset" if REPO_TYPE == "model" else "model"
        try:
            p = hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                repo_type=alt,
                token=HF_TOKEN
            )
            st.info(f"repo_type='{REPO_TYPE}' 실패 → '{alt}'로 재시도 성공")
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
       2) 아니면 BASE_MODEL 로드 후 .pt state_dict 덮어쓰기
    """
    # 토크나이저
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    except Exception:
        # sentencepiece 미설치 등 환경 이슈 시 fast로 재시도
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

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
        missing, unexpected = mdl.load_state_dict(state, strict=False)
        if missing or unexpected:
            st.caption(f"state_dict mismatch → missing:{len(missing)}, unexpected:{len(unexpected)}")
        thr = float(ckpt.get("val_thr", 0.5)) if isinstance(ckpt, dict) else 0.5
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
# 3) 추론 (모델 점수만 사용)
# ─────────────────────────────────────────────────────────────────────────────
def predict(text: str, thr_ui: float):
    mdl, tok, thr_ckpt, torch_loaded = _cached_model()
    t0 = time.time()

    if not torch_loaded:
        # 모델이 없으면 안전으로 처리(또는 차단) — 여기선 안전으로 처리
        return {
            "점수": 0.0,
            "임계값": float(thr_ui if thr_ui is not None else thr_ckpt),
            "판정": "안전(모델 미로딩)",
            "세부": {
                "model_score": 0.0,
                "torch_loaded": False,
            },
            "_elapsed_s": round(time.time() - t0, 2),
        }

    # 모델 예측
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl(**enc).logits
        if logits.size(-1) == 1:     # 시그모이드 헤드
            m_score = torch.sigmoid(logits)[0, 0].item()
        else:                        # 소프트맥스 헤드
            m_score = torch.softmax(logits, dim=-1)[0, 1].item()

    thr = float(thr_ui if thr_ui is not None else thr_ckpt)
    label = "악성" if m_score >= thr else "안전"

    return {
        "점수": round(m_score, 3),
        "임계값": round(thr, 3),
        "판정": label,
        "세부": {
            "model_score": round(m_score, 3),
            "torch_loaded": True,
        },
        "_elapsed_s": round(time.time() - t0, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4) UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛡️ KillSwitch AI")

# 사이드바
st.sidebar.header("설정")
OPENAI_API_KEY = st.sidebar.text_input("OPENAI_API_KEY", type="password")
openai_model   = st.sidebar.text_input("OpenAI 모델", value="gpt-4o-mini")
thr_ui         = st.sidebar.slider("임계값(차단 기준)", 0.05, 0.95, 0.50, step=0.05)

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
run = st.button("분석 (모델만)")

if run:
    if not (txt and txt.strip()):
        st.warning("텍스트를 입력하세요.")
    else:
        with st.spinner("분석 중..."):
            result = predict(txt.strip(), thr_ui=thr_ui)
        st.success(f"분석 완료 ({result['_elapsed_s']:.2f}s)")

        # 분석 결과
        st.subheader("분석 결과  ↪️")
        st.json({k: v for k, v in result.items() if not k.startswith("_")})

        # (선택) GPT 응답 — 차단 로직 없이 참고용
        st.subheader("GPT 응답 (선택)")
        if not OPENAI_API_KEY:
            st.info("OPENAI_API_KEY가 없어 GPT 호출은 생략했습니다.")
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
