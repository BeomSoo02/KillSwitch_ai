# app.py — KillSwitch AI · Streamlit 데모 (최종본)
# ----------------------------------------------------------------------
# ✔ HF 허브의 체크포인트(.pt) 자동 다운로드 → state_dict 로드
# ✔ 규칙(rule) + 모델 점수 융합 (보수적: max)
# ✔ 사이드바: 임계값, 입력언어, 강행 호출, OpenAI 키
# ✔ HF 연결 점검 버튼 / torch_loaded 표시
# ✔ HF_DIR 제공 시 from_pretrained 디렉토리 직접 로드(완전한 모델 형식)
# ----------------------------------------------------------------------

import os, time, re, json
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 0) 페이지 설정 (최상단에서 한 번만)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="KillSwitch AI — Streamlit 데모", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 1) 환경/시크릿 설정
#    - 아래 기본값은 Secrets/Env로 덮어쓸 수 있습니다.
#      (Streamlit Cloud → Settings → Secrets)
#      HF_REPO_ID, HF_REPO_TYPE, HF_FILENAME, HF_TOKEN, HF_DIR
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REPO_ID   = "your-username/killswitch-ai-checkpoints"   # 예: "beomsu/killswitch-ckpt"
DEFAULT_REPO_TYPE = "model"                                      # "model" 또는 "dataset"
DEFAULT_FILENAME  = "pt/prompt_guard_best.pt"                    # 허브 내부 경로

REPO_ID   = st.secrets.get("HF_REPO_ID")   or os.getenv("HF_REPO_ID")   or DEFAULT_REPO_ID
REPO_TYPE = st.secrets.get("HF_REPO_TYPE") or os.getenv("HF_REPO_TYPE") or DEFAULT_REPO_TYPE
FILENAME  = st.secrets.get("HF_FILENAME")  or os.getenv("HF_FILENAME")  or DEFAULT_FILENAME
HF_TOKEN  = st.secrets.get("HF_TOKEN")     or os.getenv("HF_TOKEN")     # 비공개 리포면 필수
HF_DIR    = st.secrets.get("HF_DIR")       or os.getenv("HF_DIR")       # 완전 모델 디렉토리

# 모델 베이스/라벨 수 (학습 설정에 맞게)
BASE_MODEL = os.getenv("BASE_MODEL") or "microsoft/deberta-v3-base"
NUM_LABELS = int(os.getenv("NUM_LABELS") or 2)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Hugging Face 체크포인트 로더
# ─────────────────────────────────────────────────────────────────────────────
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource(show_spinner=False)
def get_ckpt_path() -> str:
    """허브에서 .pt 한 파일을 받아 로컬 캐시에 저장하고 경로 반환.
       먼저 ./model/prompt_guard_best.pt 가 있으면 그걸 우선 사용."""
    # 0) 로컬 우선 (앱 레포에 포함했을 때)
    local = os.path.join("model", "prompt_guard_best.pt")
    if os.path.exists(local):
        return local

    # 1) 선언된 repo_type으로 시도
    try:
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type=REPO_TYPE,
            token=HF_TOKEN
        )
    except Exception as e1:
        # 2) 반대 타입도 한 번 시도 (dataset에 올려둔 경우를 대비)
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
            st.error("학습 체크포인트를 불러오지 못했습니다(규칙 기반만 사용). 상세:")
            st.caption(str(e1))
            st.caption(str(e2))
            raise

@st.cache_resource(show_spinner=False)
def load_model_tokenizer():
    """토크나이저/모델 로드.
       1) HF_DIR 지정 시 해당 디렉토리에서 from_pretrained
       2) 아니면 BASE_MODEL 로드 후 .pt state_dict 덮어쓰기
    """
    # 토크나이저: sentencepiece 필요 → requirements.txt에 포함
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    except Exception as e:
        # 환경에 sentencepiece 없을 때 fast로 재시도
        st.info("슬로우 토크나이저 로드 실패 → fast 토크나이저로 재시도합니다.")
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    # 1) 완전 모델 디렉토리
    if HF_DIR and os.path.isdir(HF_DIR):
        mdl = AutoModelForSequenceClassification.from_pretrained(HF_DIR)
        mdl.eval()
        return mdl, tok, 0.5, True  # thr=0.5, torch_loaded=True (완전 모델 형식)

    # 2) 베이스 모델 + 체크포인트 state_dict
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
        st.info("학습 체크포인트를 불러오지 못했습니다(규칙 기반만 사용). 상세:")
        st.caption(str(e))
    mdl.eval()
    return mdl, tok, thr, torch_loaded

@st.cache_resource(show_spinner=False)
def _cached_model():
    return load_model_tokenizer()

# ─────────────────────────────────────────────────────────────────────────────
# 3) 규칙 기반 탐지 (간단 키워드)
# ─────────────────────────────────────────────────────────────────────────────
BAD_PATTERNS = [
    r"자살|자해|극단적\s*선택",
    r"폭탄|총기|살인|테러",
    r"마약|필로폰|코카인",
    r"해킹|디도스|랜섬웨어|취약점\s*악용",
    r"비밀번호|패스워드|OTP|백도어",
    r"주민등록번호|여권번호|신용카드\s*번호",
    r"피싱|보이스\s*피싱|메신저\s*피싱|phishing",
    # 살해/살상 인텐트(한국어/영어)
    r"(사람|타인|상대|누구|그녀|그놈|그새끼).{0,6}죽(여|일|이|여줘|여라|이게|이는|이는법)",
    r"죽여줘|죽이는\s*방법|죽일\s*방법|죽이는법|죽일래|죽여\s*버려",
    r"kill( someone| him| her| them)?|how to kill|murder"
]
BAD_RE = re.compile("|".join(BAD_PATTERNS), re.I)


def rule_detect(text: str):
    if not text.strip():
        return 0.0, []
    found = sorted(set([m.group(0) for m in BAD_RE.finditer(text)]))
    score = min(1.0, len(found) * 0.4) if found else 0.0  # 키워드당 0.4 가중
    return score, found

# ─────────────────────────────────────────────────────────────────────────────
# 4) 추론 & 융합 로직
# ─────────────────────────────────────────────────────────────────────────────
def predict(text: str, thr_ui: float):
    mdl, tok, thr_ckpt, torch_loaded = _cached_model()
    t0 = time.time()

    # ① 규칙 점수
    r_score, r_keys = rule_detect(text)

    # ② 모델 점수
    m_score = 0.0
    if torch_loaded:
        enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = mdl(**enc).logits
            if logits.size(-1) == 1:   # 시그모이드 헤드
                m_score = torch.sigmoid(logits)[0, 0].item()
            else:                      # 소프트맥스 헤드
                m_score = 1.0 - torch.softmax(logits, dim=-1)[0, 1].item()   # benign 확률 → 악성 확률로 반전

    # ③ 융합: 보수적으로 max 사용
    score = max(r_score, m_score)
    thr = float(thr_ui if thr_ui is not None else thr_ckpt)
    label = "악성" if score >= thr else "안전"

    return {
        "점수": round(score, 3),
        "임계값": round(thr, 3),
        "판정": label,
        "키워드": r_keys or ["-"],
        "세부": {
            "rule_score": round(r_score, 3),
            "model_score": round(m_score, 3),
            "torch_loaded": bool(torch_loaded),
        },
        "_elapsed_s": round(time.time() - t0, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5) UI
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 5) UI  ← 기존 버튼 2개 쓰던 블록을 이걸로 통째로 교체
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛡️ KillSwitch AI — Streamlit 데모")

# 사이드바
st.sidebar.header("설정")
OPENAI_API_KEY = st.sidebar.text_input("OPENAI_API_KEY", type="password")
openai_model   = st.sidebar.text_input("OpenAI 모델", value="gpt-4o-mini")
thr_ui         = st.sidebar.slider("임계값(차단 기준)", 0.05, 0.95, 0.70, step=0.05)
input_lang     = st.sidebar.selectbox("입력 언어", ["auto", "ko", "en"], index=0)
force_call     = st.sidebar.checkbox("위험해도 GPT 호출 강행", value=False)

# HF 연결 점검 (그대로 유지)
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
txt = st.text_area("프롬프트", height=140, placeholder="예) 인천 맛집 알려줘")
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
