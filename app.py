# app.py — KillSwitch AI · Streamlit 데모 (모델 전용 + 외부 룰 플러그인)
# ----------------------------------------------------------------------
# ✔ HF 허브 체크포인트(.pt) 자동 다운로드 → state_dict 로드
# ✔ 기본: 모델만 사용. 룰은 외부 파일(RULE_PATH) 있을 때만 플러그인처럼 로드
# ✔ USE_RULE(기본 on) + RULE_PATH 둘 다 충족 시에만 룰 활성
# ✔ 융합: 룰 활성 시 max(rule, model), 비활성 시 model만
# ✔ 사이드바: 임계값, 입력언어, 강행 호출, OpenAI 키, HF 연결 점검
# ----------------------------------------------------------------------

import os, time, re, json
import streamlit as st

st.set_page_config(page_title="KillSwitch AI", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 1) 환경/시크릿
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REPO_ID   = "your-username/killswitch-ai-checkpoints"
DEFAULT_REPO_TYPE = "model"
DEFAULT_FILENAME  = "pt/prompt_guard_best.pt"

REPO_ID   = st.secrets.get("HF_REPO_ID")   or os.getenv("HF_REPO_ID")   or DEFAULT_REPO_ID
REPO_TYPE = st.secrets.get("HF_REPO_TYPE") or os.getenv("HF_REPO_TYPE") or DEFAULT_REPO_TYPE
FILENAME  = st.secrets.get("HF_FILENAME")  or os.getenv("HF_FILENAME")  or DEFAULT_FILENAME
HF_TOKEN  = st.secrets.get("HF_TOKEN")     or os.getenv("HF_TOKEN")
HF_DIR    = st.secrets.get("HF_DIR")       or os.getenv("HF_DIR")

BASE_MODEL = os.getenv("BASE_MODEL") or "microsoft/deberta-v3-base"
NUM_LABELS = int(os.getenv("NUM_LABELS") or 2)

# 룰 관련(기본 on이지만, 외부 파일 없으면 자동 비활성)
USE_RULE_ENV = os.getenv("USE_RULE", "on").lower() in ["on", "1", "true"]
RULE_PATH    = os.getenv("RULE_PATH") or st.secrets.get("RULE_PATH")  # 예: rules/patterns.txt

# ─────────────────────────────────────────────────────────────────────────────
# 2) 모델 로더
# ─────────────────────────────────────────────────────────────────────────────
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource(show_spinner=False)
def get_ckpt_path() -> str:
    local = os.path.join("model", "prompt_guard_best.pt")
    if os.path.exists(local):
        return local
    try:
        return hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type=REPO_TYPE, token=HF_TOKEN)
    except Exception as e1:
        alt = "dataset" if REPO_TYPE == "model" else "model"
        try:
            p = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type=alt, token=HF_TOKEN)
            st.info(f"repo_type='{REPO_TYPE}' 실패 → '{alt}'로 성공")
            return p
        except Exception as e2:
            st.error("학습 체크포인트를 불러오지 못했습니다. (룰 비활성 시 모델만, 룰도 없으면 판정 제한)")
            st.caption(str(e1)); st.caption(str(e2))
            raise

@st.cache_resource(show_spinner=False)
def load_model_tokenizer():
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    except Exception:
        st.info("슬로우 토크나이저 로드 실패 → fast 토크나이저로 재시도합니다.")
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    if HF_DIR and os.path.isdir(HF_DIR):
        mdl = AutoModelForSequenceClassification.from_pretrained(HF_DIR)
        mdl.eval()
        return mdl, tok, 0.5, True

    mdl = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)
    thr = 0.5; torch_loaded = False
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
        st.info("체크포인트를 불러오지 못했습니다. (룰 없으면 판정이 제한될 수 있음)")
        st.caption(str(e))
    mdl.eval()
    return mdl, tok, thr, torch_loaded

@st.cache_resource(show_spinner=False)
def _cached_model():
    return load_model_tokenizer()

# ─────────────────────────────────────────────────────────────────────────────
# 3) 외부 룰 플러그인 로딩 (코드 내 키워드 없음)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_external_rule():
    """
    RULE_PATH가 주어지고 USE_RULE_ENV=True일 때만 외부 파일을 로드하여 컴파일.
    지원 포맷:
      - .txt  : 줄당 하나의 정규식
      - .json : ["pat1", "pat2", ...]
      - .yaml/.yml : ["pat1", "pat2", ...]
    실패 시 None 반환 → 룰 비활성
    """
    if not USE_RULE_ENV or not RULE_PATH:
        return None

    try:
        ext = os.path.splitext(RULE_PATH)[1].lower()
        patterns = []
        if ext == ".txt":
            with open(RULE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        patterns.append(s)
        elif ext == ".json":
            with open(RULE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    patterns = [str(x) for x in data if str(x).strip()]
        elif ext in [".yaml", ".yml"]:
            import yaml  # pyyaml 필요
            with open(RULE_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, list):
                    patterns = [str(x) for x in data if str(x).strip()]
        else:
            st.warning(f"RULE_PATH 확장자 미지원: {ext} (txt/json/yaml 권장)")
            return None

        if not patterns:
            st.info("RULE_PATH에서 유효한 패턴을 찾지 못했습니다. 룰 비활성.")
            return None

        return re.compile("|".join(patterns), re.I)
    except Exception as e:
        st.info(f"외부 룰 로드 실패 → 비활성화합니다. 상세: {e}")
        return None

RULE_RE = _load_external_rule()
USE_RULE = bool(RULE_RE)  # 실제 활성 여부(파일 로딩 성공 시에만 True)

def rule_detect(text: str):
    if not USE_RULE or RULE_RE is None or not text.strip():
        return 0.0, None  # 키워드 목록은 반환하지 않음(리터럴 노출 방지)
    found = list(RULE_RE.finditer(text))
    score = 1.0 if found else 0.0   # 외부 룰은 일단 0/1 방식(필요 시 가중 변경)
    return score, None

# ─────────────────────────────────────────────────────────────────────────────
# 4) 추론 & 융합
# ─────────────────────────────────────────────────────────────────────────────
def predict(text: str, thr_ui: float):
    mdl, tok, thr_ckpt, torch_loaded = _cached_model()
    t0 = time.time()

    # (1) 규칙 점수
    r_score, _ = rule_detect(text)  # 키워드 목록 미노출

    # (2) 모델 점수
    m_score = 0.0
    if torch_loaded:
        enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = mdl(**enc).logits
            m_score = torch.sigmoid(logits)[0, 0].item() if logits.size(-1) == 1 \
                      else torch.softmax(logits, dim=-1)[0, 1].item()
    elif not USE_RULE:
        st.error("모델이 로드되지 않았고 외부 룰도 비활성화되어 판정이 불가합니다.")
        return {"판정": "불가", "점수": 0.0, "세부": {"rule_on": USE_RULE, "torch_loaded": torch_loaded}}

    # (3) 융합
    score = max(r_score, m_score) if USE_RULE else m_score
    thr = float(thr_ui if thr_ui is not None else thr_ckpt)
    label = "악성" if score >= thr else "안전"

    return {
        "점수": round(score, 3),
        "임계값": round(thr, 3),
        "판정": label,
        "세부": {
            "rule_on": USE_RULE,
            "rule_src": (RULE_PATH if USE_RULE else None),
            "rule_score": round(r_score, 3),
            "model_score": round(m_score, 3),
            "torch_loaded": bool(torch_loaded),
        },
        "_elapsed_s": round(time.time() - t0, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5) UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛡️ KillSwitch AI")

st.sidebar.header("설정")
st.sidebar.caption(f"규칙 기반 탐지: {'ON' if USE_RULE else 'OFF'}"
                   + (f" · {RULE_PATH}" if USE_RULE else ""))
OPENAI_API_KEY = st.sidebar.text_input("OPENAI_API_KEY", type="password")
openai_model   = st.sidebar.text_input("OpenAI 모델", value="gpt-4o-mini")
thr_ui         = st.sidebar.slider("임계값(차단 기준)", 0.05, 0.95, 0.35, step=0.05)
input_lang     = st.sidebar.selectbox("입력 언어", ["auto", "ko", "en"], index=0)
force_call     = st.sidebar.checkbox("위험해도 GPT 호출 강행", value=False)

st.sidebar.caption(f"HF: {REPO_ID} ({REPO_TYPE}) / {FILENAME}")
if st.sidebar.button("HF 연결 점검"):
    try:
        p = get_ckpt_path()
        size_mb = os.path.getsize(p) / 1_048_576
        st.sidebar.success(f"OK: {os.path.basename(p)} · {size_mb:.1f} MB")
    except Exception as e:
        st.sidebar.error("다운로드 실패")
        st.sidebar.exception(e)

txt = st.text_area("프롬프트", height=140, placeholder="예) 인천 맛집 알려줘")
run = st.button("분석 (GPT 호출)")

if run:
    with st.spinner("분석 중..."):
        result = predict(txt, thr_ui=thr_ui)
    st.success(f"분석 완료 ({result['_elapsed_s']:.2f}s)")
    st.subheader("분석 결과  ↪️")
    st.json({k: v for k, v in result.items() if not k.startswith("_")})

    st.subheader("GPT 응답")
    if not OPENAI_API_KEY:
        st.info("OPENAI_API_KEY가 없어 GPT 호출은 생략했습니다.")
    elif result["판정"] == "악성" and not force_call:
        st.warning("악성으로 판정되어 GPT 호출을 차단했습니다. ('강행' 옵션으로 무시 가능)")
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
