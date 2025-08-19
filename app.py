# app.py ─ KillSwitch AI — Streamlit 데모
import os, time, re, json
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 0) Hugging Face 체크포인트 로더 (허브→로컬 캐시)
# ─────────────────────────────────────────────────────────────────────────────
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ★★ 여기만 너의 허브 리포/파일로 수정 ★★
REPO_ID   = "your-username/killswitch-ai-checkpoints"   # 예: "beomsu/killswitch-ckpt"
REPO_TYPE = "model"                                     # 모델 리포면 "model", 데이터셋이면 "dataset"
FILENAME  = "pt/prompt_guard_best.pt"                   # 허브 내부 경로(폴더/파일)

@st.cache_resource(show_spinner=False)
def get_ckpt_path():
    # 1) 프로젝트에 파일을 같이 넣었으면 로컬 우선
    local_path = os.path.join("model", "prompt_guard_best.pt")
    if os.path.exists(local_path):
        return local_path
    # 2) 허브에서 단일 파일 다운로드
    token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    return hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                           repo_type=REPO_TYPE, token=token)

# ─────────────────────────────────────────────────────────────────────────────
# 1) 모델/토크나이저 로드
#   - HF_DIR 환경변수로 from_pretrained 디렉토리를 직접 지정할 수도 있음
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL = "microsoft/deberta-v3-base"  # 실제 학습에 쓴 베이스로 변경 가능
NUM_LABELS = 2

@st.cache_resource(show_spinner=False)
def load_model_tokenizer():
    # tokenizer: sentencepiece 필요 → requirements.txt에 포함
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

    # 우선순위 1) 완전한 HF 형식 디렉토리(HF_DIR)
    hf_dir = os.getenv("HF_DIR")
    if hf_dir and os.path.isdir(hf_dir):
        mdl = AutoModelForSequenceClassification.from_pretrained(hf_dir)
        mdl.eval()
        return mdl, tok, 0.5, True  # thr 기본 0.5, torch_loaded=True

    # 우선순위 2) BASE_MODEL 로드 후 state_dict 덮어쓰기
    mdl = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)
    thr = 0.5
    torch_loaded = False
    try:
        ckpt_path = get_ckpt_path()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        mdl.load_state_dict(state, strict=False)
        thr = float(ckpt.get("val_thr", 0.5)) if isinstance(ckpt, dict) else 0.5
        torch_loaded = True
    except Exception as e:
        st.info("학습 체크포인트를 불러오지 못했습니다(규칙 기반만 사용). 상세:")
        st.caption(str(e))
    mdl.eval()
    return mdl, tok, thr, torch_loaded

# ─────────────────────────────────────────────────────────────────────────────
# 2) 간단 규칙 기반 점수 (키워드 매칭)
# ─────────────────────────────────────────────────────────────────────────────
BAD_PATTERNS = [
    r"자살|자해|극단적\s*선택", r"폭탄|총기|살인|테러", r"마약|필로폰|코카인",
    r"해킹|디도스|랜섬웨어|취약점\s*악용", r"비밀번호|패스워드|OTP|백도어",
    r"주민등록번호|여권번호|신용카드\s*번호"
]
BAD_RE = re.compile("|".join(BAD_PATTERNS), re.I)

def rule_detect(text: str):
    if not text.strip():
        return 0.0, []
    found = sorted(set([m.group(0) for m in BAD_RE.finditer(text)]))
    score = min(1.0, len(found) * 0.4) if found else 0.0  # 키워드당 가중치
    return score, found

# ─────────────────────────────────────────────────────────────────────────────
# 3) 모델 추론 + 융합 로직 (rule ⨂ model)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _cached_mdl():
    return load_model_tokenizer()

def predict(text: str, thr_ui: float):
    mdl, tok, thr_ckpt, torch_loaded = _cached_mdl()
    t0 = time.time()

    # ① 규칙 점수
    r_score, r_keys = rule_detect(text)

    # ② 모델 점수(있으면)
    m_score = 0.0
    if torch_loaded:
        enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = mdl(**enc).logits
            if logits.size(-1) == 1:
                m_score = torch.sigmoid(logits)[0, 0].item()
            else:
                m_score = torch.softmax(logits, dim=-1)[0, 1].item()

    # ③ 융합(간단: max) — 보수적으로 막기
    score = max(r_score, m_score)
    thr = float(thr_ui if thr_ui is not None else thr_ckpt)
    label = "악성" if score >= thr else "안전"

    elapsed = time.time() - t0
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
        "_elapsed_s": round(elapsed, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4) UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="KillSwitch AI — Streamlit 데모", layout="wide")
st.title("🛡️ KillSwitch AI — Streamlit 데모")

# ─ Sidebar: 설정
st.sidebar.header("설정")
OPENAI_API_KEY = st.sidebar.text_input("OPENAI_API_KEY", type="password")
openai_model   = st.sidebar.text_input("OpenAI 모델", value="gpt-5") 
thr_ui         = st.sidebar.slider("임계값(차단 기준)", 0.05, 0.95, 0.70, step=0.05)
input_lang     = st.sidebar.selectbox("입력 언어", ["auto", "ko", "en"], index=0)
force_call     = st.sidebar.checkbox("위험해도 GPT 호출 강행", value=False)

# ─ Main: 입력 & 버튼
txt = st.text_area("프롬프트", height=140, placeholder="예) 인천 맛집 알려줘")
col1, col2 = st.columns([1,1])
btn_analyze = col1.button("분석")
btn_both    = col2.button("분석 후 GPT 호출")

# ─ 분석
result = None
if btn_analyze or btn_both:
    result = predict(txt, thr_ui=thr_ui)
    st.success(f"분석 완료 ({result['_elapsed_s']:.2f}s)")
    st.subheader("분석 결과  ↪️")
    st.json({k: v for k, v in result.items() if not k.startswith("_")})

# ─ GPT 호출
st.subheader("GPT 응답")
if btn_both:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY가 비었습니다. 사이드바에 입력해주세요.")
    elif result and result["판정"] == "악성" and not force_call:
        st.warning("악성으로 판정되어 GPT 호출을 차단했습니다. (사이드바에서 '강행' 체크 시 시도)")
    else:
        try:
            # OpenAI Chat Completions (python SDK v1)
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            # 간단한 시스템 프롬프트 — 안전 가이드 유지(우회 지시 X)
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
            st.caption("요금제/쿼터 부족(429)일 수 있습니다. 결제/모델 이름을 확인하세요.")

# ─ 하단 도움말
st.markdown("---")
st.caption("HF_DIR 환경변수로 학습 체크포인트 디렉토리를 지정할 수 있습니다. (예: models/hf_checkpoint)")
st.caption(f"허브: {REPO_ID} / 파일: {FILENAME} / RepoType: {REPO_TYPE}")
