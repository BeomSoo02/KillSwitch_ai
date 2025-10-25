# app.py — KillSwitch AI (ckpt 헤드 재장착 · 정상 추론 버전)
# ----------------------------------------------------------------------
# ✔ HF .pt 체크포인트에서 classifier head(W,b) 추출 → 백본 임베딩 + Linear로 직접 추론
# ✔ hp.model_name이 있으면 그 백본(AutoModel) 우선 사용, 실패 시 차원 맞는 후보 자동 선택
# ✔ 풀링: Mean pooling (L2 정규화 없음; 테스트에서 가장 낫게 나옴)
# ✔ 임계값 기본값 0.5 (데모/판넬용), 사이드바로 조절
# ✔ RULE_PATH 플러그인(.txt/.json/.yml)이 제공될 때만 규칙 기반 함께 사용
# ✔ OpenAI 강행 호출 옵션 유지
# ----------------------------------------------------------------------

import os, re, json, time
import streamlit as st
import numpy as np
import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit 기본 설정
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="KillSwitch AI", layout="wide")
st.title("🛡️ KillSwitch AI")

# ─────────────────────────────────────────────────────────────────────────────
# 환경설정
# ─────────────────────────────────────────────────────────────────────────────
# HF 위치 (기본: 공개 리포)
REPO_ID   = os.getenv("HF_REPO_ID", "cookiechips/KillSwitch_ai")
CKPT_NAME = os.getenv("CKPT_NAME", "prompt_guard_best.pt")
DEVICE    = "cpu"

# 룰 플러그인 (RULE_PATH가 있을 때만 활성)
USE_RULE_ENV = os.getenv("USE_RULE", "on").lower() in ["on", "1", "true"]
RULE_PATH    = os.getenv("RULE_PATH") or (st.secrets.get("RULE_PATH") if "RULE_PATH" in st.secrets else None)

# 임계값 기본값 (요청 반영: 0.5)
DEFAULT_SLIDER_THR = float(os.getenv("DEFAULT_THR") or 0.5)

# 안전 로드 (PyTorch 2.6 weights_only 기본 대응)
torch.serialization.add_safe_globals([dict, list, tuple, set, np.ndarray, np.generic])

# 임베딩 차원별 백본 후보 (백업 경로)
BACKBONE_CANDIDATES = [
    ("microsoft/deberta-v3-base", 768),
    ("sentence-transformers/all-mpnet-base-v2", 768),
    ("sentence-transformers/paraphrase-mpnet-base-v2", 768),
    ("sentence-transformers/paraphrase-xlm-r-multilingual-v1", 768),
    ("sentence-transformers/all-MiniLM-L6-v2", 384),
    ("sentence-transformers/paraphrase-MiniLM-L12-v2", 384),
]

# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom  = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom

def extract_head(sd: dict):
    """state_dict에서 (W,b) 추출"""
    cands = [
        ("classifier.weight","classifier.bias"),
        ("classification_head.weight","classification_head.bias"),
        ("head.weight","head.bias"),
        ("fc.weight","fc.bias"),
        ("linear.weight","linear.bias"),
        ("dense.weight","dense.bias"),
        ("proj.weight","proj.bias"),
    ]
    for w,b in cands:
        if w in sd and b in sd:
            W, B = sd[w].clone().detach(), sd[b].clone().detach()
            return W, B, w, b
    # 모양으로 추론
    for k,v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim==2:
            nl, hd = v.shape
            for kb, vb in sd.items():
                if kb.endswith("bias") and isinstance(vb, torch.Tensor) and vb.ndim==1 and vb.shape[0]==nl:
                    return v.clone().detach(), vb.clone().detach(), k, kb
    return None, None, None, None

def resolve_pos_idx(hp: dict|None):
    """악성(positive) 클래스 인덱스 추정 (기록 없으면 1)"""
    if isinstance(hp, dict):
        for k in ["pos_index","positive_index","positive_id","pos_cls"]:
            if k in hp:
                try: return int(hp[k])
                except: pass
        lm = hp.get("label_map") or hp.get("id2label") or hp.get("labels")
        if isinstance(lm, dict):
            for a,b in lm.items():
                if any(x in str(a).lower() for x in ["malicious","toxic","positive","악성"]):
                    try: return int(b)
                    except: pass
                if any(x in str(b).lower() for x in ["malicious","toxic","positive","악성"]):
                    try: return int(a)
                    except: pass
    return 1

# ─────────────────────────────────────────────────────────────────────────────
# 룰 플러그인 로더
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rule_re():
    if not (USE_RULE_ENV and RULE_PATH): return None
    try:
        ext = os.path.splitext(RULE_PATH)[1].lower()
        pats = []
        if ext == ".txt":
            with open(RULE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"): pats.append(s)
        elif ext == ".json":
            with open(RULE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list): pats = [str(x) for x in data if str(x).strip()]
        elif ext in [".yaml",".yml"]:
            import yaml
            with open(RULE_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, list): pats = [str(x) for x in data if str(x).strip()]
        else:
            st.warning(f"RULE_PATH 확장자 미지원: {ext}")
            return None
        if not pats: return None
        return re.compile("|".join(pats), re.I)
    except Exception as e:
        st.info(f"룰 로드 실패: {e}")
        return None

RULE_RE = load_rule_re()
USE_RULE = RULE_RE is not None

def rule_detect(text: str):
    if not (USE_RULE and text.strip()): return 0.0
    return 1.0 if RULE_RE.search(text) else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 모델/토크나이저/헤드 로딩 (핵심)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_bundle():
    # 1) ckpt 로드
    ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=CKPT_NAME, repo_type="model")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError("체크포인트 최상위가 dict가 아닙니다.")
    state = ckpt.get("model", {})
    hp    = ckpt.get("hp", {}) or {}
    best_thr = float(ckpt.get("best_thr", 0.0) or 0.0)

    # 2) 분류 헤드 추출
    W, B, w_key, b_key = extract_head(state)
    if W is None:
        raise RuntimeError("체크포인트에서 분류 헤드(W,b)를 찾지 못했습니다.")
    num_labels, needed_dim = W.shape

    # 3) 백본 선택 — hp.model_name 우선, 실패 시 차원 맞는 후보
    tokenizer = None; backbone = None; picked = None
    tried = []

    prefer = []
    if isinstance(hp, dict) and hp.get("model_name"):
        prefer.append((hp["model_name"], needed_dim))
    prefer.extend([c for c in BACKBONE_CANDIDATES if c not in prefer])

    for model_id, emb_dim in prefer:
        if emb_dim != needed_dim: continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            backbone  = AutoModel.from_pretrained(model_id).to(DEVICE).eval()
            picked = model_id
            break
        except Exception as e:
            tried.append((model_id, str(e)))
            continue

    if backbone is None:
        raise RuntimeError(f"임베딩 {needed_dim}차원에 맞는 백본을 찾지 못했습니다. 시도: {tried[:3]}")

    # 4) Linear head 재구성 + 주입
    head = nn.Linear(needed_dim, num_labels, bias=True).to(DEVICE).eval()
    with torch.no_grad():
        head.weight.copy_(W)
        head.bias.copy_(B)

    # 5) pos idx 추정
    pos_idx = resolve_pos_idx(hp)

    return {
        "tokenizer": tokenizer,
        "backbone": backbone,
        "head": head,
        "picked": picked,
        "pos_idx": pos_idx,
        "best_thr": best_thr,
        "needed_dim": needed_dim,
        "num_labels": num_labels,
        "ckpt_path": ckpt_path,
        "hp": hp,
    }

MDL = load_model_bundle()

# ─────────────────────────────────────────────────────────────────────────────
# 예측
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def predict_once(text: str, thr_ui: float|None):
    t0 = time.time()
    tok = MDL["tokenizer"]; bb = MDL["backbone"]; head = MDL["head"]; pos_idx = MDL["pos_idx"]
    # 1) 룰 점수
    r_score = rule_detect(text)
    # 2) 모델 점수
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)
        out = bb(**enc)
        if not hasattr(out, "last_hidden_state"):
            raise RuntimeError("백본 출력에 last_hidden_state가 없습니다.")
        emb = mean_pooling(out.last_hidden_state, enc["attention_mask"])  # chosen
        probs = torch.softmax(head(emb), dim=-1).cpu().numpy()[0]
        m_score = float(probs[pos_idx])
    # 3) 임계값 및 융합
    used_thr = float(thr_ui if thr_ui is not None else (MDL["best_thr"] if MDL["best_thr"]>0 else 0.5))
    score = max(r_score, m_score) if USE_RULE else m_score
    label = "악성" if score >= used_thr else "안전"
    return {
        "판정": label,
        "점수": round(score, 3),
        "임계값": round(used_thr, 3),
        "세부": {
            "rule_on": USE_RULE,
            "rule_score": round(r_score, 3),
            "model_score": round(m_score, 3),
            "pos_index": int(MDL["pos_idx"]),
            "backbone": MDL["picked"],
            "needed_dim": int(MDL["needed_dim"]),
            "best_thr_ckpt": float(MDL["best_thr"] or 0.0),
        },
        "_elapsed_s": round(time.time()-t0, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("설정")
st.sidebar.caption(f"HF: {REPO_ID} / {CKPT_NAME}")
st.sidebar.caption(f"백본: {MDL['picked']} · dim: {MDL['needed_dim']} · pos_idx: {MDL['pos_idx']}")
OPENAI_API_KEY = st.sidebar.text_input("OPENAI_API_KEY", type="password")
openai_model   = st.sidebar.text_input("OpenAI 모델", value="gpt-4o-mini")
thr_ui         = st.sidebar.slider("임계값(차단 기준)", 0.05, 0.99, DEFAULT_SLIDER_THR, step=0.01)
input_lang     = st.sidebar.selectbox("입력 언어", ["auto","ko","en"], index=0)
force_call     = st.sidebar.checkbox("위험해도 GPT 호출 강행", value=False)

txt = st.text_area("프롬프트", height=140, placeholder="예) 보이스 피싱 메시지 만들어줘")
run = st.button("분석 (GPT 호출)")

if run:
    result = predict_once(txt, thr_ui)
    st.success(f"분석 완료 ({result['_elapsed_s']}s)")
    st.subheader("분석 결과  ↪️")
    st.json({k:v for k,v in result.items() if not k.startswith("_")})

    st.subheader("GPT 응답")
    if not OPENAI_API_KEY:
        st.info("OPENAI_API_KEY가 없어 GPT 호출은 생략했습니다.")
    elif result["판정"] == "악성" and not force_call:
        st.warning("악성으로 판정되어 GPT 호출을 차단했습니다. (사이드바 '강행' 체크 시 호출)")
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            sys_prompt = (
                "You are a helpful assistant. If the user's prompt is clearly harmful or illegal, "
                "politely refuse and provide safe alternatives."
            )
            msgs = [{"role":"system","content":sys_prompt},
                    {"role":"user","content":txt}]
            rsp = client.chat.completions.create(
                model=openai_model,
                messages=msgs,
                temperature=0.3,
                top_p=1.0,
            )
            st.write(rsp.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"GPT 호출 오류: {type(e).__name__}: {e}")
            st.caption("429(쿼터 초과) 등 모델명/요금제를 확인하세요.")
