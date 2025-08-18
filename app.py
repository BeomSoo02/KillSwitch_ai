import streamlit as st
import torch
from utils import load_classifier, predict_one
from config import MODEL_NAME, CHECKPOINT_PATH, DEFAULT_THRESHOLD, MAX_LEN

st.set_page_config(page_title='KillSwitch AI — Prompt Guard (Classifier)', page_icon='🛡️')
st.title('🛡️ KillSwitch AI — Prompt Guard (Classifier)')
@st.cache_resource(show_spinner=True)
def get_runtime():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok, model, best_thr = load_classifier(MODEL_NAME, CHECKPOINT_PATH, device=device)
    return tok, model, device, best_thr
tok, model, device, best_thr = get_runtime()
with st.sidebar:
    base_thr = best_thr if best_thr is not None else DEFAULT_THRESHOLD
    thr = st.slider('Decision threshold', 0.50, 0.99, float(base_thr), 0.01)
    st.write(f'Device: {device}')
txt = st.text_area('프롬프트', height=150)
if st.button('분석하기', type='primary'):
    if not txt.strip(): st.warning('문장을 입력하세요.')
    else:
        y, p = predict_one(txt, *get_runtime()[:2], threshold=thr, max_len=MAX_LEN, device=device)
        st.write(f'악성 확률: {p:.4f}'); st.write('판정: ' + ('🚫 악성' if y else '✅ 정상'))
