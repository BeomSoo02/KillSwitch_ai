import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from contextlib import nullcontext

def _safe_ctx():
    try:
        from torch.serialization import safe_globals
        return safe_globals([np.core.multiarray.scalar])
    except Exception:
        return nullcontext()

def _extract_state(sd):
    if isinstance(sd, dict):
        if any(isinstance(k,str) and (k.endswith('.weight') or k.endswith('.bias')) for k in sd.keys()): return sd
        if 'model' in sd and isinstance(sd['model'], dict): return sd['model']
        if 'state_dict' in sd and isinstance(sd['state_dict'], dict): return sd['state_dict']
    return sd

def load_classifier(model_name, ckpt_path, device='cpu'):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    with _safe_ctx(): ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = _extract_state(ckpt); model.load_state_dict(state, strict=False)
    model.to(device).eval()
    best_thr = ckpt.get('best_thr', ckpt.get('val_thr', None))
    try: best_thr = float(best_thr) if best_thr is not None else None
    except: best_thr = None
    return tok, model, best_thr

@torch.inference_mode()
def predict_one(text, tok, model, threshold=0.95, max_len=256, device='cpu'):
    enc = tok([text], padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    enc = {k:v.to(device) for k,v in enc.items()}
    prob = torch.softmax(model(**enc).logits, dim=-1)[:,1].cpu().numpy()[0]
    return int(prob>=threshold), float(prob)
