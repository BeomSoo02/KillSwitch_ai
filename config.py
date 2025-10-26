REPO_ID  = "cookiechips/KillSwitch_ai"
REVISION = "main"           # 태그 없다면 일단 main으로
BASE_MODEL = "microsoft/deberta-v3-base"  # v3는 DebertaV2 아키텍처
DEFAULT_THRESHOLD = 0.50
MAX_LEN = 256
# (선택) 파일명을 아는 경우 명시
HF_CKPT_FILENAME = "prompt_guard_best.pt"
