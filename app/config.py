import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT_DIR / "models"))).resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data"))).resolve()


def _bool_env(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_model(env_key: str, local_name: str) -> str:
    value = os.getenv(env_key)
    if value:
        return value
    local_path = MODELS_DIR / local_name
    if local_path.is_dir():
        return str(local_path)
    return local_name


ASR_MODEL = _resolve_model("ASR_MODEL", "paraformer-zh")
VAD_MODEL = _resolve_model("VAD_MODEL", "fsmn-vad")
PUNC_MODEL = _resolve_model("PUNC_MODEL", "ct-punc")
SPK_MODEL = _resolve_model("SPK_MODEL", "cam++")

HOST = os.getenv("ASR_HOST", "0.0.0.0")
PORT = int(os.getenv("ASR_PORT", "8000"))
DEVICE = os.getenv("ASR_DEVICE", "auto")
BATCH_SIZE_S = int(os.getenv("BATCH_SIZE_S", "300"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(1024 * 1024 * 1024)))
LEADER_DB_PATH = Path(os.getenv("LEADER_DB_PATH", str(DATA_DIR / "leaders.json")))
LEADER_THRESHOLD = float(os.getenv("LEADER_THRESHOLD", "0.62"))
LEADER_SPEAKER_THRESHOLD = float(os.getenv("LEADER_SPEAKER_THRESHOLD", "0.35"))
LEADER_SEGMENT_THRESHOLD = float(os.getenv("LEADER_SEGMENT_THRESHOLD", "0.50"))
LEADER_SCORE_MARGIN = float(os.getenv("LEADER_SCORE_MARGIN", "0.03"))
LEADER_MIN_SEGMENT_SECONDS = float(os.getenv("LEADER_MIN_SEGMENT_SECONDS", "1.0"))
SEGMENT_PADDING_MS = int(os.getenv("SEGMENT_PADDING_MS", "250"))

POSTPROCESS_ENABLED = _bool_env("POSTPROCESS_ENABLED", True)
POSTPROCESS_PRELOAD = _bool_env("POSTPROCESS_PRELOAD", False)
POSTPROCESS_MODEL = os.getenv("POSTPROCESS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
POSTPROCESS_MODEL_DIR = os.getenv("POSTPROCESS_MODEL_DIR", "")
POSTPROCESS_DEVICE = os.getenv("POSTPROCESS_DEVICE", os.getenv("ASR_DEVICE", "cpu"))
POSTPROCESS_MAX_CHARS = int(os.getenv("POSTPROCESS_MAX_CHARS", "1200"))
POSTPROCESS_MAX_NEW_TOKENS = int(os.getenv("POSTPROCESS_MAX_NEW_TOKENS", "512"))
POSTPROCESS_WORKERS = max(1, int(os.getenv("POSTPROCESS_WORKERS", "1")))
POSTPROCESS_SYNC_MAX_CHARS = int(os.getenv("POSTPROCESS_SYNC_MAX_CHARS", "1200"))
POSTPROCESS_SYNC_TIMEOUT_SECONDS = int(os.getenv("POSTPROCESS_SYNC_TIMEOUT_SECONDS", "120"))
CORRECTION_TASK_WORKERS = max(1, int(os.getenv("CORRECTION_TASK_WORKERS", "1")))
