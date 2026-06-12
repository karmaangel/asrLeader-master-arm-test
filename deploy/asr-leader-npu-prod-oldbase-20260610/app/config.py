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


RESOLVE_LOCAL_ASR_MODELS = _bool_env("ASR_RESOLVE_LOCAL_MODELS", True)

ASR_MODEL_ALIAS = os.getenv("ASR_MODEL_ALIAS", "paraformer-zh")
VAD_MODEL_ALIAS = os.getenv("VAD_MODEL_ALIAS", "fsmn-vad")
PUNC_MODEL_ALIAS = os.getenv("PUNC_MODEL_ALIAS", "ct-punc")
SPK_MODEL_ALIAS = os.getenv("SPK_MODEL_ALIAS", "cam++")


def _resolve_model(env_key: str, local_name: str) -> str:
    value = os.getenv(env_key)
    if value:
        return value
    if RESOLVE_LOCAL_ASR_MODELS:
        local_path = MODELS_DIR / local_name
        if local_path.is_dir():
            return str(local_path)
    return local_name


ASR_MODEL = _resolve_model("ASR_MODEL", ASR_MODEL_ALIAS)
VAD_MODEL = _resolve_model("VAD_MODEL", VAD_MODEL_ALIAS)
PUNC_MODEL = _resolve_model("PUNC_MODEL", PUNC_MODEL_ALIAS)
SPK_MODEL = _resolve_model("SPK_MODEL", SPK_MODEL_ALIAS)

HOST = os.getenv("ASR_HOST", "0.0.0.0")
PORT = int(os.getenv("ASR_PORT", "8000"))
DEVICE = os.getenv("ASR_DEVICE", "auto")
BATCH_SIZE_S = int(os.getenv("BATCH_SIZE_S", "300"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(1024 * 1024 * 1024)))
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac"}
ALLOWED_AUDIO_FORMATS = ["WAV", "MP3", "M4A", "AAC"]
ASR_HOTWORDS = os.getenv("ASR_HOTWORDS", "")
LEADER_DB_PATH = Path(os.getenv("LEADER_DB_PATH", str(DATA_DIR / "leaders.json")))
LEADER_THRESHOLD = float(os.getenv("LEADER_THRESHOLD", "0.45"))
LEADER_SPEAKER_THRESHOLD = float(os.getenv("LEADER_SPEAKER_THRESHOLD", "0.35"))
LEADER_SEGMENT_THRESHOLD = float(os.getenv("LEADER_SEGMENT_THRESHOLD", "0.50"))
LEADER_SCORE_MARGIN = float(os.getenv("LEADER_SCORE_MARGIN", "0.03"))
LEADER_MIN_SEGMENT_SECONDS = float(os.getenv("LEADER_MIN_SEGMENT_SECONDS", "1.0"))
SEGMENT_PADDING_MS = int(os.getenv("SEGMENT_PADDING_MS", "250"))
SPEAKER_AUTO_MERGE_THRESHOLD = float(os.getenv("SPEAKER_AUTO_MERGE_THRESHOLD", "0.50"))
SPEAKER_AUTO_MERGE_MARGIN = float(os.getenv("SPEAKER_AUTO_MERGE_MARGIN", "0.10"))
SPEAKER_AUTO_MERGE_MAX_SECONDS = float(os.getenv("SPEAKER_AUTO_MERGE_MAX_SECONDS", "30.0"))
SPEAKER_AUTO_MERGE_MAX_RATIO = float(os.getenv("SPEAKER_AUTO_MERGE_MAX_RATIO", "0.08"))
SPEAKER_JITTER_MAX_SECONDS = float(os.getenv("SPEAKER_JITTER_MAX_SECONDS", "3.0"))
SPEAKER_JITTER_MAX_GAP_SECONDS = float(os.getenv("SPEAKER_JITTER_MAX_GAP_SECONDS", "0.8"))
SPEAKER_JITTER_MIN_SIMILARITY = float(os.getenv("SPEAKER_JITTER_MIN_SIMILARITY", "0.45"))
SPEAKER_JITTER_MARGIN = float(os.getenv("SPEAKER_JITTER_MARGIN", "0.08"))
SEGMENT_MERGE_MAX_GAP_SECONDS = float(os.getenv("SEGMENT_MERGE_MAX_GAP_SECONDS", "2.0"))
LEADER_RELATIVE_MIN_SCORE = float(os.getenv("LEADER_RELATIVE_MIN_SCORE", "0.12"))
LEADER_RELATIVE_MARGIN = float(os.getenv("LEADER_RELATIVE_MARGIN", "0.10"))
LEADER_RELATIVE_SUPPORT_SEGMENTS = int(os.getenv("LEADER_RELATIVE_SUPPORT_SEGMENTS", "3"))
LEADER_ENROLLMENT_MIN_SPEECH_SECONDS = float(os.getenv("LEADER_ENROLLMENT_MIN_SPEECH_SECONDS", "5.0"))
LEADER_ENROLLMENT_MIN_SIMILARITY = float(os.getenv("LEADER_ENROLLMENT_MIN_SIMILARITY", "0.60"))
LEADER_ENROLLMENT_SPEECH_DBFS = float(os.getenv("LEADER_ENROLLMENT_SPEECH_DBFS", "-50"))

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
