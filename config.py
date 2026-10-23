
import os
from dotenv import load_dotenv

load_dotenv(override=False)


def get_setting(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


GROQ_API_KEY = get_setting("GROQ_API_KEY")
GROQ_MODEL = get_setting(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)
EMBED_MODEL = get_setting(
    "EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
RERANK_MODEL = get_setting(
    "RERANK_MODEL",
    "BAAI/bge-reranker-base",
)

LOG_LEVEL = get_setting("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = get_setting("LOG_TO_FILE", "false").lower() == "true"

MAX_FILE_SIZE_MB = 10
DEFAULT_TOP_K = 3
MAX_TOP_K = 5
CHUNK_SIZE = 900
CHUNK_OVERLAP = 160
