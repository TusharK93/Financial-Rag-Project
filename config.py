
"""Central configuration. Values are read from environment variables / .env."""

import os
from dotenv import load_dotenv

load_dotenv(override=False)


def get_setting(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# --- LLM providers (both have free tiers) -----------------------------------
# Get a free Groq key: https://console.groq.com/keys
# Get a free Gemini key: https://aistudio.google.com/apikey

GROQ_API_KEY = get_setting("GROQ_API_KEY")
GROQ_MODEL = get_setting("GROQ_MODEL", "openai/gpt-oss-20b")
GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-3.8-flash")

# auto = Groq first, then Gemini if Groq is missing/failing.
# Also: groq | gemini
LLM_PROVIDER = get_setting("LLM_PROVIDER", "auto").lower()
LLM_TIMEOUT_SECONDS = int(get_setting("LLM_TIMEOUT_SECONDS", "30") or 30)
LLM_MAX_OUTPUT_TOKENS = int(
    get_setting("LLM_MAX_OUTPUT_TOKENS", "1024") or 1024
)


def _num(name: str, default: float, cast=float):
    try:
        return cast(get_setting(name, str(default)))
    except ValueError:
        return default


# --- LLM gateway -------------------------------------------------------------
# Provider priority for "auto" mode, comma separated (groq, gemini).
LLM_PROVIDER_ORDER = [
    p
    for p in get_setting(
        "LLM_PROVIDER_ORDER", "groq,gemini"
    ).lower().split(",")
    if p.strip()
]

# priority = always try in order | round_robin = spread load to save free quota
LLM_STRATEGY = get_setting("LLM_STRATEGY", "priority").lower()

# Extra models tried when the primary model is rate limited or missing.
GROQ_FALLBACK_MODELS = [
    m.strip()
    for m in get_setting(
        "GROQ_FALLBACK_MODELS", "llama-3.1-8b-instant"
    ).split(",")
    if m.strip()
]

GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in get_setting(
        "GEMINI_FALLBACK_MODELS", "gemini-flash-lite-latest"
    ).split(",")
    if m.strip()
]

LLM_MAX_RETRIES = _num("LLM_MAX_RETRIES", 1, int)
LLM_BACKOFF_SECONDS = _num("LLM_BACKOFF_SECONDS", 0.5)

# Client-side rate limits (requests/minute).
GROQ_RPM = _num("GROQ_RPM", 25, int)
GEMINI_RPM = _num("GEMINI_RPM", 10, int)

# Circuit breaker.
CIRCUIT_FAILURES = _num("LLM_CIRCUIT_FAILURES", 3, int)
CIRCUIT_COOLDOWN_SECONDS = _num(
    "LLM_CIRCUIT_COOLDOWN_SECONDS", 60.0
)

# Response cache. Set to 0 to disable.
LLM_CACHE_TTL_SECONDS = _num(
    "LLM_CACHE_TTL_SECONDS", 600, int
)
LLM_CACHE_SIZE = _num("LLM_CACHE_SIZE", 128, int)


def groq_api_key() -> str:
    return get_setting("GROQ_API_KEY")


def gemini_api_key() -> str:
    return get_setting("GEMINI_API_KEY") or get_setting("GOOGLE_API_KEY")


# --- Retrieval ----------------------------------------------------------------
EMBED_MODEL = get_setting(
    "EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
RERANK_MODEL = get_setting(
    "RERANK_MODEL",
    "BAAI/bge-reranker-base",
)

# --- Guardrails ---------------------------------------------------------------
GUARDRAILS_ENABLED = (
    get_setting("GUARDRAILS_ENABLED", "true").lower() != "false"
)

# --- Logging / limits ---------------------------------------------------------
LOG_LEVEL = get_setting("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = get_setting("LOG_TO_FILE", "false").lower() == "true"

MAX_FILE_SIZE_MB = 10
DEFAULT_TOP_K = 3
MAX_TOP_K = 5
CHUNK_SIZE = 900
CHUNK_OVERLAP = 160
