"""Thin facade over the LLM gateway (Groq + Gemini free tiers).

Provider modes: "auto" (gateway routing + fallback), "groq", "gemini".
"""

from __future__ import annotations

import threading

import config
from gateway import LLMError, LLMGateway, LLMResult  # re-exported

PROVIDERS = ("auto", "groq", "gemini")

__all__ = ["LLMError", "LLMGateway", "LLMResult", "PROVIDERS", "available_providers",
           "gateway_status", "generate", "get_gateway", "set_gateway"]

_gateway: LLMGateway | None = None
_lock = threading.Lock()


def get_gateway() -> LLMGateway:
    """Process-wide gateway so rate limits, breakers and cache are shared by all sessions."""
    global _gateway
    with _lock:
        if _gateway is None:
            _gateway = LLMGateway()
        return _gateway


def set_gateway(gateway: LLMGateway | None) -> None:
    """Replace the shared gateway (used by tests); None resets it."""
    global _gateway
    with _lock:
        _gateway = gateway


def available_providers() -> dict[str, bool]:
    """Which providers have an API key configured (never exposes the key)."""
    return {"groq": bool(config.groq_api_key()), "gemini": bool(config.gemini_api_key())}


def gateway_status() -> dict:
    return get_gateway().status()


def generate(system: str, user: str, provider: str | None = None) -> LLMResult:
    return get_gateway().generate(system, user, provider)
