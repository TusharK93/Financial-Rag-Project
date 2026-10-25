"""LLM gateway: one entry point in front of Groq and Gemini free tiers.

Features (stdlib only):
  * Routing      - priority order, or round-robin to spread free-tier quota
  * Fallback     - next model of the same provider, then the next provider
  * Retries      - exponential backoff + jitter on transient errors only
  * Rate limits  - per-provider token bucket; a throttled provider is skipped, not blocked on
  * Circuit break- a failing provider is paused for a cooldown, auth errors pause longer
  * Cache        - small TTL/LRU cache for identical prompts
  * Metrics      - calls, failures, latency, cache hits per provider
SDKs are imported lazily so the app still starts if one is missing.
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import config
from logger import logger


class LLMError(RuntimeError):
    """Raised when no provider/model could produce an answer."""


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: int = 0
    cached: bool = False
    attempts: int = 1
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

AUTH, NOT_FOUND, RATE_LIMIT, TRANSIENT, OTHER = "auth", "not_found", "rate_limit", "transient", "other"


def classify_error(exc: Exception) -> str:
    status = None
    for attr in ("status_code", "code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            status = value
            break
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    name, msg = type(exc).__name__.lower(), str(exc).lower()

    if status in (401, 403) or "authentication" in name or "permissiondenied" in name or "api key not valid" in msg:
        return AUTH
    if "not configured" in msg or "not installed" in msg:
        return AUTH  # cannot succeed until the user fixes setup
    if status == 404 or "notfound" in name:
        return NOT_FOUND
    if status == 429 or "ratelimit" in name or "resource_exhausted" in msg or "quota" in msg:
        return RATE_LIMIT
    if (isinstance(status, int) and status >= 500) or any(
        k in name for k in ("timeout", "connection", "unavailable", "internalserver")
    ):
        return TRANSIENT
    return OTHER


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TokenBucket:
    """Requests-per-minute limiter. try_acquire never blocks."""

    def __init__(self, per_minute: int, clock=time.monotonic):
        self.capacity = max(int(per_minute), 0)
        self.tokens = float(self.capacity)
        self.rate = self.capacity / 60.0
        self.clock = clock
        self.updated = clock()
        self.lock = threading.Lock()

    def try_acquire(self) -> bool:
        if self.capacity == 0:  # 0 = unlimited
            return True
        with self.lock:
            now = self.clock()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown: float, clock=time.monotonic):
        self.threshold, self.cooldown, self.clock = max(threshold, 1), cooldown, clock
        self.failures = 0
        self.open_until = 0.0
        self.lock = threading.Lock()

    def available(self) -> bool:
        with self.lock:
            return self.clock() >= self.open_until

    def success(self):
        with self.lock:
            self.failures, self.open_until = 0, 0.0

    def failure(self, hard: bool = False):
        with self.lock:
            self.failures += 1
            if hard:  # e.g. bad API key: don't hammer the provider
                self.open_until = self.clock() + self.cooldown * 5
            elif self.failures >= self.threshold:
                self.open_until = self.clock() + self.cooldown


class TTLCache:
    def __init__(self, size: int, ttl: float, clock=time.monotonic):
        self.size, self.ttl, self.clock = size, ttl, clock
        self.data: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.size > 0 and self.ttl > 0

    def get(self, key):
        if not self.enabled:
            return None
        with self.lock:
            item = self.data.get(key)
            if item is None:
                return None
            expires, value = item
            if self.clock() > expires:
                del self.data[key]
                return None
            self.data.move_to_end(key)
            return value

    def put(self, key, value):
        if not self.enabled:
            return
        with self.lock:
            self.data[key] = (self.clock() + self.ttl, value)
            self.data.move_to_end(key)
            while len(self.data) > self.size:
                self.data.popitem(last=False)


@dataclass
class ProviderStats:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    skipped: int = 0
    total_latency_ms: int = 0
    last_error: str = ""

    @property
    def avg_latency_ms(self) -> int:
        return self.total_latency_ms // self.successes if self.successes else 0


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class Provider:
    name = "base"

    def __init__(self, models: list[str], rpm: int):
        self.models = [m for m in models if m]
        self.limiter = TokenBucket(rpm)
        self.breaker = CircuitBreaker(config.CIRCUIT_FAILURES, config.CIRCUIT_COOLDOWN_SECONDS)
        self.stats = ProviderStats()

    def configured(self) -> bool:
        raise NotImplementedError

    def complete(self, system: str, user: str, model: str) -> str:
        raise NotImplementedError


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, models=None, rpm=None):
        super().__init__(
            models or [config.GROQ_MODEL, *config.GROQ_FALLBACK_MODELS],
            config.GROQ_RPM if rpm is None else rpm,
        )

    def configured(self) -> bool:
        return bool(config.groq_api_key())

    def complete(self, system: str, user: str, model: str) -> str:
        api_key = config.groq_api_key()
        if not api_key:
            raise LLMError("GROQ_API_KEY is not configured.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMError("The 'groq' package is not installed.") from exc
        client = Groq(api_key=api_key, timeout=config.LLM_TIMEOUT_SECONDS, max_retries=0)
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise LLMError("Groq returned an empty answer.")
        return text


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, models=None, rpm=None):
        super().__init__(
            models or [config.GEMINI_MODEL, *config.GEMINI_FALLBACK_MODELS],
            config.GEMINI_RPM if rpm is None else rpm,
        )

    def configured(self) -> bool:
        return bool(config.gemini_api_key())

    def complete(self, system: str, user: str, model: str) -> str:
        api_key = config.gemini_api_key()
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not configured.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError("The 'google-genai' package is not installed.") from exc
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=config.LLM_TIMEOUT_SECONDS * 1000),
        )
        response = client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.1,
                max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            ),
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty or blocked answer.")
        return text


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

class LLMGateway:
    def __init__(
        self,
        providers: dict[str, Provider] | None = None,
        order: list[str] | None = None,
        strategy: str | None = None,
        max_retries: int | None = None,
        backoff: float | None = None,
        cache: TTLCache | None = None,
        sleep=time.sleep,
    ):
        self.providers = providers or {"groq": GroqProvider(), "gemini": GeminiProvider()}
        self.order = [p for p in (order or config.LLM_PROVIDER_ORDER) if p in self.providers] or list(self.providers)
        self.strategy = (strategy or config.LLM_STRATEGY)
        self.max_retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
        self.backoff = config.LLM_BACKOFF_SECONDS if backoff is None else backoff
        self.cache = cache or TTLCache(config.LLM_CACHE_SIZE, config.LLM_CACHE_TTL_SECONDS)
        self.sleep = sleep
        self.cache_hits = 0
        self.requests = 0
        self._rr = 0
        self._lock = threading.Lock()

    # -- routing ---------------------------------------------------------
    def _route(self, provider: str | None) -> list[str]:
        provider = (provider or config.LLM_PROVIDER or "auto").lower()
        if provider in self.providers:
            return [provider]  # explicitly pinned: no cross-provider fallback
        order = list(self.order)
        if self.strategy == "round_robin" and len(order) > 1:
            with self._lock:
                shift = self._rr % len(order)
                self._rr += 1
            order = order[shift:] + order[:shift]
        return order

    @staticmethod
    def _key(system: str, user: str, route: list[str]) -> str:
        return hashlib.sha256("\x00".join([",".join(route), system, user]).encode()).hexdigest()

    # -- main entry ------------------------------------------------------
    def generate(self, system: str, user: str, provider: str | None = None) -> LLMResult:
        with self._lock:
            self.requests += 1
        route = self._route(provider)

        key = self._key(system, user, sorted(route))
        hit = self.cache.get(key)
        if hit is not None:
            with self._lock:
                self.cache_hits += 1
            return LLMResult(hit.text, hit.provider, hit.model, 0, True, 0, hit.fallback_used)

        errors: list[str] = []
        attempts = 0
        started = time.perf_counter()
        first = route[0]

        for name in route:
            prov = self.providers[name]
            if not prov.configured():
                errors.append(f"{name}: not configured")
                prov.stats.skipped += 1
                continue
            if not prov.breaker.available():
                errors.append(f"{name}: circuit open")
                prov.stats.skipped += 1
                continue

            for model in prov.models:
                abandon_provider = False
                for attempt in range(self.max_retries + 1):
                    if not prov.limiter.try_acquire():
                        errors.append(f"{name}: client rate limit reached")
                        prov.stats.skipped += 1
                        abandon_provider = True
                        break
                    attempts += 1
                    prov.stats.calls += 1
                    t0 = time.perf_counter()
                    try:
                        text = prov.complete(system, user, model)
                    except Exception as exc:
                        kind = classify_error(exc)
                        prov.stats.failures += 1
                        prov.stats.last_error = f"{type(exc).__name__} ({kind})"
                        errors.append(f"{name}/{model}: {type(exc).__name__} [{kind}]")
                        logger.warning("Gateway: %s/%s failed: %s [%s]", name, model, type(exc).__name__, kind)
                        if kind == AUTH:
                            prov.breaker.failure(hard=True)
                            abandon_provider = True
                            break
                        if kind == TRANSIENT and attempt < self.max_retries:
                            self.sleep(self.backoff * (2 ** attempt) * (0.5 + random.random()))
                            continue
                        prov.breaker.failure()
                        break  # next model (rate limit / not found / other / retries exhausted)
                    else:
                        latency = int((time.perf_counter() - t0) * 1000)
                        prov.stats.successes += 1
                        prov.stats.total_latency_ms += latency
                        prov.breaker.success()
                        result = LLMResult(
                            text, name, model,
                            int((time.perf_counter() - started) * 1000),
                            False, attempts, name != first,
                        )
                        self.cache.put(key, result)
                        logger.info("Gateway: answered by %s/%s in %d ms (%d attempt(s))",
                                    name, model, result.latency_ms, attempts)
                        return result
                if abandon_provider or not prov.breaker.available():
                    break

        raise LLMError("; ".join(errors) or "No LLM provider is configured.")

    # -- observability ---------------------------------------------------
    def status(self) -> dict:
        out = {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "strategy": self.strategy,
            "order": self.order,
            "providers": {},
        }
        for name, p in self.providers.items():
            out["providers"][name] = {
                "configured": p.configured(),
                "circuit": "closed" if p.breaker.available() else "open",
                "models": p.models,
                "calls": p.stats.calls,
                "successes": p.stats.successes,
                "failures": p.stats.failures,
                "skipped": p.stats.skipped,
                "avg_latency_ms": p.stats.avg_latency_ms,
                "last_error": p.stats.last_error,
            }
        return out
