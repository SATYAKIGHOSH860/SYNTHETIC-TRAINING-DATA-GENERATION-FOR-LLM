"""Shared Groq plumbing for generate.py and validate.py.

One place for: building the chat model, pacing requests under the provider's
rate limits, retrying with exponential backoff, and turning the model's text
into JSON. Generation and judging both use it, so the behaviour described in
the spec ("same fence-stripping, backoff ... as Task 3") is literally the same
code.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.config import CONFIG, Config, require_api_key


class FatalLLMError(RuntimeError):
    """An error that retrying cannot fix: bad key, unknown model id, daily quota spent.

    Raised immediately (no backoff) so a run stops with a clear message instead
    of burning minutes producing nothing but failed chunks. Checkpoints written
    so far are kept, so the run can be resumed once the cause is fixed.
    """


class LLMCallError(RuntimeError):
    """All retries for one call failed (network, rate limit, unparseable output)."""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Remove a surrounding ```json ... ``` fence if present.

    Why: models wrap JSON in markdown fences despite being told not to.
    """
    if text is None:
        return ""
    text = text.strip()
    match = _FENCE.match(text)
    return match.group(1).strip() if match else text


def _first_balanced(text: str, opener: str, closer: str) -> str | None:
    """Return the first balanced ``opener...closer`` span, respecting JSON strings."""
    start = text.find(opener)
    while start != -1:
        depth, in_str, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find(opener, start + 1)
    return None


def parse_json(text: str, expect: type = list) -> Any:
    """Parse model output into ``expect`` (list or dict), tolerating fences and prose.

    Order: fence-stripped text as-is; then the first balanced JSON span of the
    expected kind embedded in surrounding prose. When a list is expected but
    the model returned ``{"pairs": [...]}``, the single list value is used.
    Raises ValueError if nothing usable is found.
    """
    cleaned = strip_code_fences(text)
    candidates = [cleaned]
    opener, closer = ("[", "]") if expect is list else ("{", "}")
    span = _first_balanced(cleaned, opener, closer)
    if span and span != cleaned:
        candidates.append(span)
    if expect is list:
        obj_span = _first_balanced(cleaned, "{", "}")
        if obj_span:
            candidates.append(obj_span)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, expect):
            return value
        if expect is list and isinstance(value, dict):
            lists = [v for v in value.values() if isinstance(v, list)]
            if len(lists) == 1:
                return lists[0]
    raise ValueError(f"no JSON {expect.__name__} found in model output: {cleaned[:120]!r}")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
class RateLimiter:
    """Thread-safe sliding-window limiter for requests/minute and tokens/minute.

    Why client-side pacing: on Groq's free tier the binding limit is tokens per
    minute (8k), not requests. Waiting just long enough before each call is
    faster and quieter than firing requests and sleeping through HTTP 429s.
    Groq admits a request only when ``prompt + max_tokens`` fits in the
    remaining budget, but charges only what is used; ``acquire`` mirrors that
    (admit on the worst case, record the expected use, settle to the actual).
    ``min_interval`` enforces ``model.request_delay_seconds`` between request
    starts. One limiter exists per model, shared by every thread in the
    process, so concurrent web jobs share the budget instead of colliding.
    """

    def __init__(self, requests_per_minute: int, tokens_per_minute: int, min_interval: float):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.min_interval = min_interval
        self._events: deque[list[float]] = deque()  # [start_time, tokens]
        self._last_start: float | None = None
        self._lock = threading.Lock()

    def acquire(self, admit_tokens: int, reserve_tokens: int | None = None) -> list[float]:
        """Block until a request needing ``admit_tokens`` of headroom may start.

        ``reserve_tokens`` (default: ``admit_tokens``) is what is recorded until
        ``settle`` replaces it with the real usage.
        """
        reserve = admit_tokens if reserve_tokens is None else reserve_tokens
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0][0] >= 60.0:
                    self._events.popleft()
                waits = [0.0]
                if self._last_start is not None:
                    waits.append(self._last_start + self.min_interval - now)
                if len(self._events) >= self.rpm:
                    waits.append(self._events[0][0] + 60.0 - now)
                used = sum(tokens for _, tokens in self._events)
                if used + admit_tokens > self.tpm:
                    remaining = used
                    for start, tokens in self._events:
                        remaining -= tokens
                        if remaining + admit_tokens <= self.tpm:
                            waits.append(start + 60.0 - now)
                            break
                wait = max(waits)
                if wait <= 0:
                    event = [now, float(reserve)]
                    self._events.append(event)
                    self._last_start = now
                    return event
            time.sleep(min(wait, 5.0))

    def settle(self, event: list[float], actual_tokens: int | None) -> None:
        """Replace the estimate with the real token count once the response arrives."""
        if actual_tokens:
            with self._lock:
                event[1] = float(actual_tokens)


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_limiter(model_id: str, cfg: Config = CONFIG) -> RateLimiter:
    with _LIMITERS_LOCK:
        if model_id not in _LIMITERS:
            _LIMITERS[model_id] = RateLimiter(
                cfg.model.requests_per_minute,
                cfg.model.tokens_per_minute,
                cfg.model.request_delay_seconds,
            )
        return _LIMITERS[model_id]


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token for English prose)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------
@dataclass
class Usage:
    calls: int = 0
    failed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.failed_calls += other.failed_calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        for model, n in other.by_model.items():
            self.by_model[model] = self.by_model.get(model, 0) + n

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_calls": self.calls,
            "failed_calls": self.failed_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_by_model": dict(self.by_model),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Usage":
        data = data or {}
        return cls(
            calls=data.get("api_calls", 0),
            failed_calls=data.get("failed_calls", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            by_model=dict(data.get("tokens_by_model", {})),
        )


# ---------------------------------------------------------------------------
# Model construction and invocation
# ---------------------------------------------------------------------------
_MODELS: dict[tuple[str, float, str], Any] = {}
_MODELS_LOCK = threading.Lock()


def _key_fingerprint(key: str) -> str:
    """Cache key for a client without keeping the API key itself as a dict key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def get_chat_model(model_id: str, temperature: float, cfg: Config = CONFIG, api_key: str | None = None,
                   max_tokens: int = 1024):
    """Build (once) a ChatGroq client for ``model_id`` at ``temperature`` with an output cap.

    ``reasoning_effort`` is only sent to gpt-oss models, the only ones on Groq
    that accept it. The SDK's own retry handles transient HTTP 429/5xx using
    the server's retry-after header; our backoff loop wraps that for everything
    else (timeouts, malformed JSON).
    """
    from langchain_groq import ChatGroq  # imported lazily: tests never need it

    key = require_api_key(api_key)
    cache_key = (model_id, float(temperature), int(max_tokens), _key_fingerprint(key))
    with _MODELS_LOCK:
        if cache_key not in _MODELS:
            kwargs: dict[str, Any] = dict(
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=cfg.model.request_timeout_seconds,
                max_retries=2,
                api_key=key,
            )
            if "gpt-oss" in model_id and cfg.model.reasoning_effort:
                kwargs["reasoning_effort"] = cfg.model.reasoning_effort
            _MODELS[cache_key] = ChatGroq(**kwargs)
        return _MODELS[cache_key]


def redact(message: str, api_key: str | None) -> str:
    key = api_key or ""
    if len(key) > 8:
        message = message.replace(key, "***")
    return re.sub(r"gsk_[A-Za-z0-9]{8,}", "gsk_***", message)


def _classify_fatal(exc: Exception, model_id: str) -> str | None:
    """Return a user-facing message if ``exc`` cannot be fixed by retrying."""
    name = type(exc).__name__
    text = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if name == "AuthenticationError" or status == 401 or "invalid api key" in text:
        return "Groq rejected the API key (HTTP 401). Check GROQ_API_KEY in .env."
    if name == "NotFoundError" or status == 404 or "model_not_found" in text or "does not exist" in text:
        return (
            f"Groq does not recognise the model id '{model_id}' (HTTP 404). Models are renamed "
            "regularly: check https://console.groq.com/docs/models and update config.yaml."
        )
    if (name == "RateLimitError" or status == 429) and ("per day" in text or "(tpd)" in text or "(rpd)" in text):
        return (
            f"The daily Groq quota for '{model_id}' is used up. Progress is checkpointed: after the quota "
            "resets, press Generate dataset again in the dashboard (or run main.py with --resume) to continue, "
            "or use a key with a higher tier."
        )
    if status == 413 or "request too large" in text:
        return f"Request too large for '{model_id}'. Lower validate.passage_chars or ingest.chunk_size."
    return None


def invoke_json(
    *,
    model_id: str,
    temperature: float,
    system: str,
    user: str,
    expect: type,
    expected_output_tokens: int,
    max_tokens: int,
    usage: Usage,
    cfg: Config = CONFIG,
    api_key: str | None = None,
) -> Any:
    """Call the model and parse its reply as JSON, with backoff retries.

    Retries ``model.max_retries`` times after the first attempt, sleeping 1s,
    2s, 4s, ... between attempts. Malformed JSON counts as a failed attempt:
    a second sample often parses. Raises FatalLLMError for non-retryable
    problems and LLMCallError once retries are exhausted.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_chat_model(model_id, temperature, cfg, api_key, max_tokens)
    limiter = get_limiter(model_id, cfg)
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    prompt_estimate = estimate_tokens(system + user)
    estimate = prompt_estimate + expected_output_tokens
    last_error: Exception | None = None
    for attempt in range(cfg.model.max_retries + 1):
        if attempt:
            time.sleep(2 ** (attempt - 1))
        event = limiter.acquire(prompt_estimate + max_tokens, estimate)
        usage.calls += 1
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            limiter.settle(event, estimate)
            fatal = _classify_fatal(exc, model_id)
            if fatal:
                raise FatalLLMError(fatal) from None
            last_error = exc
            usage.failed_calls += 1
            continue
        meta = getattr(response, "usage_metadata", None) or {}
        prompt_toks = int(meta.get("input_tokens", 0))
        completion_toks = int(meta.get("output_tokens", 0))
        total = prompt_toks + completion_toks or estimate
        limiter.settle(event, total)
        usage.prompt_tokens += prompt_toks
        usage.completion_tokens += completion_toks
        usage.by_model[model_id] = usage.by_model.get(model_id, 0) + total
        try:
            return parse_json(response.content, expect)
        except ValueError as exc:
            last_error = exc
            usage.failed_calls += 1
    raise LLMCallError(redact(f"{type(last_error).__name__}: {last_error}", api_key))
