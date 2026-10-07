"""
Token budget accounting for the Groq free tier.

Why this module exists: the first full run of this pipeline burned the entire
daily token allowance and stalled for four hours, leaving 141 of 263 pairs
ungraded. Nothing in the API told us in advance - the daily token cap is not
in the response headers, it only appears inside the 429 error body:

    Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199706

So the budget is estimated before the run and metered during it, and the run
stops cleanly rather than thrashing against a wall it cannot see.

Measured limits for this account (openai/gpt-oss-120b, free tier, 2026-09-03):
    requests/day    1,000
    tokens/day    200,000   <- the binding constraint
    tokens/minute   8,000
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    from config import UNANSWERABLE_RATIO
except ImportError:  # running as `python src/budget.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import UNANSWERABLE_RATIO

# Not exposed in headers - only in the 429 body. Override if the tier changes.
TOKENS_PER_DAY = 200_000
REQUESTS_PER_DAY = 1_000
TOKENS_PER_MINUTE = 8_000

# Calibrated 2026-09-19 against a real 11-chunk run: 17,624 tokens total,
# split generate 9,399 / judge 8,225. The pre-compression figures
# (900 / 620 / 140) described the old verbose prompts and overstated cost by
# ~44%; a first pass at the new ones (800 / 420 / 65) undershot the judge.
# These reproduce the measured ~1,600 tokens per chunk.
EST_TOKENS_PER_GENERATION = 860
# Batched judging: terse rubric + context sent once per chunk, plus the pairs.
EST_JUDGE_FIXED = 450
EST_JUDGE_PER_PAIR = 110

# A share of chunks also get an abstention example, which costs TWO extra
# calls: one to write the unanswerable question (~305 tokens - short rules,
# one-line reply) and one for the judge to verify it (~370 - its own two-part
# rubric plus the truncated passage). Both are folded into this one figure.
EST_TOKENS_PER_UNANSWERABLE = 680

# Python spends ~35s importing torch, langchain and the Groq client before the
# first request. Small runs are dominated by it, so the time a user is quoted
# has to include it or a "2 minute" run feels like a broken promise.
STARTUP_SECONDS = 35

# How many requests may be in flight at once. The ceiling is tokens/minute,
# not concurrency, so this only exists to stop request latency (~3s) from
# leaving the token budget idle while a single worker waits for a reply.
MAX_CONCURRENCY = 4


def estimate_run(chunks: int, questions_per_chunk: int = 3) -> dict:
    """Estimate tokens and requests for a run before spending anything."""
    pairs = chunks * questions_per_chunk
    # Abstention examples are one extra call on a fraction of chunks, so they
    # are costed as a fraction of a call on every chunk rather than tracked
    # per chunk - the totals come out the same and the arithmetic stays simple.
    abstain_calls = chunks * UNANSWERABLE_RATIO
    gen_tokens = int(
        chunks * EST_TOKENS_PER_GENERATION
        + abstain_calls * EST_TOKENS_PER_UNANSWERABLE
    )
    # One judge call per chunk, because pairs from a chunk share its passage.
    judge_tokens = chunks * (EST_JUDGE_FIXED + EST_JUDGE_PER_PAIR * questions_per_chunk)
    total = gen_tokens + judge_tokens
    return {
        "chunks": chunks,
        "estimated_pairs": pairs,
        "generation_tokens": gen_tokens,
        "judge_tokens": judge_tokens,
        "total_tokens": total,
        # Two extra calls per abstention chunk: write it, then verify it.
        "requests": int(chunks * 2 + abstain_calls * 2),
        "pct_of_daily_tokens": round(total / TOKENS_PER_DAY * 100, 1),
        "pct_of_daily_requests": round(
            (chunks * 2 + abstain_calls * 2) / REQUESTS_PER_DAY * 100, 1
        ),
        # At 8,000 tokens/minute the run cannot go faster than this, plus the
        # fixed cost of starting Python and loading the libraries.
        "min_minutes": round(total / TOKENS_PER_MINUTE + STARTUP_SECONDS / 60, 1),
    }


def max_affordable_chunks(
    tokens_available: int = TOKENS_PER_DAY, questions_per_chunk: int = 3
) -> int:
    """How many chunks fit in the remaining daily token budget."""
    per_chunk = (
        EST_TOKENS_PER_GENERATION
        + EST_TOKENS_PER_UNANSWERABLE * UNANSWERABLE_RATIO
        + EST_JUDGE_FIXED
        + EST_JUDGE_PER_PAIR * questions_per_chunk
    )
    return max(0, int(tokens_available / per_chunk))


def preflight(chunks: int, questions_per_chunk: int = 3, budget: int = TOKENS_PER_DAY) -> dict:
    """
    Print the cost of the planned run and whether it fits the daily budget.

    Returns the estimate so the caller can decide; it never exits on its own.
    """
    est = estimate_run(chunks, questions_per_chunk)
    affordable = max_affordable_chunks(budget, questions_per_chunk)

    print("-" * 62)
    print("TOKEN BUDGET PREFLIGHT")
    print("-" * 62)
    print(f"  Chunks planned        : {est['chunks']}")
    print(f"  Estimated pairs       : {est['estimated_pairs']}")
    print(f"  Estimated requests    : {est['requests']} of {REQUESTS_PER_DAY}/day"
          f"  ({est['pct_of_daily_requests']}%)")
    print(f"  Estimated tokens      : {est['total_tokens']:,} of {budget:,}/day"
          f"  ({est['pct_of_daily_tokens']}%)")
    print(f"  Floor on runtime      : ~{est['min_minutes']} min at {TOKENS_PER_MINUTE:,} tokens/min")

    if est["total_tokens"] > budget:
        print()
        print(f"  WARNING: this run needs {est['total_tokens']:,} tokens but only")
        print(f"  {budget:,} are available. It will stall part-way and leave pairs")
        print(f"  ungraded. Use --max-chunks {affordable} or fewer to fit.")
    else:
        headroom = budget - est["total_tokens"]
        print(f"  Fits, with {headroom:,} tokens spare "
              f"({affordable} chunks would be the maximum).")
    print("-" * 62)
    return est


class TokenBucket:
    """
    Paces requests against the tokens-per-minute ceiling.

    Replaces the old fixed `time.sleep(1.0)` between calls, which was wrong in
    both directions: too slow when the prompts are small (a second of dead
    time per request that the budget did not require), and far too fast once
    several requests landed in the same minute - which earned 429s whose
    backoff cost minutes, not seconds.

    A bucket holding one minute's tokens, refilling continuously, converges on
    the real limit instead of guessing at it. Thread-safe, so several workers
    can share one ceiling.
    """

    def __init__(self, tokens_per_minute: int = TOKENS_PER_MINUTE, safety: float = 0.9):
        # 10% headroom by default: the estimate before a call is approximate,
        # and overshooting the ceiling is far more expensive than undershooting.
        self.capacity = tokens_per_minute * safety
        self.refill_per_second = self.capacity / 60.0
        self._available = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _top_up(self) -> None:
        now = time.monotonic()
        self._available = min(
            self.capacity, self._available + (now - self._last) * self.refill_per_second
        )
        self._last = now

    def acquire(self, tokens: int) -> float:
        """Block until `tokens` are available. Returns seconds spent waiting."""
        waited = 0.0
        # Never ask for more than a full bucket, or this would deadlock.
        need = min(float(tokens), self.capacity)
        while True:
            with self._lock:
                self._top_up()
                if self._available >= need:
                    self._available -= need
                    return waited
                shortfall = need - self._available
                delay = shortfall / self.refill_per_second
            delay = min(max(delay, 0.05), 30.0)
            time.sleep(delay)
            waited += delay

    def refund(self, tokens: int) -> None:
        """Return unused budget when a call cost less than estimated."""
        if tokens <= 0:
            return
        with self._lock:
            self._top_up()
            self._available = min(self.capacity, self._available + tokens)


@dataclass
class TokenMeter:
    """
    Accumulates real token usage during a run and stops it before the wall.

    Groq returns exact usage on every response, so the estimate above is only
    needed before the first call - after that this is ground truth.
    """

    budget: int = TOKENS_PER_DAY
    used: int = 0
    requests: int = 0
    per_stage: dict = field(default_factory=dict)
    # Several workers record into this at once, so the counters need a lock -
    # without it two concurrent += updates can lose one another and the run
    # would under-count its own spending against the daily cap.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, usage, stage: str = "other") -> None:
        """Add one response's usage. Safe to call with None, and thread-safe."""
        total = getattr(usage, "total_tokens", 0) or 0
        with self._lock:
            self.used += total
            self.requests += 1
            self.per_stage[stage] = self.per_stage.get(stage, 0) + total

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    def exhausted(self, reserve: int = 3_000) -> bool:
        """
        True when too little budget remains to be worth continuing.

        `reserve` leaves room for the calls already in flight, so the run ends
        on its own terms instead of on a 429.
        """
        return self.remaining <= reserve

    def report(self) -> str:
        stages = ", ".join(f"{k} {v:,}" for k, v in sorted(self.per_stage.items()))
        return (
            f"{self.used:,} tokens over {self.requests} requests "
            f"({self.used / max(self.budget, 1) * 100:.1f}% of budget)"
            + (f" [{stages}]" if stages else "")
        )


def live_limits(api_key: str, model: str) -> dict:
    """
    Ask the API what is left right now.

    Only per-minute tokens and per-day requests come back in headers; the
    daily token count does not, which is exactly why TokenMeter exists.
    """
    import httpx

    try:
        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1},
            timeout=30,
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    headers = response.headers
    out = {
        "status": response.status_code,
        "requests_remaining": headers.get("x-ratelimit-remaining-requests"),
        "requests_limit": headers.get("x-ratelimit-limit-requests"),
        "tokens_per_min_remaining": headers.get("x-ratelimit-remaining-tokens"),
        "resets_in": headers.get("x-ratelimit-reset-requests"),
    }
    if response.status_code == 429:
        out["error_body"] = response.text[:300]
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import GROQ_MODEL, get_groq_key, use_utf8_console

    use_utf8_console()

    print("Estimated cost per run size (3 questions/chunk):\n")
    header = f"{'chunks':>7} {'pairs':>7} {'tokens':>10} {'% day':>7} {'requests':>9} {'min run':>9}"
    print(header)
    print("-" * len(header))
    for n in (25, 50, 75, 100, 150, 200, 400, 817):
        e = estimate_run(n)
        flag = "  <-- over budget" if e["total_tokens"] > TOKENS_PER_DAY else ""
        print(
            f"{e['chunks']:>7} {e['estimated_pairs']:>7} {e['total_tokens']:>10,} "
            f"{e['pct_of_daily_tokens']:>6}% {e['requests']:>9} {e['min_minutes']:>8}m{flag}"
        )

    print(f"\nMaximum chunks on a full daily budget: {max_affordable_chunks()}")

    print("\nLive limits right now:")
    for k, v in live_limits(get_groq_key(), GROQ_MODEL).items():
        print(f"  {k}: {v}")
