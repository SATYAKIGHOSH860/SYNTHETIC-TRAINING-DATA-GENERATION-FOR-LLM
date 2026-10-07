"""
Shared progress file, written by the pipeline and read by the dashboard.

The pipeline runs as a separate process so the dashboard stays responsive, so
the two cannot talk directly. They communicate through one small JSON file that
the pipeline overwrites as it goes.

Every write includes a heartbeat timestamp. If a run dies without saying so -
killed, crashed, machine slept - the heartbeat goes stale and the dashboard can
tell the difference between "still working" and "gone".
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

try:
    from config import OUTPUT_DIR
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import OUTPUT_DIR

PROGRESS_FILE = OUTPUT_DIR / "run_progress.json"

# The five pipeline stages, with the share of total time each typically takes.
# Used to turn per-stage progress into one overall percentage. Generation and
# validation dominate because both are API-bound; the rest are local and fast.
STAGE_WEIGHTS = {
    "ingest": 0.03,
    "generate": 0.45,
    "validate": 0.45,
    "deduplicate": 0.04,
    "export": 0.03,
}
STAGES = list(STAGE_WEIGHTS)

# A run whose heartbeat is older than this is presumed dead. Generous, because
# a single rate-limit backoff can legitimately block for ~5 minutes.
STALE_AFTER_SECONDS = 420


def write(
    stage: str,
    current: int = 0,
    total: int = 0,
    message: str = "",
    status: str = "running",
    extra: dict | None = None,
) -> None:
    """Overwrite the progress file. Never raises - progress must not kill a run."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": stage,
            "current": current,
            "total": total,
            "message": message,
            "status": status,
            "heartbeat": time.time(),
            "pid": os.getpid(),
            "overall_pct": _overall_pct(stage, current, total, status),
        }
        if extra:
            payload.update(extra)
        tmp = PROGRESS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        # Atomic replace, so the dashboard never reads a half-written file.
        os.replace(tmp, PROGRESS_FILE)
    except Exception:
        pass


PLAN_FILE = OUTPUT_DIR / "run_plan.json"

# Measured on this machine, 2026-09-19.
STARTUP_SECONDS = 35.0          # importing torch, langchain, groq before step 1
SECONDS_PER_CHUNK_GENERATE = 5.4  # token-paced: ~860 tokens at 7,200/min
SECONDS_PER_CHUNK_JUDGE = 4.8     # token-paced: ~780 tokens at 7,200/min
DEDUP_FIXED_SECONDS = 12.0      # loading MiniLM dominates; encoding is fast
EXPORT_SECONDS = 4.0
SECONDS_PER_PDF_PAGE = 1.6      # pypdf extract_text, measured on a 592-page WHO PDF


def set_plan(chunks: int, questions_per_chunk: int = 3, pages: int = 0) -> None:
    """
    Record how long each stage is expected to take, in seconds.

    Fixed percentage weights cannot be accurate for both a 12-chunk run and a
    45-chunk one, because the ~35 second Python startup is a constant: it is
    23% of a three-minute run and 7% of a nine-minute one. Estimating real
    seconds per stage from the chunk count makes the bar mean the same thing
    at any run size.
    """
    # Ingest is no longer negligible: reading a PDF costs ~1.6s per page on
    # top of the fixed Python startup, so a 20-page sample is ~32s - a real
    # share of a three-minute run, and the bar has to reflect that or it sits
    # at zero through the slowest opening minute.
    plan = {
        "chunks": chunks,
        "pages": pages,
        "ingest": STARTUP_SECONDS + pages * SECONDS_PER_PDF_PAGE,
        "generate": chunks * SECONDS_PER_CHUNK_GENERATE,
        "validate": chunks * SECONDS_PER_CHUNK_JUDGE,
        "deduplicate": DEDUP_FIXED_SECONDS,
        "export": EXPORT_SECONDS,
    }
    plan["total"] = sum(plan[s] for s in STAGES)
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PLAN_FILE.write_text(json.dumps(plan), encoding="utf-8")
    except Exception:
        pass


def _read_plan() -> dict | None:
    try:
        return json.loads(PLAN_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _overall_pct(stage: str, current: int, total: int, status: str) -> float:
    """
    Blend per-stage progress into a single 0-100 figure.

    Uses the second-based plan when one exists, falling back to the fixed
    weights for runs started before a plan was written.
    """
    if status == "done":
        return 100.0
    if stage not in STAGES:
        return 0.0

    within = (current / total) if total else 0.0
    plan = _read_plan()

    if plan and plan.get("total"):
        before = sum(plan.get(s, 0.0) for s in STAGES[: STAGES.index(stage)])
        spent = before + within * plan.get(stage, 0.0)
        return round(min(100.0, spent / plan["total"] * 100), 1)

    done = sum(
        w for s, w in STAGE_WEIGHTS.items() if STAGES.index(s) < STAGES.index(stage)
    )
    return round(min(100.0, (done + within * STAGE_WEIGHTS.get(stage, 0.0)) * 100), 1)


def interpolated_pct(prog: dict, elapsed: float) -> float:
    """
    The bar position, filling in the opening phase from elapsed time.

    Python spends ~35 seconds importing torch and langchain before the
    pipeline can report anything at all, and during that window `current` and
    `total` are both zero - so the honest percentage is zero, and the bar sits
    dead for the slowest minute of a three-minute run. Here the ingest stage
    is interpolated against its planned duration instead, which is a fair
    estimate rather than a fake one: it is the same plan the ETA uses, and it
    stops as soon as real page counts start arriving.
    """
    reported = float(prog.get("overall_pct", 0.0) or 0.0)
    if prog.get("status") != "running" or prog.get("stage") != "ingest":
        return reported
    if prog.get("total"):
        return reported  # real page progress has begun; trust it

    plan = _read_plan()
    if not plan or not plan.get("total") or not plan.get("ingest"):
        return reported

    share = plan["ingest"] / plan["total"] * 100
    # Cap at 90% of the ingest share so the bar never overtakes reality and
    # then has to wait; the remaining tenth fills when real pages report.
    return round(min(elapsed / plan["ingest"], 0.9) * share, 1)


def eta_seconds(stage: str, current: int, total: int, elapsed: float) -> float | None:
    """
    Seconds still to go, blending the plan with how this run is actually going.

    Pure estimate drifts when the API is slow; pure measurement is useless in
    the first seconds when almost nothing has finished. Weighting the measured
    rate in as progress accumulates gives a figure that starts sane and gets
    steadily more truthful.
    """
    plan = _read_plan()
    if not plan or not plan.get("total") or stage not in STAGES:
        return None

    before = sum(plan.get(s, 0.0) for s in STAGES[: STAGES.index(stage)])
    within = (current / total) if total else 0.0
    planned_spent = before + within * plan.get(stage, 0.0)
    if planned_spent <= 1:
        return max(plan["total"] - elapsed, 0.0)

    fraction = planned_spent / plan["total"]
    measured_total = elapsed / fraction
    # Trust the measurement more as the run progresses.
    blended = plan["total"] * (1 - fraction) + measured_total * fraction
    return max(blended - elapsed, 0.0)


def read() -> dict:
    """Return the current progress, or an empty dict if there is none."""
    try:
        if not PROGRESS_FILE.exists():
            return {}
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

    age = time.time() - data.get("heartbeat", 0)
    data["age_seconds"] = round(age, 1)
    # Only a "running" record can go stale; finished ones stay as they are.
    data["stale"] = data.get("status") == "running" and age > STALE_AFTER_SECONDS
    return data


def clear() -> None:
    """
    Remove the progress file AND the timing plan.

    Dropping the plan matters: it is sized for a particular chunk count, and a
    leftover plan from a 45-chunk run made a fresh 8-chunk run report "8.4
    minutes remaining" from its first second.
    """
    for path in (PROGRESS_FILE, PLAN_FILE):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def finish(message: str = "Complete", extra: dict | None = None) -> None:
    write("export", 1, 1, message, status="done", extra=extra)


def fail(message: str) -> None:
    write("error", 0, 0, message[:500], status="error")
