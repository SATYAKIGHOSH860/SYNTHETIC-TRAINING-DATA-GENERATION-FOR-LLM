"""
Central configuration - paths, model id, and the Windows console UTF-8 fix.

Everything that can drift lives here so it changes in ONE place.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -----------------------------------------------------------------
# Resolved from this file, not the cwd, so modules work when run as
# `python src/generate.py` OR `python main.py` OR from the dashboard.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

DATASET_FILE = OUTPUT_DIR / "synthetic_dataset.jsonl"
CHATML_FILE = OUTPUT_DIR / "synthetic_dataset_chatml.jsonl"
REJECTED_FILE = OUTPUT_DIR / "rejected_pairs.jsonl"
# Every judged pair with its sub-scores, kept and rejected alike. Not part of
# the deliverable dataset - it exists so the threshold and dedup experiments
# can re-filter offline instead of paying for thousands of judge calls again.
GRADED_FILE = OUTPUT_DIR / "graded_pairs.jsonl"
# Abstention examples: questions the passage cannot answer, kept separately so
# the dashboard can show what they look like.
UNANSWERABLE_FILE = OUTPUT_DIR / "unanswerable_pairs.jsonl"

# --- Checkpoints -----------------------------------------------------------
# Kept apart from data/output because they are run scaffolding, not results:
# clearing the outputs to re-export must not throw away hours of API calls.
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
GEN_CHECKPOINT_FILE = CACHE_DIR / "generate_checkpoint.jsonl"
JUDGE_CHECKPOINT_FILE = CACHE_DIR / "validate_checkpoint.jsonl"
DUPLICATE_FILE = OUTPUT_DIR / "duplicate_pairs.jsonl"
STATS_FILE = OUTPUT_DIR / "pipeline_stats.json"

# --- Model -----------------------------------------------------------------
# THE single model constant. Change here, changes everywhere.
#
# Why not `llama-3.3-70b-versatile` (the id named in the project brief)?
# It was verified against the live Groq API on 2026-09-02 and no longer
# exists on this account - Groq retired the Llama chat models. Querying
# `client.models.list()` returned gpt-oss, qwen, compound and whisper only.
# `openai/gpt-oss-120b` is the largest general instruct model available and
# was smoke-tested to return clean JSON arrays for this exact task.
# If this 404s, run `python src/config.py` to print the live model list.
GROQ_MODEL = "openai/gpt-oss-120b"

# The judge is a DIFFERENT model to the generator, on purpose. A model grading
# its own output invites the obvious objection - self-preference bias - and
# with one model doing both, the judge handed out 6/6 to almost everything.
# An independent judge makes the pass rate mean something.
JUDGE_MODEL = "qwen/qwen3.8-27b"

# Why 0.4 for generation: high enough that three questions from one passage
# differ in angle and phrasing, low enough that the model stays anchored to
# the passage instead of inventing plausible-sounding clinical detail.
GEN_TEMPERATURE = 0.4

# Why 0.0 for judging: a grade must be reproducible. Re-running the judge on
# the same pair must give the same score, or the reported pass rate is noise.
JUDGE_TEMPERATURE = 0.0

# Groq free tier is ~30 requests/minute. 1.0s between calls keeps us under it
# with headroom for the occasional retry.
REQUEST_DELAY = 1.0

# --- Output caps -----------------------------------------------------------
# Groq admits a request only when prompt + max_tokens fits inside what is left
# of the tokens-per-minute budget, so an oversized cap does not just risk a
# long reply - it makes every call wait longer for admission. Measured on real
# runs: generation never exceeded ~423 completion tokens including hidden
# reasoning, and a judge reply is about 200.
GEN_MAX_TOKENS = 1024
JUDGE_MAX_TOKENS = 512

# gpt-oss models reason before answering. "low" keeps the quality on an
# extraction task like this one while cutting the hidden reasoning tokens that
# count against the same per-minute budget. Sent only to gpt-oss models - no
# other family on Groq accepts the parameter.
REASONING_EFFORT = "low"


def model_kwargs(model_id: str, max_tokens: int) -> dict:
    """
    Extra arguments for one chat completion, chosen by model family.

    reasoning_effort exists only on gpt-oss. Sending it to qwen - the judge -
    is a 400 from the API, so the family check is not a nicety.
    """
    kwargs = {"max_tokens": max_tokens}
    if "gpt-oss" in (model_id or "") and REASONING_EFFORT:
        kwargs["reasoning_effort"] = REASONING_EFFORT
    return kwargs

# --- Abstention training examples -----------------------------------------
# A model trained only on answerable questions learns that every question has
# an answer, and invents one when it does not know. A share of chunks also
# produce ONE plausible, on-topic question the passage does NOT answer, paired
# with a fixed refusal. That teaches the model to say "not in the document".
UNANSWERABLE_RATIO = 0.10

# Set here in code, never taken from the model, so every abstention example
# carries exactly the same wording and the model can actually learn it.
ABSTENTION_ANSWER = "This information is not available in the provided document."

# Which chunks get one is chosen with this seed, so a re-run picks the same
# chunks and the dataset is reproducible.
RUN_SEED = 42

# --- Secrets ---------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")


def get_groq_key() -> str:
    """Read GROQ_API_KEY, failing loudly with an actionable message."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            f"GROQ_API_KEY missing.\n"
            f"  Create {PROJECT_ROOT / '.env'} containing:\n"
            f"    GROQ_API_KEY=gsk_...\n"
            f"  Get a free key at https://console.groq.com/keys"
        )
    if key.startswith("gsk_your") or len(key) < 40:
        raise RuntimeError(
            f"GROQ_API_KEY looks like a placeholder (value starts '{key[:8]}', "
            f"length {len(key)}).\n"
            f"  A real Groq key starts 'gsk_' and is ~56 characters.\n"
            f"  Edit {PROJECT_ROOT / '.env'} with your real key."
        )
    return key


def use_utf8_console() -> None:
    """
    Force UTF-8 on stdout/stderr.

    Why this exists: Windows consoles default to cp1252. WHO guideline PDFs
    are full of non-breaking hyphens (U+2011), en-dashes and curly quotes, so
    a plain print() of generated text dies with UnicodeEncodeError and kills a
    20-minute run at the last step. Verified failure, not a hypothetical.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    use_utf8_console()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw PDFs     : {RAW_DIR}  ({len(list(RAW_DIR.glob('*.pdf')))} files)")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Model        : {GROQ_MODEL}")

    print("\nChecking API key...")
    try:
        key = get_groq_key()
        print(f"  OK - key present ({key[:8]}..., {len(key)} chars)")
    except RuntimeError as exc:
        print(f"  FAILED - {exc}")
        raise SystemExit(1)

    print("\nLive models available on this account:")
    from groq import Groq

    for model in sorted(m.id for m in Groq(api_key=key).models.list().data):
        marker = "  <-- in use" if model == GROQ_MODEL else ""
        print(f"  {model}{marker}")
