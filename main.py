"""
Run the whole pipeline: PDFs -> chunks -> Q&A -> judge -> dedup -> JSONL.

    python main.py                    # ~3.5 min, 20 chunks, ~55 pairs
    python main.py --max-chunks 60    # ~10 min, a larger dataset
    python main.py --min-score 5      # stricter quality filter

Two Groq free-tier limits shape this: 8,000 tokens/MINUTE sets the wall-clock
floor, and 200,000 tokens/DAY caps the total. Requests are paced by a shared
token bucket and run four at a time, so a run finishes close to its token
floor rather than well behind it. `python src/budget.py` prints the table.

Tune the constants below, or override any of them from the command line.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import OUTPUT_DIR, REQUEST_DELAY, use_utf8_console  # noqa: E402

# --- Pipeline parameters ---------------------------------------------------
QUESTIONS_PER_CHUNK = 3
MIN_QUALITY_SCORE = 4  # of 6; pairs below this are rejected
# Imported, not redeclared: a second copy of this number here silently
# overrode the module's own default and the run used 0.85 while the code and
# the comments said 0.90.
from deduplicate import SIMILARITY_THRESHOLD  # noqa: E402
# Wall-clock is governed by the 8,000 tokens/minute ceiling, not by CPU: at a
# measured ~1,415 tokens per chunk, runtime is roughly chunks / 5.7 minutes.
# 20 chunks lands near 3.5 minutes and ~55 pairs, which is a usable dataset
# from a standing start. Raise it for a bigger corpus and expect the time to
# scale linearly - `python src/budget.py` prints the table.
MAX_CHUNKS = 20  # 0 = the whole corpus
PUSH_TO_HF = False  # never automatic - pushing is a deliberate act
HF_REPO_ID = "your-username/who-guidelines-synthetic-qa"


def select_chunks(chunks: list, max_chunks: int) -> list:
    """
    Take an evenly spaced sample rather than the first N.

    Why: chunks arrive grouped by document, so `chunks[:100]` would come
    entirely from the first PDF - the per-source chart would show one bar and
    the test run would tell you nothing about the other two guidelines.
    Striding across the corpus keeps all three sources represented.
    """
    if not max_chunks or max_chunks >= len(chunks):
        return chunks
    step = len(chunks) / max_chunks
    return [chunks[int(i * step)] for i in range(max_chunks)]


def run(
    questions_per_chunk: int = QUESTIONS_PER_CHUNK,
    min_quality_score: int = MIN_QUALITY_SCORE,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    max_chunks: int = MAX_CHUNKS,
    push_to_hf: bool = PUSH_TO_HF,
    hf_repo_id: str = HF_REPO_ID,
    delay: float = REQUEST_DELAY,
    token_budget: int = 0,
    resume: bool = True,
) -> dict:
    import checkpoint
    import progress
    from config import GEN_CHECKPOINT_FILE, JUDGE_CHECKPOINT_FILE
    from budget import TOKENS_PER_DAY, TokenBucket, TokenMeter, preflight
    from deduplicate import deduplicate_pairs
    from export import export_all, push_to_hub
    from generate import generate_pairs
    from ingest import chunk_documents, load_documents
    from validate import validate_pairs

    if not resume:
        checkpoint.clear(GEN_CHECKPOINT_FILE, JUDGE_CHECKPOINT_FILE)
        print("--fresh: checkpoints cleared, everything will be re-run")

    started = time.time()

    def banner(step: str, title: str) -> None:
        print("\n" + "=" * 70)
        print(f"STEP {step}: {title}")
        print("=" * 70)

    progress.write("ingest", 0, 1, "Reading documents")
    banner("1/5", "INGEST - documents to chunks")

    # Only read as many pages as the run can actually use. A page yields
    # roughly 3 chunks of 600 characters, a quarter are lost to boilerplate,
    # and a margin covers documents that are sparser than average. Reading a
    # 592-page PDF in full costs 15.6 minutes of text extraction for a run
    # that consumes 25 chunks of it.
    page_budget = None
    if max_chunks:
        page_budget = max(8, int(max_chunks / 3 / 0.75 * 1.8))

    chunks = chunk_documents(
        load_documents(
            page_budget=page_budget,
            on_progress=lambda done, total, name: progress.write(
                "ingest", done, total, f"Reading {name}  ({done}/{total} pages)"
            ),
        )
    )
    chunks = select_chunks(chunks, max_chunks)
    print(f"\nProcessing {len(chunks)} chunks this run")

    # Estimate the cost before spending anything, then meter the real usage.
    # The free tier's daily token cap is invisible until a 429 announces it,
    # so the run has to track its own spending.
    # Now that the chunk count is known, record how long each stage should
    # take so the dashboard's bar and ETA mean something.
    progress.set_plan(len(chunks), questions_per_chunk, pages=page_budget or 0)

    budget_limit = token_budget or TOKENS_PER_DAY
    preflight(len(chunks), questions_per_chunk, budget=budget_limit)
    meter = TokenMeter(budget=budget_limit)
    # One bucket for the whole run: generation and judging draw on the same
    # tokens-per-minute ceiling, so they must share one pacer or they would
    # each assume the full rate and together exceed it.
    bucket = TokenBucket()

    progress.write("generate", 0, len(chunks), "Generating Q&A pairs")
    banner("2/5", "GENERATE - chunks to Q&A pairs")
    raw_pairs = generate_pairs(
        chunks,
        questions_per_chunk,
        delay=delay,
        meter=meter,
        bucket=bucket,
        progress_cb=lambda i, n, msg: progress.write("generate", i, n, msg),
        resume=resume,
    )
    if not raw_pairs:
        raise RuntimeError(
            "No pairs generated. Check GROQ_API_KEY and the model id in src/config.py "
            "(run `python src/config.py` to list live models)."
        )

    progress.write("validate", 0, len(raw_pairs), "Judging pairs")
    banner("3/5", "VALIDATE - LLM-as-judge scoring")
    validated, rejected = validate_pairs(
        raw_pairs,
        min_score=min_quality_score,
        delay=delay,
        meter=meter,
        bucket=bucket,
        progress_cb=lambda i, n, msg: progress.write("validate", i, n, msg),
        resume=resume,
    )
    if not validated:
        raise RuntimeError(
            f"Every pair was rejected at threshold {min_quality_score}/6. "
            f"Inspect {OUTPUT_DIR / 'rejected_pairs.jsonl'} before lowering it."
        )

    progress.write("deduplicate", 0, 1, "Embedding questions and removing duplicates")
    banner("4/5", "DEDUPLICATE - semantic near-duplicate removal")
    unique, duplicates = deduplicate_pairs(validated, threshold=similarity_threshold)

    progress.write("export", 0, 1, "Writing dataset and statistics")
    banner("5/5", "EXPORT - JSONL, ChatML and statistics")
    stats = export_all(
        final_pairs=unique,
        raw_pairs=raw_pairs,
        validated_pairs=validated,
        rejected_pairs=rejected,
        duplicates=duplicates,
        chunks_processed=len(chunks),
        min_quality_score=min_quality_score,
        similarity_threshold=similarity_threshold,
        # So pipeline_stats.json records what the run really cost, not an
        # estimate. The dashboard shows these in the sidebar.
        meter=meter,
        wall_seconds=time.time() - started,
    )

    elapsed = time.time() - started
    print(f"\n  Total runtime: {elapsed / 60:.1f} minutes")
    print(f"  Token usage  : {meter.report()}")

    if push_to_hf:
        banner("+", "PUSH TO HUGGINGFACE")
        push_to_hub(hf_repo_id)

    # Mark the run finished. Without this the progress file stays at
    # "running", and once its heartbeat goes stale the dashboard reports a
    # successful run as "presumed dead" - which is exactly what happened
    # before this line existed.
    progress.finish(
        f"{stats['counts']['final_pairs']} pairs, "
        f"average quality {stats['quality']['avg_quality_score']}/6",
        extra={
            "final_pairs": stats["counts"]["final_pairs"],
            "avg_quality": stats["quality"]["avg_quality_score"],
            "runtime_minutes": round(elapsed / 60, 1),
        },
    )

    print(f"\nOutputs in {OUTPUT_DIR}")
    print("View them with:  streamlit run dashboard/app.py")
    return stats


def main() -> None:
    use_utf8_console()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions-per-chunk", type=int, default=QUESTIONS_PER_CHUNK)
    p.add_argument("--min-score", type=int, default=MIN_QUALITY_SCORE, choices=range(0, 7))
    p.add_argument("--similarity", type=float, default=SIMILARITY_THRESHOLD)
    p.add_argument(
        "--fresh",
        action="store_true",
        help="ignore any checkpoint and re-generate and re-judge everything "
        "from scratch (costs the full token budget again)",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=MAX_CHUNKS,
        help="0 processes the whole corpus (hours of rate-limited calls)",
    )
    p.add_argument("--delay", type=float, default=REQUEST_DELAY)
    p.add_argument(
        "--token-budget",
        type=int,
        default=0,
        help="tokens this run may spend (default: the full 200,000/day)",
    )
    p.add_argument("--push-to-hf", action="store_true", default=PUSH_TO_HF)
    p.add_argument("--hf-repo-id", default=HF_REPO_ID)
    args = p.parse_args()

    try:
        run(
            questions_per_chunk=args.questions_per_chunk,
            min_quality_score=args.min_score,
            similarity_threshold=args.similarity,
            max_chunks=args.max_chunks,
            push_to_hf=args.push_to_hf,
            hf_repo_id=args.hf_repo_id,
            delay=args.delay,
            token_budget=args.token_budget,
            resume=not args.fresh,
        )
    except Exception as exc:
        import progress

        progress.fail(f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
