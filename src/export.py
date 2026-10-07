"""
Step 5: Export the clean dataset, the ChatML variant, and pipeline statistics.

Internal working fields (`chunk_text`, the raw `scores` dict) are stripped
here. They exist to let the judge work and the dashboard explain itself; they
are not part of the training data.

pipeline_stats.json is written for the dashboard, which will not start without
it. Every number the dashboard shows comes from this file.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

try:
    from config import (
        ABSTENTION_ANSWER,
        CHATML_FILE,
        DATASET_FILE,
        GROQ_MODEL,
        JUDGE_MODEL,
        PROJECT_ROOT,
        GEN_MAX_TOKENS,
        JUDGE_MAX_TOKENS,
        REASONING_EFFORT,
        STATS_FILE,
        UNANSWERABLE_FILE,
        UNANSWERABLE_RATIO,
        ensure_output_dir,
        use_utf8_console,
    )
except ImportError:  # running as `python src/export.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (
        ABSTENTION_ANSWER,
        CHATML_FILE,
        DATASET_FILE,
        GROQ_MODEL,
        JUDGE_MODEL,
        PROJECT_ROOT,
        GEN_MAX_TOKENS,
        JUDGE_MAX_TOKENS,
        REASONING_EFFORT,
        STATS_FILE,
        UNANSWERABLE_FILE,
        UNANSWERABLE_RATIO,
        ensure_output_dir,
        use_utf8_console,
    )

# Required on the README, the dashboard and any HuggingFace dataset card.
# Defined once here so all three carry exactly the same wording.
SAFETY_DISCLAIMER = (
    "EDUCATIONAL AND RESEARCH USE ONLY. This dataset was generated "
    "automatically from the source documents supplied to the pipeline by a "
    "large language model, and validated by another language model. It is NOT "
    "professional advice of any kind and MUST NOT be used for decision-making. "
    "Answers may be incomplete, outdated or wrong. Always consult the original "
    "document and a qualified professional. Where the sources are clinical, "
    "this is explicitly NOT medical advice and NOT for patient care; no "
    "patient data was used at any stage."
)

# Domain-specific extra wording, appended when the sources look clinical.
# Uploads made the pipeline domain-agnostic - the same code now runs over
# banking circulars as readily as WHO guidelines - so the disclaimer adapts
# rather than asserting a medical framing that may not apply.
_CLINICAL_HINT = (
    "who", "clinical", "health", "patient", "hiv", "medical", "disease",
    "guideline", "treatment", "care",
)


def looks_clinical(sources) -> bool:
    """Whether the source filenames suggest clinical content."""
    joined = " ".join(str(s).lower() for s in sources)
    return any(h in joined for h in _CLINICAL_HINT)

# The fields that make it into the training data. Everything else is internal.
EXPORT_FIELDS = (
    "question",
    "answer",
    "source",
    "page",
    "question_type",
    # Whether the passage can answer this at all. Anyone fine-tuning on the
    # file needs it: the abstention rows are the ones teaching refusal, and a
    # trainer that cannot tell them apart cannot weight or hold them out.
    "answerable",
    # None on abstention rows - they were never graded 0-6. See
    # validate.UNANSWERABLE_JUDGE_PROMPT for why.
    "quality_score",
)


def _wrap(text: str, width: int) -> list[str]:
    """Wrap a message for the terminal report."""
    import textwrap

    return textwrap.wrap(text, width)


def to_clean_records(pairs: list[dict]) -> list[dict]:
    """Strip internal fields, keeping only what belongs in the dataset."""
    return [{k: p.get(k) for k in EXPORT_FIELDS} for p in pairs]


def to_chatml(pairs: list[dict], include_system: bool = True) -> list[dict]:
    """
    Convert to ChatML `messages` format for SFTTrainer.

    A system message carrying the disclaimer is included by default: if this
    data is ever used to fine-tune a model, the safety framing should be part
    of the training signal rather than a note in a README nobody reads.
    """
    sources = {p.get("source", "") for p in pairs}
    if looks_clinical(sources):
        system = (
            "You are a clinical guidelines assistant answering from guideline "
            "documents. For education and research only - not medical advice."
        )
    else:
        system = (
            "You answer questions strictly from the supplied source documents. "
            "For education and research only - not professional advice."
        )
    records = []
    for p in pairs:
        messages = []
        if include_system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": p["question"]})
        messages.append({"role": "assistant", "content": p["answer"]})
        records.append(
            {
                "messages": messages,
                "source": p.get("source"),
                "page": p.get("page"),
                "question_type": p.get("question_type"),
                # Carried here too, not only in the plain dataset: a trainer
                # reading the ChatML file needs to be able to hold the
                # abstention rows out, weight them, or count them.
                "answerable": p.get("answerable", True),
                "quality_score": p.get("quality_score"),
            }
        )
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    """One JSON object per line, UTF-8, no ASCII escaping."""
    ensure_output_dir()
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_stats(
    final_pairs: list[dict],
    raw_pairs: list[dict],
    validated_pairs: list[dict],
    rejected_pairs: list[dict],
    duplicates: list[dict],
    chunks_processed: int,
    min_quality_score: int,
    similarity_threshold: float,
    meter=None,
    wall_seconds: float | None = None,
) -> dict:
    """Assemble every number the dashboard and the report need."""
    scores = [p["quality_score"] for p in final_pairs if p.get("quality_score") is not None]

    # Abstention pairs carry no score, so every quality figure here is over
    # answerable pairs only. Splitting them out keeps the averages meaningful
    # and lets the dashboard report the two kinds separately.
    def answerable_only(rows):
        return [p for p in rows if p.get("answerable", True)]

    def abstention_only(rows):
        return [p for p in rows if not p.get("answerable", True)]

    raw_answerable = answerable_only(raw_pairs)
    raw_abstention = abstention_only(raw_pairs)
    kept_answerable = answerable_only(validated_pairs)
    kept_abstention = abstention_only(validated_pairs)
    final_abstention = abstention_only(final_pairs)

    # A judge failure is "not assessed" - counting it as a quality rejection
    # would understate the pass rate, so it is separated out.
    judge_failures = sum(
        1 for r in rejected_pairs if r.get("reject_reason") == "judge call failed - not assessed"
    )
    ungrounded = sum(
        1 for r in rejected_pairs if str(r.get("reject_reason", "")).startswith("UNGROUNDED")
    )
    assessed = len(raw_pairs) - judge_failures

    # The rubric pass rate, over the pairs the rubric was actually applied to.
    # This is the number the report should quote: the headline pass_rate_pct
    # below mixes in abstention pairs, which pass a different test entirely.
    answerable_judge_failures = sum(
        1
        for r in rejected_pairs
        if r.get("answerable", True)
        and r.get("reject_reason") == "judge call failed - not assessed"
    )
    answerable_assessed = len(raw_answerable) - answerable_judge_failures

    all_graded = [
        p for p in validated_pairs + rejected_pairs if p.get("quality_score") is not None
    ]

    from deduplicate import (
        EMBED_QUESTION_AND_ANSWER,
        REQUIRE_SAME_NUMBERS,
        question_diversity,
    )
    from validate import MIN_CRITERION_SCORES

    # A run that ran out of quota part-way still produces a valid dataset, but
    # its headline rates describe only the pairs that were actually graded.
    # Saying so here keeps the dashboard and the report honest: a 100% pass
    # rate over 122 of 263 pairs is not the same claim as a 100% pass rate.
    unassessed_pct = round(judge_failures / max(len(raw_pairs), 1) * 100, 1)
    if judge_failures == 0:
        health = {
            "complete": True,
            "note": "Every generated pair was graded by the judge.",
        }
    else:
        health = {
            "complete": False,
            "unassessed_pairs": judge_failures,
            "unassessed_pct": unassessed_pct,
            "note": (
                f"{judge_failures} of {len(raw_pairs)} pairs ({unassessed_pct}%) were "
                f"never graded - the judge could not score them, usually because the "
                f"API returned an unparseable response or the daily token budget ran "
                f"out. The pass rate and average quality below describe only the "
                f"{assessed} pairs that were assessed. Check the run log to see which "
                f"applies; if it was the budget, re-run with a smaller --max-chunks."
            ),
        }

    return {
        "run_health": health,
        "model": GROQ_MODEL,
        "judge_model": JUDGE_MODEL,
        "embedding_model": "all-MiniLM-L6-v2",
        "parameters": {
            "min_quality_score": min_quality_score,
            "similarity_threshold": similarity_threshold,
            "max_total_score": 6,
            # Recorded because the keep rule is no longer the total alone, and
            # a stats file that does not say what the floors were cannot be
            # compared against another run.
            "criterion_floors": dict(MIN_CRITERION_SCORES),
            "dedup_embeds": (
                "question + answer" if EMBED_QUESTION_AND_ANSWER else "question"
            ),
            "dedup_number_guard": REQUIRE_SAME_NUMBERS,
            "generation_max_tokens": GEN_MAX_TOKENS,
            "judge_max_tokens": JUDGE_MAX_TOKENS,
            "reasoning_effort": REASONING_EFFORT,
        },
        "counts": {
            "chunks_processed": chunks_processed,
            "raw_pairs": len(raw_pairs),
            "raw_answerable": len(raw_answerable),
            "raw_unanswerable": len(raw_abstention),
            "validated_pairs": len(validated_pairs),
            "rejected_pairs": len(rejected_pairs),
            "duplicates_removed": len(duplicates),
            "final_pairs": len(final_pairs),
            "final_unanswerable": len(final_abstention),
        },
        "rates": {
            "pairs_per_chunk": round(len(raw_pairs) / max(chunks_processed, 1), 2),
            "pass_rate_pct": round(len(validated_pairs) / max(assessed, 1) * 100, 1),
            "answerable_pass_rate_pct": round(
                len(kept_answerable) / max(answerable_assessed, 1) * 100, 1
            ),
            "duplicate_rate_pct": round(
                len(duplicates) / max(len(validated_pairs), 1) * 100, 1
            ),
            "overall_yield_pct": round(len(final_pairs) / max(len(raw_pairs), 1) * 100, 1),
            "judge_failures": judge_failures,
            "ungrounded_vetoed": ungrounded,
        },
        "quality": {
            "avg_quality_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "min_quality_score_observed": min(scores) if scores else None,
            "max_quality_score_observed": max(scores) if scores else None,
            "score_distribution": {
                str(s): c for s, c in sorted(Counter(scores).items())
            },
            "score_distribution_all_graded": {
                str(s): c
                for s, c in sorted(Counter(p["quality_score"] for p in all_graded).items())
            },
            "avg_subscores": {
                crit: round(
                    sum(p["scores"][crit] for p in final_pairs if p.get("scores"))
                    / max(sum(1 for p in final_pairs if p.get("scores")), 1),
                    2,
                )
                for crit in ("groundedness", "specificity", "completeness")
            },
        },
        # Abstention examples: generated, verified, and how many survived
        # deduplication into the dataset.
        "abstention": {
            "ratio_requested": UNANSWERABLE_RATIO,
            "generated": len(raw_abstention),
            "verified": len(kept_abstention),
            "in_dataset": len(final_abstention),
            "answer": ABSTENTION_ANSWER,
            "failed_check": sum(
                1
                for r in rejected_pairs
                if not r.get("answerable", True)
                and str(r.get("reject_reason", "")).startswith("abstention check failed")
            ),
        },
        # What the run actually cost. Worth recording because the binding
        # constraint on this project is the daily token allowance, not compute,
        # and an estimate is no substitute for the measured figure.
        "run_cost": {
            "api_calls": getattr(meter, "requests", None),
            "tokens_used": getattr(meter, "used", None),
            "tokens_by_stage": dict(getattr(meter, "per_stage", {}) or {}),
            "token_budget": getattr(meter, "budget", None),
            "wall_seconds": round(wall_seconds, 1) if wall_seconds else None,
            "wall_human": (
                f"{int(wall_seconds // 60)}m {int(wall_seconds % 60)}s"
                if wall_seconds
                else None
            ),
        },
        "per_source": dict(Counter(p.get("source", "?") for p in final_pairs)),
        "question_types": dict(Counter(p.get("question_type", "?") for p in final_pairs)),
        "diversity": question_diversity(final_pairs),
        "top_reject_reasons": [
            {"reason": r, "count": c}
            for r, c in Counter(
                str(p.get("reject_reason", "unknown"))[:80] for p in rejected_pairs
            ).most_common(10)
        ],
        "disclaimer": SAFETY_DISCLAIMER,
    }


def export_all(
    final_pairs: list[dict],
    raw_pairs: list[dict],
    validated_pairs: list[dict],
    rejected_pairs: list[dict],
    duplicates: list[dict],
    chunks_processed: int,
    min_quality_score: int,
    similarity_threshold: float,
    meter=None,
    wall_seconds: float | None = None,
) -> dict:
    """Write dataset + ChatML + stats, print the final report, return stats."""
    ensure_output_dir()

    clean = to_clean_records(final_pairs)
    write_jsonl(clean, DATASET_FILE)
    write_jsonl(to_chatml(final_pairs), CHATML_FILE)

    # Independent, non-LLM grounding check. Run here because `final_pairs`
    # still carries chunk_text and the judge's sub-scores - after
    # to_clean_records strips them, the comparison is no longer possible
    # without re-reading the PDFs.
    from grounding import write_analysis

    # Answerable pairs only. The grounding check measures how much of an answer
    # is traceable to its passage, and an abstention answer is traceable to
    # nothing by design - including them would manufacture a hallucination
    # problem that is not there.
    grounding_summary = write_analysis([p for p in final_pairs if p.get("answerable", True)])

    stats = build_stats(
        final_pairs,
        raw_pairs,
        validated_pairs,
        rejected_pairs,
        duplicates,
        chunks_processed,
        min_quality_score,
        similarity_threshold,
        meter=meter,
        wall_seconds=wall_seconds,
    )
    stats["grounding"] = grounding_summary

    # Every abstention pair with the judge's verdict on it, kept or not, so the
    # dashboard can show the ones that failed next to the ones that passed.
    abstention_rows = [
        {
            "question": p["question"],
            "answer": p["answer"],
            "source": p.get("source"),
            "page": p.get("page"),
            "in_dataset": any(
                f["question"] == p["question"] for f in final_pairs
            ),
            "check": p.get("unanswerable_check"),
            "reject_reason": p.get("reject_reason"),
        }
        for p in validated_pairs + rejected_pairs
        if not p.get("answerable", True)
    ]
    write_jsonl(abstention_rows, UNANSWERABLE_FILE)

    with open(STATS_FILE, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    c, r, q = stats["counts"], stats["rates"], stats["quality"]

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  Chunks processed      : {c['chunks_processed']}")
    print(f"  Raw pairs generated   : {c['raw_pairs']}")
    print(
        f"  After quality filter  : {c['validated_pairs']}"
        f"   ({r['pass_rate_pct']}% pass rate, threshold >= {min_quality_score}/6)"
    )
    if r["ungrounded_vetoed"]:
        print(f"      ungrounded vetoed : {r['ungrounded_vetoed']}")
    if r["judge_failures"]:
        print(f"      judge failures    : {r['judge_failures']} (not assessed)")
    print(
        f"  After deduplication   : {c['final_pairs']}"
        f"   ({c['duplicates_removed']} removed, {r['duplicate_rate_pct']}%,"
        f" threshold {similarity_threshold})"
    )
    print(f"  Overall yield         : {r['overall_yield_pct']}% of raw pairs kept")
    print(f"  Average quality score : {q['avg_quality_score']}/6"
          f"   (answerable pairs only)")
    ab = stats.get("abstention", {})
    if ab.get("generated"):
        print(
            f"  Abstention examples   : {ab['in_dataset']} in the dataset"
            f"   ({ab['generated']} written, {ab['verified']} verified,"
            f" {ab['failed_check']} failed the check)"
        )
        note = (
            f" (the headline {r['pass_rate_pct']}% mixes in the abstention check)"
            if r["answerable_pass_rate_pct"] != r["pass_rate_pct"]
            else ""
        )
        print(
            f"  Rubric pass rate      : {r['answerable_pass_rate_pct']}%"
            f" of answerable pairs{note}"
        )
    if not stats["run_health"]["complete"]:
        print()
        print("  INCOMPLETE RUN")
        for line in _wrap(stats["run_health"]["note"], 66):
            print(f"    {line}")
    print(f"  Sub-scores (avg)      : {q['avg_subscores']}")
    print(f"  Question types        : {stats['question_types']}")
    g = stats.get("grounding", {})
    if g.get("analysed"):
        print(
            f"  Grounding (non-LLM)   : {g['band_pct']['strong']}% strong overlap, "
            f"{g['unsupported_number_pairs']} pairs with unsupported numbers"
        )
    print(f"  Distinct openers      : {stats['diversity']['unique_openers']}")
    print()
    print(f"  {DATASET_FILE.name:<32} {c['final_pairs']} rows")
    print(f"  {CHATML_FILE.name:<32} {c['final_pairs']} rows")
    if stats.get("abstention", {}).get("generated"):
        print(f"  {UNANSWERABLE_FILE.name:<32} {stats['abstention']['generated']} rows")
    print(f"  {STATS_FILE.name:<32} written")
    rc = stats.get("run_cost", {})
    if rc.get("api_calls"):
        print(
            f"\n  Run cost: {rc['api_calls']} API calls, "
            f"{rc['tokens_used']:,} tokens, {rc['wall_human']} wall time"
        )
    print(f"\n  {SAFETY_DISCLAIMER[:66]}...")
    return stats


def dataset_card(stats: dict) -> str:
    """Markdown card for the HuggingFace repo, carrying the disclaimer."""
    c = stats["counts"]
    return f"""---
license: cc-by-nc-sa-4.0
task_categories:
  - question-answering
language:
  - en
tags:
  - synthetic
  - medical
  - who-guidelines
---

# Synthetic WHO Clinical Guideline Q&A

## Disclaimer

**{SAFETY_DISCLAIMER}**

## Summary

{c['final_pairs']} question-answer pairs generated from public WHO clinical
guideline PDFs, scored by an LLM judge and semantically deduplicated.

| Stage | Count |
|---|---|
| Raw pairs generated | {c['raw_pairs']} |
| Passed quality filter (>= {stats['parameters']['min_quality_score']}/6) | {c['validated_pairs']} |
| After deduplication (cosine < {stats['parameters']['similarity_threshold']}) | {c['final_pairs']} |

Average quality score: **{stats['quality']['avg_quality_score']}/6**
Pass rate: **{stats['rates']['pass_rate_pct']}%**

Generator and judge: `{stats['model']}` (Groq).
Embeddings: `{stats['embedding_model']}`.

## Fields

`question`, `answer`, `source`, `page`, `question_type`, `quality_score`

## Provenance

Built only from publicly available WHO guideline documents. No patient data
was used at any stage.
"""


def push_to_hub(repo_id: str, private: bool = True) -> None:
    """
    Upload the dataset to HuggingFace. Never called automatically.

    Requires HF_TOKEN in .env. Kept explicit because pushing generated medical
    text to a public hub is a decision a person should make deliberately.
    """
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN missing. Add it to .env to push:\n"
            "  HF_TOKEN=hf_...\n"
            "  Get one at https://huggingface.co/settings/tokens"
        )
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"{DATASET_FILE} not found. Run the pipeline first.")

    from datasets import load_dataset
    from huggingface_hub import HfApi

    print(f"Pushing {DATASET_FILE.name} to {repo_id} (private={private})...")
    ds = load_dataset("json", data_files=str(DATASET_FILE), split="train")
    ds.push_to_hub(repo_id, token=token, private=private)

    if STATS_FILE.exists():
        stats = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        card_path = PROJECT_ROOT / "data" / "output" / "README.md"
        card_path.write_text(dataset_card(stats), encoding="utf-8")
        HfApi().upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print("  Dataset card uploaded (includes the safety disclaimer).")

    print(f"  Done: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    use_utf8_console()
    print("=" * 70)
    print("STEP 5: EXPORT - self-test with synthetic records")
    print("=" * 70)

    demo = [
        {
            "question": "What CD4 count defines advanced HIV disease in adults?",
            "answer": "A CD4 cell count below 200 cells/mm3 defines advanced HIV disease in adults.",
            "source": "demo.pdf",
            "page": 12,
            "question_type": "definitional",
            "quality_score": 6,
            "scores": {"groundedness": 2, "specificity": 2, "completeness": 2},
            "chunk_text": "INTERNAL - must not appear in the export",
        }
    ]
    rejected = [{"quality_score": 2, "reject_reason": "answer too vague", "scores": {}}]

    stats = export_all(
        final_pairs=demo,
        raw_pairs=demo * 3,
        validated_pairs=demo,
        rejected_pairs=rejected,
        duplicates=[],
        chunks_processed=1,
        min_quality_score=4,
        similarity_threshold=0.85,
    )

    print("\n--- Exported row (chunk_text and scores must be gone) ---")
    first = DATASET_FILE.read_text(encoding="utf-8").splitlines()[0]
    print(first)
    assert "chunk_text" not in first, "LEAK: chunk_text reached the export"
    assert "groundedness" not in first, "LEAK: raw scores reached the export"
    print("\nOK - internal fields stripped.")
