"""
Experiment 1: How does the quality threshold trade dataset size against quality?

Re-filters the already-graded pairs at min_quality_score 3, 4 and 5, then runs
deduplication on each, and reports the surviving count and average score.

No API calls - it reads data/output/graded_pairs.jsonl, so the sweep is free
and reproducible. Run the pipeline first.

    python experiments/threshold_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from validate import MIN_CRITERION_SCORES, keep_decision  # noqa: E402
from config import GRADED_FILE, OUTPUT_DIR, use_utf8_console  # noqa: E402
from deduplicate import deduplicate_pairs, question_diversity  # noqa: E402

# The brief asks for 3/4/5. On a corpus where the judge is generous those
# three land within a pair or two of each other and the table says nothing -
# measured here: 183, 183, 182 survivors. 6 is included because it is the only
# threshold that actually bites when the score distribution is skewed high,
# and a sensitivity experiment that shows no sensitivity is not an experiment.
THRESHOLDS = (3, 4, 5, 6)
# The project's actual default, imported rather than restated - holding the
# sweep at a value the pipeline no longer uses would measure the wrong thing.
from deduplicate import SIMILARITY_THRESHOLD as SIMILARITY  # noqa: E402


def load_graded() -> list[dict]:
    if not GRADED_FILE.exists():
        raise SystemExit(
            f"{GRADED_FILE} not found.\n"
            f"  Run the pipeline first:  python main.py --max-chunks 100"
        )
    rows = [
        json.loads(line)
        for line in GRADED_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Judge failures were never scored, so they cannot take part in a
    # threshold sweep. Excluding them keeps every row comparable.
    return [r for r in rows if r.get("quality_score") is not None and r.get("scores")]


def main() -> None:
    use_utf8_console()
    graded = load_graded()

    print("=" * 78)
    print("EXPERIMENT 1: QUALITY THRESHOLD SENSITIVITY")
    print("=" * 78)
    print(f"Graded pairs available: {len(graded)}")
    print(f"Dedup similarity held constant at {SIMILARITY}\n")

    results = []
    for threshold in THRESHOLDS:
        for veto in (True, False):
            kept = [
                p
                for p in graded
                if keep_decision(
                    p["scores"], threshold, MIN_CRITERION_SCORES if veto else {}
                )
            ]
            if not kept:
                continue

            # Silence dedup's own progress output - only the totals matter here.
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                unique, dupes = deduplicate_pairs(
                    kept, threshold=SIMILARITY, write_duplicates=False
                )

            scores = [p["quality_score"] for p in unique]
            results.append(
                {
                    "min_score": threshold,
                    "groundedness_veto": veto,
                    "passed_filter": len(kept),
                    "after_dedup": len(unique),
                    "duplicates": len(dupes),
                    "avg_quality": round(sum(scores) / len(scores), 3),
                    "pct_of_graded": round(len(unique) / len(graded) * 100, 1),
                    "unique_openers": question_diversity(unique)["unique_openers"],
                }
            )

    header = (
        f"{'min':>4} {'veto':>6} {'passed':>8} {'dedup':>7} {'final':>7} "
        f"{'avg qual':>9} {'% graded':>9} {'openers':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['min_score']:>4} {str(r['groundedness_veto']):>6} "
            f"{r['passed_filter']:>8} {r['duplicates']:>7} {r['after_dedup']:>7} "
            f"{r['avg_quality']:>9} {r['pct_of_graded']:>8}% {r['unique_openers']:>8}"
        )

    out = OUTPUT_DIR / "experiment_threshold_sensitivity.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")

    print("\nReading the table:")
    print(
        "  'veto' True applies the per-criterion floors "
        + str(MIN_CRITERION_SCORES)
        + " as well as the total."
    )
    print("  Raising min_score shrinks the dataset and raises average quality.")
    print("  The right threshold is the one where the loss stops buying quality.")


if __name__ == "__main__":
    main()
