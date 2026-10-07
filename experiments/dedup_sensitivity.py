"""
Experiment 2: How does the similarity threshold affect what survives dedup?

Runs deduplication at cosine 0.75, 0.85, 0.90 and 0.95 over the pairs that passed
the quality filter, and reports survivors plus example removals at each level.

The point is to show the failure modes at both ends: too low removes
genuinely different questions, too high keeps obvious rewordings.

No API calls - embeddings are computed locally on CPU.

    python experiments/dedup_sensitivity.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deduplicate import SIMILARITY_THRESHOLD  # noqa: E402
from validate import keep_decision  # noqa: E402
from config import GRADED_FILE, OUTPUT_DIR, use_utf8_console  # noqa: E402
from deduplicate import deduplicate_pairs  # noqa: E402

SIMILARITIES = (0.75, 0.85, 0.90, 0.95)
MIN_SCORE = 4


def load_passing() -> list[dict]:
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
    return [
        r
        for r in rows
        if r.get("quality_score") is not None
        and keep_decision(r.get("scores") or {}, MIN_SCORE)
    ]


def main() -> None:
    use_utf8_console()
    pairs = load_passing()

    print("=" * 78)
    print("EXPERIMENT 2: DEDUPLICATION THRESHOLD SENSITIVITY")
    print("=" * 78)
    print(f"Input: {len(pairs)} pairs that passed the quality filter (>= {MIN_SCORE}/6)\n")

    results = []
    per_threshold_examples = {}

    for sim in SIMILARITIES:
        with contextlib.redirect_stdout(io.StringIO()):
            unique, dupes = deduplicate_pairs(pairs, threshold=sim, write_duplicates=False)
        results.append(
            {
                "similarity_threshold": sim,
                "survivors": len(unique),
                "removed": len(dupes),
                "removed_pct": round(len(dupes) / max(len(pairs), 1) * 100, 1),
            }
        )
        # The closest call at this threshold is the most informative example -
        # it sits right on the boundary.
        per_threshold_examples[sim] = sorted(
            dupes, key=lambda d: d["similarity"]
        )[:3]

    header = f"{'cosine':>8} {'survivors':>10} {'removed':>9} {'removed %':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['similarity_threshold']:>8} {r['survivors']:>10} "
            f"{r['removed']:>9} {r['removed_pct']:>9}%"
        )

    for sim in SIMILARITIES:
        examples = per_threshold_examples[sim]
        if not examples:
            continue
        print(f"\n--- Borderline removals at {sim} (the least similar ones cut) ---")
        for d in examples:
            print(f"  sim {d['similarity']}")
            print(f"    removed: {d['removed_question'][:88]}")
            print(f"    kept   : {d['duplicate_of_question'][:88]}")

    out = OUTPUT_DIR / "experiment_dedup_sensitivity.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")

    print("\nReading the results:")
    print("  0.75 is aggressive - inspect the borderline removals above for")
    print("       questions that are actually distinct.")
    print("  0.95 is permissive - near-identical rewordings survive.")
    print(
        f"  {SIMILARITY_THRESHOLD} is the project default: the point where "
        f"removals still read as duplicates."
    )


if __name__ == "__main__":
    main()
