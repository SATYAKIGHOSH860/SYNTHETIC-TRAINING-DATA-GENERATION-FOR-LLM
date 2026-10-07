"""
Experiment 3: Is the LLM judge any good? Cohen's kappa against a human.

Samples 20% of the graded pairs, shows you the passage and the pair, and asks
you to score the same 0-2 rubric the model used. Then it computes agreement.

    python experiments/judge_reliability.py           # score them by hand
    python experiments/judge_reliability.py --report  # results so far

Your scores are saved after every pair to
data/output/human_scores.json, so you can stop and resume.

Report the kappa honestly. If agreement is weak (< 0.4), the judge's numbers
should be downweighted in the conclusions and the report should say so - that
paragraph is worth more than another feature.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from validate import keep_decision  # noqa: E402
from config import GRADED_FILE, OUTPUT_DIR, use_utf8_console  # noqa: E402

HUMAN_FILE = OUTPUT_DIR / "human_scores.json"
# A fixed count, not a fraction. 20% of a 60-pair run is 12 pairs, and a
# kappa over 12 items has confidence bounds so wide it cannot support any
# claim about the judge. 40 is the smallest sample that gives the statistic
# something to stand on; smaller corpora simply use everything they have.
SAMPLE_SIZE = 40
SEED = 42  # fixed so the sample is the same every time you resume
CRITERIA = ("groundedness", "specificity", "completeness")

RUBRIC = """
  groundedness  2 = every claim is in the passage
                1 = mostly supported, adds a small unstated detail
                0 = contradicts the passage or invents information

  specificity   2 = precise question about a named topic/threshold/procedure
                1 = general but still answerable on its own
                0 = vague, or refers to "the passage"

  completeness  2 = fully answers the question, stands alone
                1 = partial, or needs the question for context
                0 = does not answer, or is truncated
"""


def load_sample() -> list[dict]:
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
    rows = [r for r in rows if r.get("scores")]
    n = min(len(rows), SAMPLE_SIZE)
    rng = random.Random(SEED)
    # Sort first so the sample does not depend on file ordering.
    rows.sort(key=lambda r: (r.get("question", ""), r.get("page", 0)))
    return rng.sample(rows, n)


def load_human() -> dict:
    if HUMAN_FILE.exists():
        return json.loads(HUMAN_FILE.read_text(encoding="utf-8"))
    return {}


def save_human(data: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HUMAN_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def ask_score(criterion: str) -> int | None:
    """Prompt until a valid 0/1/2 arrives. 's' skips, 'q' quits."""
    while True:
        raw = input(f"    {criterion:<14} 0/1/2 (s=skip, q=quit): ").strip().lower()
        if raw == "q":
            raise KeyboardInterrupt
        if raw == "s":
            return None
        if raw in {"0", "1", "2"}:
            return int(raw)
        print("      enter 0, 1, 2, s or q")


def score_interactively() -> None:
    sample = load_sample()
    human = load_human()

    todo = [p for p in sample if p["question"] not in human]
    print("=" * 78)
    print("EXPERIMENT 3: JUDGE RELIABILITY - manual scoring")
    print("=" * 78)
    print(
        f"Sample: {len(sample)} pairs (target {SAMPLE_SIZE}, seed {SEED})"
    )
    print(f"Already scored: {len(sample) - len(todo)}   Remaining: {len(todo)}")
    print("\nScore each pair WITHOUT looking at the model's grade.")
    print(RUBRIC)
    print("Progress is saved after every pair. Ctrl-C or 'q' to stop.\n")

    try:
        for i, pair in enumerate(todo, 1):
            print("=" * 78)
            print(f"[{i}/{len(todo)}]  {pair.get('source')} p.{pair.get('page')}")
            print("-" * 78)
            print("PASSAGE:")
            print("  " + pair.get("chunk_text", "")[:800].replace("\n", "\n  "))
            print("-" * 78)
            print(f"QUESTION: {pair['question']}")
            print(f"ANSWER  : {pair['answer']}")
            print("-" * 78)

            scores = {}
            for criterion in CRITERIA:
                value = ask_score(criterion)
                if value is None:
                    scores = {}
                    break
                scores[criterion] = value

            if scores:
                human[pair["question"]] = scores
                save_human(human)
                print(f"    saved (total {sum(scores.values())}/6)\n")
            else:
                print("    skipped\n")
    except (KeyboardInterrupt, EOFError):
        print("\n\nStopped. Progress saved.")

    print(f"Scored {len(human)} of {len(sample)}.")
    print("Run with --report to see agreement.")


def report() -> None:
    from sklearn.metrics import cohen_kappa_score

    sample = load_sample()
    human = load_human()
    paired = [p for p in sample if p["question"] in human]

    print("=" * 78)
    print("EXPERIMENT 3: JUDGE RELIABILITY - agreement report")
    print("=" * 78)
    if len(paired) < 10:
        print(f"Only {len(paired)} pairs scored by hand. Kappa needs more to mean")
        print("anything - score at least 10, ideally the whole 20% sample.")
        print("  python experiments/judge_reliability.py")
        if not paired:
            return

    print(f"Pairs scored by both: {len(paired)}\n")

    # A human rater who gave the same score to every pair has not produced a
    # measurement, and no statistic computed from it means anything. Catch it
    # here rather than let it become a number in a report.
    flat = [
        criterion
        for criterion in CRITERIA
        if len({human[p["question"]][criterion] for p in paired}) < 2
    ]
    if flat:
        print("  WARNING: the hand labels have no variance on " + ", ".join(flat))
        print("  Every pair was given the same score, so agreement cannot be")
        print("  measured. Re-score the sample, changing the scores to reflect")
        print("  each pair, before quoting any kappa.")
        print()

    results = {"n": len(paired), "criteria": {}}
    header = f"{'criterion':<15} {'kappa':>8} {'weighted':>9} {'exact agr':>11} {'strength':>14}"
    print(header)
    print("-" * len(header))

    for criterion in CRITERIA:
        llm = [p["scores"][criterion] for p in paired]
        hum = [human[p["question"]][criterion] for p in paired]

        # Kappa is undefined when EITHER rater used a single value: it
        # measures agreement beyond chance, and a constant rater has no
        # chance-corrected scale to measure against. The condition used to be
        # `and`, which let the single-rater case through to sklearn - that
        # returns a confident-looking 0.000 for what is really "not
        # measurable", and a 0.000 in a report reads as "the judge is no
        # better than chance" rather than "these labels cannot answer that".
        if len(set(llm)) < 2 or len(set(hum)) < 2:
            agreement = sum(a == b for a, b in zip(llm, hum)) / len(llm)
            print(
                f"{criterion:<15} {'n/a':>8} {'n/a':>9} {agreement:>10.1%} "
                f"{'no variance':>14}"
            )
            which = (
                "human" if len(set(hum)) < 2 else "judge"
            ) if (len(set(hum)) < 2) != (len(set(llm)) < 2) else "both raters"
            results["criteria"][criterion] = {
                "kappa": None,
                "weighted_kappa": None,
                "exact_agreement": round(agreement, 3),
                "note": f"{which} used a single value - kappa undefined",
            }
            continue

        kappa = cohen_kappa_score(hum, llm, labels=[0, 1, 2])
        # Quadratic weights: 2-vs-0 is a worse disagreement than 2-vs-1, which
        # matters for an ordinal scale.
        weighted = cohen_kappa_score(hum, llm, labels=[0, 1, 2], weights="quadratic")
        agreement = sum(a == b for a, b in zip(llm, hum)) / len(llm)

        print(
            f"{criterion:<15} {kappa:>8.3f} {weighted:>9.3f} {agreement:>10.1%} "
            f"{strength(kappa):>14}"
        )
        results["criteria"][criterion] = {
            "kappa": round(float(kappa), 3),
            "weighted_kappa": round(float(weighted), 3),
            "exact_agreement": round(agreement, 3),
            "strength": strength(kappa),
        }

    # The decision that actually matters: keep or reject.
    # Exactly the rule the pipeline applies, imported rather than restated -
    # a sensitivity study that measures a different rule measures nothing.
    llm_keep = [1 if keep_decision(p["scores"]) else 0 for p in paired]
    hum_keep = [1 if keep_decision(human[p["question"]]) else 0 for p in paired]
    print()
    if len(set(llm_keep)) < 2 or len(set(hum_keep)) < 2:
        agr = sum(a == b for a, b in zip(llm_keep, hum_keep)) / len(llm_keep)
        print(f"Keep/reject decision: {agr:.1%} agreement (no variance - kappa undefined)")
        results["keep_reject"] = {"kappa": None, "agreement": round(agr, 3)}
    else:
        k = cohen_kappa_score(hum_keep, llm_keep, labels=[0, 1])
        agr = sum(a == b for a, b in zip(llm_keep, hum_keep)) / len(llm_keep)
        print(
            f"Keep/reject decision: kappa {k:.3f} ({strength(k)}), "
            f"{agr:.1%} agreement"
        )
        results["keep_reject"] = {
            "kappa": round(float(k), 3),
            "agreement": round(agr, 3),
            "strength": strength(k),
        }

    out = OUTPUT_DIR / "experiment_judge_reliability.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")

    print("\nInterpreting kappa (Landis & Koch):")
    print("  < 0.00 none | 0.00-0.20 slight | 0.21-0.40 fair")
    print("  0.41-0.60 moderate | 0.61-0.80 substantial | 0.81-1.00 almost perfect")
    print("\nIf agreement is fair or worse, say so in the report and downweight")
    print("the judge's numbers. An honest limitation beats an inflated metric.")


def strength(kappa: float) -> str:
    if kappa < 0:
        return "none"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


if __name__ == "__main__":
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="show agreement, do not score")
    args = ap.parse_args()
    report() if args.report else score_interactively()
