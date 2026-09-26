"""Task 11 experiments, computed offline from a finished run (no API calls).

    python -m src.experiments

Reads data/output/judged_pairs.jsonl (every judged pair with its sub-scores)
and writes data/output/experiments.json and experiments.md:

1. Quality threshold sensitivity: dataset size vs average quality at 3/4/5/6,
   under the spec's total-only rule and under total + per-criterion floors.
2. Deduplication sensitivity: removals at 0.75-0.95 for question-only
   embeddings (the spec) and question+answer embeddings with the number guard
   (used), with the removals nearest each threshold as inspectable examples.
3. Judge reliability: Cohen's kappa from human_labels.json, once labelled.
4. Question-type and opening-word diversity before and after deduplication.

The judge scored each pair once, independent of any threshold, so every
threshold here is applied to the same scores: the comparison is exact.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.config import CONFIG, Config, resolve_path
from src.deduplicate import answer_numbers, diversity_stats, embed_questions
from src.export import load_jsonl
from src.validate import apply_threshold

QUALITY_THRESHOLDS = (3, 4, 5, 6)
DEDUP_THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 3) if values else None


def _dedup(indices: list[int], pairs: list[dict[str, Any]], sim: np.ndarray, threshold: float,
           numbers: list[frozenset[str]] | None) -> tuple[list[int], list[tuple[int, int, float]]]:
    """The pipeline's greedy dedup rule on a subset, reusing one precomputed similarity matrix.

    Same visiting order as src.deduplicate (higher quality first) and the same
    optional number guard, so each setting is evaluated without re-encoding.
    """
    if not indices:
        return [], []
    order = sorted(indices, key=lambda i: (-(pairs[i].get("quality_score") or 0), indices.index(i)))
    kept: list[int] = []
    removed: list[tuple[int, int, float]] = []
    for i in order:
        if kept:
            sims = sim[i, kept].copy()
            if numbers is not None and numbers[i]:
                for pos, j in enumerate(kept):
                    if numbers[j] and numbers[j] != numbers[i]:
                        sims[pos] = -1.0
            best = int(np.argmax(sims))
            if sims[best] > threshold:
                removed.append((i, kept[best], float(sims[best])))
                continue
        kept.append(i)
    return sorted(kept), removed


def judge_reliability(out: Path, cfg: Config) -> dict[str, Any] | None:
    from src import agreement

    if not (out / "human_labels.json").is_file():
        return None
    report = agreement.agreement_report(out, cfg)
    return report if report["n"] else None


def run_experiments(output_dir: str | Path | None = None, cfg: Config = CONFIG) -> dict[str, Any]:
    out = resolve_path(output_dir or cfg.paths.output_dir)
    path = out / "judged_pairs.jsonl"
    if not path.is_file():
        raise SystemExit(f"No {path}. Generate a dataset first (dashboard: Pipeline overview, Upload PDFs; "
                         "or `python main.py`).")
    judged = load_jsonl(path)
    index = {p["id"]: i for i, p in enumerate(judged)}
    emb_q = np.asarray(embed_questions([p["question"] for p in judged], cfg), dtype=np.float32)
    emb_qa = np.asarray(embed_questions([f"{p['question']} {p['answer']}" for p in judged], cfg), dtype=np.float32)
    sims = {"question": emb_q @ emb_q.T, "question_answer": emb_qa @ emb_qa.T}
    numbers = [answer_numbers(p["answer"]) for p in judged]
    used_sim = sims[cfg.deduplicate.embed]
    used_numbers = numbers if cfg.deduplicate.require_same_numbers else None
    used_t = cfg.deduplicate.similarity_threshold
    floors = cfg.validate.min_criterion_scores.to_dict()

    # 1. Quality threshold sensitivity --------------------------------------------------
    answerable = [p for p in judged if p.get("answerable", True)]
    quality_rows = []
    for rule, rule_floors in (("total only (spec)", None), ("total + floors (used)", floors)):
        for t in QUALITY_THRESHOLDS:
            kept, _ = apply_threshold(answerable, t, rule_floors)
            survivors, _ = _dedup([index[p["id"]] for p in kept], judged, used_sim, used_t, used_numbers)
            quality_rows.append({
                "rule": rule, "min_score": t, "kept": len(kept),
                "kept_pct_of_judged": round(len(kept) / max(len(answerable), 1), 4),
                "after_dedup": len(survivors),
                "avg_quality": _mean([p["quality_score"] for p in kept]),
                "kept_with_groundedness_below_2": sum(1 for p in kept if p["scores"]["groundedness"] < 2),
            })

    # 2. Deduplication sensitivity (on the pairs that passed validation) -----------------
    kept_now, _ = apply_threshold(judged, cfg.validate.min_quality_score, floors)
    kept_idx = [index[p["id"]] for p in kept_now]
    dedup_rows, examples = [], {}
    settings = (("question only (spec)", "question", None),
                ("question + answer, number guard (used)", "question_answer", numbers))
    for label, basis, guard in settings:
        for t in DEDUP_THRESHOLDS:
            survivors, removed = _dedup(kept_idx, judged, sims[basis], t, guard)
            dedup_rows.append({"setting": label, "threshold": t, "input": len(kept_idx),
                               "removed": len(removed), "survivors": len(survivors),
                               "removed_pct": round(len(removed) / max(len(kept_idx), 1), 4)})
            # The weakest removals decide whether a threshold is too low: show them.
            weakest = sorted(removed, key=lambda r: r[2])[:4]
            examples[f"{basis}@{t}"] = [{"similarity": round(s, 3),
                                         "removed": judged[r]["question"], "removed_answer": judged[r]["answer"],
                                         "duplicate_of": judged[k]["question"], "kept_answer": judged[k]["answer"]}
                                        for r, k, s in weakest]

    # 4. Diversity before / after dedup ----------------------------------------------------
    survivors, _ = _dedup(kept_idx, judged, used_sim, used_t, used_numbers)
    final = [judged[i] for i in survivors]
    diversity = {
        "types_before": dict(Counter(p["question_type"] for p in kept_now).most_common()),
        "types_after": dict(Counter(p["question_type"] for p in final).most_common()),
        "first_words_before": diversity_stats(kept_now),
        "first_words_after": diversity_stats(final),
    }

    results = {
        "judged_pairs": len(judged),
        "quality_thresholds": quality_rows,
        "dedup_thresholds": dedup_rows,
        "dedup_weakest_removals": examples,
        "judge_reliability": judge_reliability(out, cfg),
        "diversity": diversity,
    }
    with open(out / "experiments.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    (out / "experiments.md").write_text(markdown(results), encoding="utf-8")
    return results


def markdown(results: dict[str, Any]) -> str:
    lines = ["### 1. Quality threshold sensitivity", "",
             "| Rule | Min score | Kept | % of judged | After dedup | Avg quality | Kept with groundedness < 2 |",
             "|---|---|---|---|---|---|---|"]
    for r in results["quality_thresholds"]:
        lines.append(f"| {r['rule']} | ≥ {r['min_score']} | {r['kept']} | {r['kept_pct_of_judged']:.0%} | "
                     f"{r['after_dedup']} | {r['avg_quality']} | {r['kept_with_groundedness_below_2']} |")
    lines += ["", "### 2. Deduplication sensitivity", "",
              "| Setting | Cosine threshold | Input | Removed | Survivors | Removed % |", "|---|---|---|---|---|---|"]
    for r in results["dedup_thresholds"]:
        lines.append(f"| {r['setting']} | {r['threshold']} | {r['input']} | {r['removed']} | {r['survivors']} | "
                     f"{r['removed_pct']:.1%} |")
    lines += ["", "### 3. Judge reliability (Cohen's kappa, judge vs human)", ""]
    jr = results.get("judge_reliability")
    if jr:
        lines += ["| Criterion | κ | Weighted κ | Exact agreement | Band |", "|---|---|---|---|---|"]
        for c, r in jr["criteria"].items():
            k = "–" if r["kappa"] is None else f"{r['kappa']:.2f}"
            w = "–" if r["weighted_kappa"] is None else f"{r['weighted_kappa']:.2f}"
            lines.append(f"| {c} | {k} | {w} | {r['agreement']:.0%} | {r['interpretation']} |")
        d = jr["decision"]
        k = "–" if d["kappa"] is None else f"{d['kappa']:.2f}"
        lines.append(f"| keep / reject | {k} | | {d['agreement']:.0%} | {d['interpretation']} |")
        lines.append(f"\nn = {jr['n']} human-labelled pairs.")
    else:
        lines.append("_Not yet computed: label data/output/human_labels.json (dashboard tab 4)._")
    div = results["diversity"]
    lines += ["", "### 4. Question-type and opening-word diversity", "",
              "| | Before dedup | After dedup |", "|---|---|---|"]
    for t in sorted(set(div["types_before"]) | set(div["types_after"])):
        lines.append(f"| {t} | {div['types_before'].get(t, 0)} | {div['types_after'].get(t, 0)} |")
    b, a = div["first_words_before"], div["first_words_after"]
    lines.append(f"| distinct opening words | {b['unique_first_words']} | {a['unique_first_words']} |")
    lines.append(f"| most common opening | “{b['top_first_word']}” {b['top_first_word_share']:.0%} | "
                 f"“{a['top_first_word']}” {a['top_first_word_share']:.0%} |")
    return "\n".join(lines)


if __name__ == "__main__":
    res = run_experiments()
    print(markdown(res))
    print("\nWeakest removals per setting (to judge whether a threshold is too low):")
    for key in (f"question@0.85", f"question_answer@{CONFIG.deduplicate.similarity_threshold}"):
        print(f"\n  {key}:")
        for ex in res["dedup_weakest_removals"].get(key, []):
            print(f"    {ex['similarity']:.3f}  REMOVED: {ex['removed']}\n           KEPT:    {ex['duplicate_of']}")
