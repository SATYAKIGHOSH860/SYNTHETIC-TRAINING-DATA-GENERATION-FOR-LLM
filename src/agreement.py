"""Judge validation: does the LLM judge agree with a human?

Without this step the quality claims are circular: the pipeline would be
reporting its own judge's opinion of its own generator. A person scores a
stratified sample with the same rubric, and Cohen's kappa measures agreement
beyond chance.

The tool only prepares the file and computes the statistic. It never fills in
human scores: the gold labels must come from a person.

Usage:
    python -m src.agreement sample   # write data/output/human_labels.json
    python -m src.agreement score    # compute kappa once the labels are filled in
(The dashboard's "Judge Reliability" tab offers the same labelling in a form.)
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.config import CONFIG, Config, resolve_path
from src.export import load_jsonl
from src.validate import keep_decision

CRITERIA = ("groundedness", "specificity", "completeness")
SCORE_BANDS = (("0-2", 0, 2), ("3", 3, 3), ("4", 4, 4), ("5", 5, 5), ("6", 6, 6))

LABEL_INSTRUCTIONS = (
    "Score each pair against its passage WITHOUT looking at the judge's scores. "
    "Each criterion is 0, 1 or 2. "
    "Groundedness: 0 = contains information absent from the passage; 1 = mixes passage content "
    "with outside knowledge; 2 = every claim traceable to the passage. "
    "Specificity: 0 = vague, could apply to any text; 1 = somewhat specific; 2 = tests knowledge "
    "of this particular content. "
    "Completeness: 0 = truncated or bare yes/no; 1 = partial; 2 = fully answers, no follow-up needed."
)


class LabelsExistError(RuntimeError):
    """Refuse to overwrite a labels file that already holds human scores."""


def labels_path(output_dir: str | Path | None = None, cfg: Config = CONFIG) -> Path:
    return resolve_path(output_dir or cfg.paths.output_dir) / "human_labels.json"


def _band(score: int) -> str:
    for name, lo, hi in SCORE_BANDS:
        if lo <= score <= hi:
            return name
    return SCORE_BANDS[0][0]


def stratified_sample(pairs: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Reproducible sample spread evenly across score bands.

    Why stratify: pairs that passed the judge dominate the population, so a
    plain random sample would be almost all 5s and 6s, and agreement on the
    keep/reject boundary (the decision that matters) would go untested.
    Unused quota from a sparse band is handed to the others.
    """
    rng = random.Random(seed)
    bands: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in SCORE_BANDS}
    for p in sorted(pairs, key=lambda p: p["id"]):
        bands[_band(int(p.get("quality_score") or 0))].append(p)
    for members in bands.values():
        rng.shuffle(members)
    n = min(n, len(pairs))
    quota = {name: 0 for name in bands}
    remaining = n
    while remaining > 0:
        open_bands = [b for b in bands if quota[b] < len(bands[b])]
        if not open_bands:
            break
        for b in open_bands:
            if remaining == 0:
                break
            quota[b] += 1
            remaining -= 1
    sample = [p for b, members in bands.items() for p in members[: quota[b]]]
    rng.shuffle(sample)  # hide the band order from the labeller
    return sample


def sample_for_labelling(
    pairs: list[dict[str, Any]],
    n: int = 40,
    seed: int = 42,
    path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Write human_labels.json: pair id, question, answer, passage and EMPTY score fields.

    Only answerable pairs the judge actually scored are eligible (kept and
    rejected alike). Judge scores are deliberately left out of the file so the
    human labels blind.
    """
    path = Path(path) if path else labels_path()
    if path.is_file() and not force:
        existing = load_labels(path)
        if any(item.get(c) is not None for item in existing.get("items", []) for c in CRITERIA):
            raise LabelsExistError(
                f"{path} already contains human scores. Refusing to overwrite them; "
                "pass force=True (or delete the file) to start a new labelling round."
            )
    eligible = [p for p in pairs if p.get("answerable", True) and p.get("judge_ok") and p.get("scores")]
    if not eligible:
        raise ValueError("No judged answerable pairs available to sample. Run the pipeline first.")
    chosen = stratified_sample(eligible, n, seed)
    doc = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "instructions": LABEL_INSTRUCTIONS,
        "seed": seed,
        "labeller": "",
        "items": [
            {
                "id": p["id"],
                "source": p.get("source"),
                "page": p.get("page"),
                "passage": p.get("chunk_text", ""),
                "question": p["question"],
                "answer": p["answer"],
                "groundedness": None,
                "specificity": None,
                "completeness": None,
                "notes": "",
            }
            for p in chosen
        ],
    }
    save_labels(doc, path)
    print(f"Wrote {len(chosen)} items for labelling to {path}")
    return path


def load_labels(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else labels_path()
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def clear_labels(path: str | Path | None = None) -> Path:
    """Blank every score and note in human_labels.json (same sample, fresh start)."""
    doc = load_labels(path)
    for item in doc.get("items", []):
        item.update({c: None for c in CRITERIA}, notes="")
    return save_labels(doc, path)


def save_labels(doc: dict[str, Any], path: str | Path | None = None) -> Path:
    path = Path(path) if path else labels_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return path


def is_complete(item: dict[str, Any]) -> bool:
    return all(item.get(c) in (0, 1, 2) for c in CRITERIA)


def interpret_kappa(kappa: float | None) -> str:
    """Bands used throughout the report: >0.8 strong, 0.6-0.8 substantial, 0.4-0.6 moderate, <0.4 weak."""
    if kappa is None:
        return "undefined"
    if kappa > 0.8:
        return "strong"
    if kappa >= 0.6:
        return "substantial"
    if kappa >= 0.4:
        return "moderate"
    return "weak"


def _kappa(a: list[int], b: list[int], labels: list[int], weights: str | None = None) -> float | None:
    """Cohen's kappa, or None when it is undefined (both raters used one identical label)."""
    if not a:
        return None
    if len(set(a) | set(b)) == 1:
        return None
    value = cohen_kappa_score(a, b, labels=labels, weights=weights)
    return None if np.isnan(value) else round(float(value), 4)


def compute_kappa(
    human_labels: list[dict[str, Any]],
    judge_scores: dict[str, dict[str, int]],
    min_score: int = 4,
    floors: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Agreement between human and judge, per criterion and on keep/reject.

    ``human_labels``: items with an ``id`` and 0-2 scores per criterion
    (incomplete items are ignored). ``judge_scores``: id -> criterion scores.
    The keep/reject decision for both raters uses the pipeline's own rule
    (validate.keep_decision: total >= min_score and the per-criterion floors).
    Reports unweighted Cohen's kappa (the headline number), linear-weighted
    kappa (credits near-misses on the ordinal 0-2 scale), raw percent
    agreement, and the keep/reject confusion matrix
    [[both reject, human reject & judge keep], [human keep & judge reject, both keep]].
    """
    rows = [(h, judge_scores[h["id"]]) for h in human_labels
            if is_complete(h) and h.get("id") in judge_scores]
    result: dict[str, Any] = {"n": len(rows), "min_quality_score": min_score,
                              "min_criterion_scores": dict(floors or {}), "criteria": {}}
    for c in CRITERIA:
        hs = [int(h[c]) for h, _ in rows]
        js = [int(j[c]) for _, j in rows]
        k = _kappa(hs, js, [0, 1, 2])
        result["criteria"][c] = {
            "kappa": k,
            "weighted_kappa": _kappa(hs, js, [0, 1, 2], weights="linear"),
            "agreement": round(sum(x == y for x, y in zip(hs, js)) / len(hs), 4) if hs else None,
            "interpretation": interpret_kappa(k),
        }
    h_keep = [int(keep_decision({c: int(h[c]) for c in CRITERIA}, min_score, floors)) for h, _ in rows]
    j_keep = [int(keep_decision({c: int(j[c]) for c in CRITERIA}, min_score, floors)) for _, j in rows]
    k = _kappa(h_keep, j_keep, [0, 1])
    result["decision"] = {
        "kappa": k,
        "agreement": round(sum(x == y for x, y in zip(h_keep, j_keep)) / len(h_keep), 4) if h_keep else None,
        "interpretation": interpret_kappa(k),
        "confusion_matrix": confusion_matrix(h_keep, j_keep, labels=[0, 1]).tolist() if rows else [[0, 0], [0, 0]],
    }
    h_tot = [sum(int(h[c]) for c in CRITERIA) for h, _ in rows]
    j_tot = [sum(int(j[c]) for c in CRITERIA) for _, j in rows]
    result["mean_total_human"] = round(float(np.mean(h_tot)), 3) if h_tot else None
    result["mean_total_judge"] = round(float(np.mean(j_tot)), 3) if j_tot else None
    result["warnings"] = label_warnings(rows)
    return result


def label_warnings(rows: list[tuple[dict[str, Any], dict[str, int]]]) -> list[str]:
    """Flag human labels that cannot measure agreement.

    Why: when one rater gives the same score to every pair, Cohen's kappa is 0
    (or undefined) by construction, whatever the other rater did. That number
    would look like "weak agreement" but it says nothing about the judge.
    """
    if len(rows) < 5:
        return []
    warnings = []
    patterns = {tuple(int(h[c]) for c in CRITERIA) for h, _ in rows}
    if len(patterns) == 1:
        g, s, c = next(iter(patterns))
        warnings.append(f"Every labelled pair has the same human scores ({g}/{s}/{c}). Kappa cannot measure "
                        "agreement when one rater never varies, so these results say nothing about the judge. "
                        "Score each pair on its own merits.")
    else:
        for crit in CRITERIA:
            values = {int(h[crit]) for h, _ in rows}
            if len(values) == 1:
                warnings.append(f"All human {crit} scores are {values.pop()}, so the {crit} kappa is "
                                "uninformative.")
    return warnings


def judge_scores_from_audit(output_dir: str | Path | None = None, cfg: Config = CONFIG) -> dict[str, dict[str, int]]:
    """id -> judge sub-scores, read from judged_pairs.jsonl (the full audit file)."""
    path = resolve_path(output_dir or cfg.paths.output_dir) / "judged_pairs.jsonl"
    return {p["id"]: p["scores"] for p in load_jsonl(path) if p.get("scores") and p.get("judge_ok")}


def agreement_report(output_dir: str | Path | None = None, cfg: Config = CONFIG) -> dict[str, Any]:
    """Compute kappa from human_labels.json + judged_pairs.jsonl and save judge_agreement.json."""
    out = resolve_path(output_dir or cfg.paths.output_dir)
    doc = load_labels(out / "human_labels.json")
    result = compute_kappa(doc.get("items", []), judge_scores_from_audit(out, cfg), cfg.validate.min_quality_score,
                           cfg.validate.min_criterion_scores.to_dict())
    result["labelled"] = sum(1 for i in doc.get("items", []) if is_complete(i))
    result["total_items"] = len(doc.get("items", []))
    result["labeller"] = doc.get("labeller", "")
    with open(out / "judge_agreement.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


def _print_report(result: dict[str, Any]) -> None:
    print(f"\nJudge vs human agreement on {result['n']} labelled pairs")
    print(f"{'criterion':<14}{'kappa':>8}{'weighted':>10}{'agree':>8}  interpretation")
    for c, r in result["criteria"].items():
        fmt = lambda v: "   n/a" if v is None else f"{v:>6.3f}"
        print(f"{c:<14}{fmt(r['kappa']):>8}{fmt(r['weighted_kappa']):>10}"
              f"{'n/a' if r['agreement'] is None else f'{r['agreement']:.0%}':>8}  {r['interpretation']}")
    d = result["decision"]
    kappa_txt = "n/a" if d["kappa"] is None else f"{d['kappa']:.3f}"
    print(f"{'keep/reject':<14}{kappa_txt:>8}{'':>10}"
          f"{'n/a' if d['agreement'] is None else f'{d['agreement']:.0%}':>8}  {d['interpretation']}")
    print(f"Confusion matrix [human rows x judge cols, reject/keep]: {d['confusion_matrix']}")
    for warning in result.get("warnings", []):
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "score"
    out_dir = resolve_path(CONFIG.paths.output_dir)
    if command == "sample":
        audit = load_jsonl(out_dir / "judged_pairs.jsonl")
        sample_for_labelling(audit, CONFIG.agreement.sample_size, CONFIG.agreement.seed,
                             out_dir / "human_labels.json", force="--force" in sys.argv)
        print("Now fill in groundedness/specificity/completeness (0, 1 or 2) for every item, "
              "or use the dashboard's Judge Reliability tab, then run: python -m src.agreement score")
    elif command == "score":
        path = out_dir / "human_labels.json"
        if not path.is_file():
            raise SystemExit("No human_labels.json yet. Run: python -m src.agreement sample")
        report = agreement_report(out_dir)
        if report["n"] == 0:
            raise SystemExit(f"{path} has no completed items yet. Fill in the scores first.")
        print(f"Labelled {report['labelled']}/{report['total_items']} items.")
        _print_report(report)
    else:
        raise SystemExit("Usage: python -m src.agreement [sample|score] [--force]")
