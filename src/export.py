"""Write the dataset, the evidence files and the run statistics.

JSONL (one JSON object per line) is the format TRL, Axolotl and the Hugging
Face trainers read directly, and it can be streamed and appended safely.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CONFIG, Config, resolve_path
from src.deduplicate import diversity_stats

EXPORT_FIELDS = ("question", "answer", "source", "page", "question_type", "answerable", "quality_score")

DISCLAIMER = (
    "Educational and research use only. This dataset was generated automatically from public "
    "WHO guideline documents by a language model. It is not medical advice and must not be used "
    "for clinical decision-making. Pairs may contain errors inherited from the generating model; "
    "automated validation reduces but does not eliminate them."
)


def clean_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the public fields; drop chunk_text, raw scores and judge internals."""
    return [{field: p.get(field) for field in EXPORT_FIELDS} for p in pairs]


def save_jsonl(pairs: list[dict[str, Any]], path: str | Path) -> Path:
    """UTF-8 JSONL with ``ensure_ascii=False`` so µ, ≥ and accented names stay readable."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = resolve_path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_chatml_format(
    pairs: list[dict[str, Any]],
    include_context: bool = False,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Convert pairs to ``{"messages": [...]}`` records for supervised fine-tuning.

    With ``include_context`` the user turn carries the source passage, which is
    what makes the unanswerable examples learnable: the model can only learn
    "this is not in the document" if it is shown the document.
    """
    records = []
    for p in pairs:
        if include_context and p.get("chunk_text"):
            user = f"Context:\n{p['chunk_text']}\n\nQuestion: {p['question']}"
        else:
            user = p["question"]
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": p["answer"]}]
        records.append({"messages": messages})
    return records


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def build_stats(
    *,
    raw_pairs: list[dict[str, Any]],
    judged: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    final: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    cfg: Config = CONFIG,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every count and distribution the dashboard and README report."""
    answerable_judged = [p for p in judged if p.get("answerable", True)]
    unans_judged = [p for p in judged if not p.get("answerable", True)]
    final_answerable = [p for p in final if p.get("answerable", True)]
    final_unans = [p for p in final if not p.get("answerable", True)]
    kept_answerable = [p for p in kept if p.get("answerable", True)]

    score_dist = Counter(int(p.get("quality_score") or 0) for p in answerable_judged)
    criteria = ("groundedness", "specificity", "completeness")
    sub_means = {
        c: _mean([p["scores"][c] for p in answerable_judged if p.get("judge_ok") and p.get("scores")])
        for c in criteria
    }
    raw_count, validated, final_count = len(raw_pairs), len(kept), len(final)
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_count": raw_count,
        "raw_answerable": sum(1 for p in raw_pairs if p.get("answerable", True)),
        "raw_unanswerable": sum(1 for p in raw_pairs if not p.get("answerable", True)),
        "validated_count": validated,
        "final_count": final_count,
        "pass_rate": round(validated / raw_count, 4) if raw_count else 0.0,
        "answerable_pass_rate": round(len(kept_answerable) / len(answerable_judged), 4) if answerable_judged else 0.0,
        "rejected_count": len(judged) - validated,
        "duplicates_removed": len(duplicates),
        "duplicate_rate": round(len(duplicates) / validated, 4) if validated else 0.0,
        "final_share_of_raw": round(final_count / raw_count, 4) if raw_count else 0.0,
        "unanswerable_generated": len(unans_judged),
        "unanswerable_kept": sum(1 for p in kept if not p.get("answerable", True)),
        "unanswerable_count": len(final_unans),
        "average_quality_score": _mean([p["quality_score"] for p in final_answerable]),
        "average_quality_score_all_judged": _mean([p.get("quality_score") or 0 for p in answerable_judged]),
        "average_subscores_all_judged": sub_means,
        "judge_failures": sum(1 for p in judged if not p.get("judge_ok") and "failed" in (p.get("judge_note") or "")),
        "per_source_counts": dict(Counter(p["source"] for p in final).most_common()),
        "per_source_raw_counts": dict(Counter(p["source"] for p in raw_pairs).most_common()),
        "question_type_distribution": dict(Counter(p["question_type"] for p in final).most_common()),
        "question_type_distribution_raw": dict(Counter(p["question_type"] for p in raw_pairs).most_common()),
        "score_distribution": {str(s): score_dist.get(s, 0) for s in range(7)},
        "diversity_before_dedup": diversity_stats(kept),
        "diversity": diversity_stats(final),
        "config": cfg.to_dict(),
    }
    if extra:
        stats.update(extra)
    return stats


def save_stats(stats: dict[str, Any], path: str | Path) -> Path:
    """Write pipeline_stats.json. The dashboard reads this file, so it must be valid JSON."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic: the dashboard never reads a half-written file
    return path


def print_report(stats: dict[str, Any]) -> None:
    raw = stats["raw_count"] or 1
    rows = [
        ("Raw pairs generated", stats["raw_count"]),
        ("Passed validation", stats["validated_count"]),
        ("After deduplication", stats["final_count"]),
    ]
    print("\n" + "=" * 60)
    print(" DATASET REPORT")
    print("=" * 60)
    for label, n in rows:
        print(f" {label:<28} {n:>6}   ({n / raw:6.1%} of raw)")
    print(f" {'Rejected by judge':<28} {stats['rejected_count']:>6}")
    print(f" {'Duplicates removed':<28} {stats['duplicates_removed']:>6}")
    print(f" {'Unanswerable in dataset':<28} {stats['unanswerable_count']:>6}")
    avg = stats.get("average_quality_score")
    print(f" {'Average quality score':<28} {avg if avg is not None else 'n/a':>6} / 6")
    print(f" {'Answerable pass rate':<28} {stats['answerable_pass_rate']:>6.1%}")
    div = stats.get("diversity", {})
    if div.get("questions"):
        print(f" {'Unique question starters':<28} {div['unique_first_words']:>6}   "
              f"(top: '{div['top_first_word']}' {div['top_first_word_share']:.0%})")
    print("=" * 60)


def dataset_card(stats: dict[str, Any] | None, repo_id: str) -> str:
    lines = [
        "---", "license: cc-by-nc-sa-3.0", "language: [en]",
        "task_categories: [question-answering]", "tags: [synthetic, medical, who-guidelines]", "---",
        f"# {repo_id.split('/')[-1]}", "",
        f"> **Disclaimer.** {DISCLAIMER}", "",
        "Synthetic question-answer pairs generated from public WHO guideline PDFs, scored by an "
        "LLM judge from a different model family (groundedness, specificity, completeness; kept at "
        ">= 4/6 with groundedness 2/2 required) and semantically deduplicated on question + answer. "
        "Includes deliberately unanswerable questions with an abstention answer. Source documents "
        "are licensed CC BY-NC-SA 3.0 IGO by WHO; this derived dataset follows the same terms.",
        "", "Fields: " + ", ".join(f"`{f}`" for f in EXPORT_FIELDS),
    ]
    if stats:
        lines += ["", f"Pairs: {stats.get('final_count')}; average quality {stats.get('average_quality_score')}/6; "
                      f"unanswerable {stats.get('unanswerable_count')}."]
    return "\n".join(lines) + "\n"


def push_to_hub(pairs: list[dict[str, Any]], repo_id: str, private: bool = True,
                stats: dict[str, Any] | None = None) -> str:
    """Upload the cleaned dataset (with a disclaimer card) to the Hugging Face Hub.

    Explicit call only (``main.py --push-to-hub``), never automatic.
    Authenticates with HF_TOKEN from .env. Private by default.
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not set in .env; it is required for --push-to-hub.")
    from datasets import Dataset
    from huggingface_hub import HfApi

    Dataset.from_list(clean_pairs(pairs)).push_to_hub(repo_id, token=token, private=private)
    HfApi(token=token).upload_file(
        path_or_fileobj=dataset_card(stats, repo_id).encode("utf-8"),
        path_in_repo="README.md", repo_id=repo_id, repo_type="dataset",
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Pushed {len(pairs)} pairs to {url}")
    return url


def _judged_signature(pairs: list[dict[str, Any]]) -> dict[str, str]:
    """id -> answer and judge scores: what the agreement and experiment files were computed from."""
    return {p["id"]: json.dumps([p.get("answer"), p.get("scores")], sort_keys=True, ensure_ascii=False)
            for p in pairs if p.get("id")}


def archive_stale_files(output_dir: str | Path, judged: list[dict[str, Any]]) -> tuple[Path, list[str]] | None:
    """Before a run writes its dataset, move aside files that describe a different, earlier dataset.

    A run rewrites the dataset and evidence files, but human_labels.json,
    judge_agreement.json and experiments.* belong to the dataset they were made
    from. human_labels.json holds a person's scores (gold data), so nothing is
    ever deleted: stale files move to ``output_dir/previous/<UTC time>/``.
    Labels are stale when any labelled pair is missing from the new run; the
    agreement and experiment files when the judged pairs or their scores
    changed. An identical re-run (same pairs, same scores) moves nothing.
    Returns (archive folder, moved file names), or None.
    """
    out = resolve_path(output_dir)
    new = _judged_signature(judged)
    old_path = out / "judged_pairs.jsonl"
    old = _judged_signature(load_jsonl(old_path)) if old_path.is_file() else {}
    stale: list[str] = []
    labels = out / "human_labels.json"
    if labels.is_file():
        with open(labels, "r", encoding="utf-8") as fh:
            items = json.load(fh).get("items", [])
        if any(item.get("id") not in new for item in items):
            stale += ["human_labels.json", "judge_agreement.json"]
    if old != new:
        stale += ["judge_agreement.json", "experiments.json", "experiments.md"]
    moved = [name for name in dict.fromkeys(stale) if (out / name).is_file()]
    if not moved:
        return None
    dest = out / "previous" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for name in moved:
        os.replace(out / name, dest / name)
    return dest, moved


def write_outputs(
    output_dir: str | Path,
    *,
    kept: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    final: list[dict[str, Any]],
    stats: dict[str, Any],
    cfg: Config = CONFIG,
) -> dict[str, Path]:
    """Write the dataset, ChatML, unanswerable and audit files plus the stats.

    rejected_pairs.jsonl (validate) and duplicate_pairs.jsonl (deduplicate) are
    written by their own stages. judged_pairs.jsonl is the full audit trail:
    every judged pair with sub-scores and a status of kept / rejected /
    duplicate. The dashboard's quality explorer and the human-labelling
    sample are built from it.
    """
    out = resolve_path(output_dir)
    final_ids = {p["id"] for p in final}
    judged = kept + rejected
    unans = [p for p in judged if not p.get("answerable", True)]
    audit = [{**p, "status": "kept" if p["id"] in final_ids else "duplicate"} for p in kept]
    audit += [{**p, "status": "rejected"} for p in rejected]
    paths = {
        "dataset": save_jsonl(clean_pairs(final), out / "synthetic_dataset.jsonl"),
        "chatml": save_jsonl(to_chatml_format(final, cfg.export.chatml_include_context),
                             out / "synthetic_dataset_chatml.jsonl"),
        "unanswerable": save_jsonl(
            [{**clean_pairs([p])[0], "kept": p["id"] in final_ids,
              "check": p.get("unanswerable_check"), "reason": p.get("reject_reason") or p.get("judge_note", "")}
             for p in unans],
            out / "unanswerable_pairs.jsonl"),
        "judged": save_jsonl(audit, out / "judged_pairs.jsonl"),
        "stats": save_stats(stats, out / "pipeline_stats.json"),
    }
    return paths


if __name__ == "__main__":
    out_dir = resolve_path(CONFIG.paths.output_dir)
    stats_path = out_dir / "pipeline_stats.json"
    if not stats_path.is_file():
        raise SystemExit(f"No results yet at {stats_path}. Run `python main.py` first.")
    with open(stats_path, encoding="utf-8") as fh:
        print_report(json.load(fh))
    dataset = load_jsonl(out_dir / "synthetic_dataset.jsonl")
    print(f"synthetic_dataset.jsonl: {len(dataset)} rows; first row:")
    print(json.dumps(dataset[0], ensure_ascii=False, indent=1) if dataset else "(empty)")
