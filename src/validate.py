"""LLM-as-judge: score every generated pair against its source passage.

Rubric (each criterion 0, 1 or 2; total 0-6):

| Criterion     | 0                                   | 1                              | 2                                  |
|---------------|-------------------------------------|--------------------------------|------------------------------------|
| Groundedness  | info absent from passage            | passage + outside knowledge    | every claim traceable to passage   |
| Specificity   | vague, fits any text                | somewhat specific              | tests this particular content      |
| Completeness  | truncated or bare yes/no            | partial                        | fully answers, no follow-up needed |

Pairs scoring >= ``validate.min_quality_score`` (default 4) are kept, provided
they also meet ``validate.min_criterion_scores`` (default: groundedness 2,
specificity 1, completeness 1). Why 4: stricter (5-6) discards too many usable
pairs; looser (2-3) admits noise that degrades fine-tuning. Why floors as well:
the total alone accepts 2+2+0, e.g. a fluent answer with a fabricated number.
Unanswerable pairs use a separate check: is the question genuinely not
answerable from the passage (and on-topic), and is the answer the abstention
string.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from src.config import CONFIG, Config, resolve_path
from src.generate import append_checkpoint, load_checkpoint
from src.llm import FatalLLMError, LLMCallError, Usage, invoke_json

ProgressCallback = Callable[[int, int, str], None]

PROMPT_VERSION = "judge-v2"
CRITERIA = ("groundedness", "specificity", "completeness")


def judge_system_prompt(cfg: Config = CONFIG) -> str:
    return (
        f"You are a strict, impartial examiner auditing training data for a {cfg.domain.name} "
        "assistant. You score question-answer pairs against their source passage using a fixed "
        "rubric, and you reply with valid JSON only."
    )


def judge_prompt(passage: str, pairs: list[dict[str, Any]], cfg: Config = CONFIG) -> str:
    """Rubric prompt for one or more answerable pairs sharing a passage.

    The keep/reject threshold is deliberately not shown to the judge: it
    scores on the rubric alone, so the threshold can be varied later (Task 11)
    without re-judging and without the judge nudging borderline scores.
    """
    items = [{"id": i + 1, "question": p["question"], "answer": p["answer"]} for i, p in enumerate(pairs)]
    source = pairs[0].get("source", "") if pairs else ""
    origin = f" It is an excerpt from {source}, page {pairs[0].get('page')}." if source else ""
    return f"""Score each question-answer pair against the SOURCE PASSAGE using this rubric. Give every criterion 0, 1 or 2.
The passage comes from a {cfg.domain.name} document.{origin} Naming the organisation or guideline the passage belongs to is not outside knowledge.

GROUNDEDNESS - is every claim in the ANSWER supported by the passage? (Judge the answer's claims, not the question's wording.)
  0 = the answer contains information absent from the passage (hallucination) or contradicts it
  1 = mostly from the passage, but mixes in outside knowledge or details the passage does not state
  2 = every claim is traceable to the passage
SPECIFICITY - does the question test this particular content?
  0 = vague or generic; could be asked of almost any text
  1 = somewhat specific
  2 = tests knowledge of specific content in this passage
COMPLETENESS - does the answer fully answer the question?
  0 = truncated, a bare yes/no, or does not answer the question
  1 = partial; misses part of what was asked
  2 = fully answers, no follow-up needed

Judge each pair independently. Check the answer against the passage claim by claim; do not reward a fluent answer that adds unsupported details.
total = groundedness + specificity + completeness.
reject_reason = one short sentence naming the main weakness if any criterion is below 2 (for example "answer states a dosage not present in the passage"); "" if all three are 2.

SOURCE PASSAGE:
\"\"\"
{passage[: cfg.validate.passage_chars]}
\"\"\"

PAIRS:
{json.dumps(items, ensure_ascii=False, indent=1)}

Output ONLY a JSON object, with no prose before or after it:
{{"scores": [{{"id": 1, "groundedness": 0, "specificity": 0, "completeness": 0, "total": 0, "reject_reason": ""}}]}}"""


def unanswerable_judge_prompt(passage: str, pair: dict[str, Any], cfg: Config = CONFIG) -> str:
    return f"""The QUESTION below was written so that the SOURCE PASSAGE does not answer it. Verify that.

1. unanswerable: true if the passage contains no information that answers the question, fully or partly; false if the passage answers it even partly.
2. plausible: true if the question is on the passage's topic and is a natural question for {cfg.domain.name}; false if it is off-topic, nonsensical, or about the document itself.
reason = one short sentence explaining any false verdict; "" if both are true.

SOURCE PASSAGE:
\"\"\"
{passage[: cfg.validate.passage_chars]}
\"\"\"

QUESTION: {pair['question']}

Output ONLY a JSON object, with no prose before or after it:
{{"unanswerable": true, "plausible": true, "reason": ""}}"""


# ---------------------------------------------------------------------------
# Score parsing and filtering
# ---------------------------------------------------------------------------
def _as_score(value: Any) -> int | None:
    """Coerce a criterion score to 0/1/2; None if it is not a valid score."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number not in (0, 1, 2):
        return None
    return int(number)


def parse_scores(item: Any) -> dict[str, Any] | None:
    """Validate one judge score object. The total is recomputed from sub-scores.

    Why recompute: the model's arithmetic is not trusted. If its "total"
    disagrees with the sum of its own sub-scores, the sum wins.
    """
    if not isinstance(item, dict):
        return None
    scores = {c: _as_score(item.get(c)) for c in CRITERIA}
    if any(v is None for v in scores.values()):
        return None
    reason = item.get("reject_reason", "")
    return {
        "scores": scores,
        "quality_score": sum(scores.values()),
        "judge_note": " ".join(str(reason).split()) if reason else "",
    }


def fallback_reason(scores: dict[str, int]) -> str:
    weak = [f"{c} {v}/2" for c, v in scores.items() if v < 2]
    return "low " + ", ".join(weak) if weak else ""


def keep_decision(scores: dict[str, int], min_score: int, floors: dict[str, int] | None = None) -> bool:
    """The keep rule for one set of rubric scores: total >= min_score AND every floor met.

    Why floors on top of the total: a sum alone lets 2+2+0 = 4 through, i.e. a
    specific, complete answer containing a fabricated number. For medical
    training data groundedness is non-negotiable, so it has its own floor.
    Shared by validation, the threshold experiment and the human-vs-judge
    keep/reject comparison, so all three apply exactly the same rule.
    """
    if sum(scores.values()) < min_score:
        return False
    return all(scores.get(c, 0) >= floor for c, floor in (floors or {}).items())


def passes(pair: dict[str, Any], min_score: int, floors: dict[str, int] | None = None) -> bool:
    """Answerable pairs: judged successfully and keep_decision(); unanswerable: both checks passed."""
    if pair.get("answerable", True):
        return bool(pair.get("judge_ok")) and bool(pair.get("scores")) and \
            keep_decision(pair["scores"], min_score, floors)
    return bool(pair.get("judge_ok"))


def apply_threshold(
    pairs: list[dict[str, Any]], min_score: int, floors: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split judged pairs into (kept, rejected) and set ``reject_reason``.

    Pure function: used by validate_all, the threshold experiment and the tests.
    """
    kept, rejected = [], []
    for pair in pairs:
        p = dict(pair)
        if passes(p, min_score, floors):
            p["reject_reason"] = ""
            kept.append(p)
        else:
            if p.get("answerable", True) and p.get("judge_ok"):
                p["reject_reason"] = p.get("judge_note") or fallback_reason(p.get("scores", {}))
            else:
                p["reject_reason"] = p.get("judge_note") or "failed validation"
            rejected.append(p)
    return kept, rejected


# ---------------------------------------------------------------------------
# Judge calls
# ---------------------------------------------------------------------------
def _judge_kwargs(cfg: Config, usage: Usage, api_key: str | None) -> dict[str, Any]:
    return dict(model_id=cfg.model.judge_id, temperature=cfg.model.judge_temperature,
                system=judge_system_prompt(cfg), max_tokens=cfg.model.judge_max_tokens,
                usage=usage, cfg=cfg, api_key=api_key)


def score_pairs(passage: str, pairs: list[dict[str, Any]], cfg: Config = CONFIG, *,
                usage: Usage, api_key: str | None = None) -> list[dict[str, Any] | str]:
    """Judge answerable pairs that share ``passage``. Returns a result or an error string per pair.

    Pairs missing from a grouped reply are re-judged one at a time, so a
    partial answer from the judge never silently drops a pair.
    """
    results: list[dict[str, Any] | str] = ["judge returned no score"] * len(pairs)
    try:
        raw = invoke_json(user=judge_prompt(passage, pairs, cfg), expect=dict,
                          expected_output_tokens=70 * len(pairs) + 60, **_judge_kwargs(cfg, usage, api_key))
        items = raw.get("scores", [raw] if len(pairs) == 1 else [])
        by_id = {}
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                try:
                    by_id[int(item.get("id", 1 if len(pairs) == 1 else -1))] = item
                except (TypeError, ValueError):
                    continue
        for i in range(len(pairs)):
            parsed = parse_scores(by_id.get(i + 1))
            if parsed:
                results[i] = parsed
    except FatalLLMError:
        raise
    except LLMCallError as exc:
        results = [f"judge call failed: {exc}"] * len(pairs)

    if len(pairs) > 1:
        for i, res in enumerate(results):
            if isinstance(res, str):
                results[i] = score_pairs(passage, [pairs[i]], cfg, usage=usage, api_key=api_key)[0]
    return results


def check_unanswerable(passage: str, pair: dict[str, Any], cfg: Config = CONFIG, *,
                       usage: Usage, api_key: str | None = None) -> dict[str, Any] | str:
    """Verify an unanswerable pair. The abstention-string check is done in code, not by the LLM."""
    if pair.get("answer", "").strip() != cfg.generate.abstention_answer.strip():
        return {"unanswerable_check": {"unanswerable": None, "plausible": None, "abstention_answer": False},
                "judge_note": "answer is not the abstention string", "passed": False}
    try:
        raw = invoke_json(user=unanswerable_judge_prompt(passage, pair, cfg), expect=dict,
                          expected_output_tokens=80, **_judge_kwargs(cfg, usage, api_key))
    except FatalLLMError:
        raise
    except LLMCallError as exc:
        return f"judge call failed: {exc}"
    unans, plausible = raw.get("unanswerable"), raw.get("plausible")
    if not isinstance(unans, bool) or not isinstance(plausible, bool):
        return "judge returned an invalid verdict"
    ok = unans and plausible
    note = " ".join(str(raw.get("reason", "")).split())
    if not ok and not note:
        note = "passage answers the question" if not unans else "question is off-topic or implausible"
    return {"unanswerable_check": {"unanswerable": unans, "plausible": plausible, "abstention_answer": True},
            "judge_note": "" if ok else note, "passed": ok}


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------
def _groups(pairs: list[dict[str, Any]], cfg: Config) -> list[list[dict[str, Any]]]:
    """Answerable pairs grouped by chunk (one judge call per passage); unanswerable alone."""
    groups: dict[str, list[dict[str, Any]]] = {}
    singles: list[list[dict[str, Any]]] = []
    for p in pairs:
        if p.get("answerable", True) and cfg.validate.group_by_chunk:
            groups.setdefault(p["chunk_id"], []).append(p)
        else:
            singles.append([p])
    return list(groups.values()) + singles


def _judge_fingerprint(cfg: Config) -> str:
    settings = [PROMPT_VERSION, cfg.model.judge_id, cfg.model.judge_temperature,
                cfg.validate.passage_chars, cfg.domain.to_dict(), cfg.generate.abstention_answer]
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _group_key(group: list[dict[str, Any]]) -> str:
    joined = "|".join(p["id"] + hashlib.sha1((p["question"] + p["answer"]).encode("utf-8")).hexdigest()[:8]
                      for p in group)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:20]


def save_jsonl_records(pairs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")


def validate_all(
    pairs: list[dict[str, Any]],
    cfg: Config = CONFIG,
    progress_callback: ProgressCallback | None = None,
    *,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    api_key: str | None = None,
    report: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score every pair, split at the threshold, write rejected_pairs.jsonl.

    Temperature 0.0: judging must be reproducible. The same pair should get the
    same score on every run, or the reported metrics mean nothing.

    Every pair ends up with ``scores``, ``quality_score``, ``judge_note`` and
    ``reject_reason``. A judge failure counts as rejected with score 0; it is
    never silently kept. Checkpointing and ``progress_callback`` follow the
    same contract as generate_from_chunks.
    """
    cache = resolve_path(cache_dir or cfg.paths.cache_dir)
    out_dir = resolve_path(output_dir or cfg.paths.output_dir)
    ckpt_path = cache / "validate_checkpoint.jsonl"
    fingerprint = _judge_fingerprint(cfg)
    if not resume and ckpt_path.exists():
        ckpt_path.unlink()
    done = load_checkpoint(ckpt_path, fingerprint) if resume else {}

    groups = _groups(pairs, cfg)
    total = len(pairs)
    usage = Usage()
    pending: list[dict[str, Any]] = []
    judged: list[dict[str, Any]] = []
    failures = resumed = 0
    progress = 0
    started = time.time()
    # The checkpoint may also hold judgements of pairs that are no longer in this run
    # (e.g. chunks regenerated by --refill); only groups present now are reused and counted.
    reusable = sum(1 for g in groups if _group_key(g) in done)
    print(f"Judging {total} pairs with {cfg.model.judge_id} in {len(groups) - reusable} calls"
          + (f"; {reusable} groups reused from checkpoint" if reusable else ""))

    try:
        for g_index, group in enumerate(groups, start=1):
            key = _group_key(group)
            passage = group[0].get("chunk_text", "")
            if key in done:
                results = done[key]["results"]
                usage.add(Usage.from_dict(done[key].get("usage")))  # cost of producing this judgement
                resumed += 1
            else:
                call_usage = Usage()
                if group[0].get("answerable", True):
                    results = score_pairs(passage, group, cfg, usage=call_usage, api_key=api_key)
                else:
                    results = [check_unanswerable(passage, group[0], cfg, usage=call_usage, api_key=api_key)]
                usage.add(call_usage)
                # Failed judgements are not checkpointed, so --resume retries them.
                if not any(isinstance(r, str) for r in results):
                    pending.append({"fingerprint": fingerprint, "key": key, "results": results,
                                    "usage": call_usage.to_dict()})

            for pair, result in zip(group, results):
                p = dict(pair)
                if isinstance(result, str):
                    failures += 1
                    p.update(judge_ok=False, scores={c: 0 for c in CRITERIA}, quality_score=0,
                             judge_note=result)
                    if not p.get("answerable", True):
                        p["quality_score"] = None
                elif p.get("answerable", True):
                    p.update(judge_ok=True, **result)
                else:
                    p.update(judge_ok=result["passed"], quality_score=None, scores=None,
                             unanswerable_check=result["unanswerable_check"], judge_note=result["judge_note"])
                judged.append(p)

            progress += len(group)
            shown = [str(r["quality_score"]) if isinstance(r, dict) and "quality_score" in r
                     else ("pass" if isinstance(r, dict) and r.get("passed") else
                           "fail" if isinstance(r, dict) else "ERR") for r in results]
            print(f"  [{progress:>3}/{total}] {group[0]['source']} p{group[0]['page']}: "
                  f"{', '.join(shown)}" + (" (resumed)" if key in done else ""), flush=True)
            if progress_callback:
                progress_callback(progress, total, "validating")
            if len(pending) >= cfg.run.checkpoint_every:
                append_checkpoint(ckpt_path, pending)
    finally:
        append_checkpoint(ckpt_path, pending)

    kept, rejected = apply_threshold(judged, cfg.validate.min_quality_score,
                                     cfg.validate.min_criterion_scores.to_dict())
    save_jsonl_records(rejected, out_dir / "rejected_pairs.jsonl")

    answerable = [p for p in judged if p.get("answerable", True)]
    kept_answerable = sum(1 for p in kept if p.get("answerable", True))
    rate = len(kept) / total if total else 0.0
    elapsed = time.time() - started
    floors = cfg.validate.min_criterion_scores.to_dict()
    total_only = sum(1 for p in answerable if p.get("judge_ok")
                     and (p.get("quality_score") or 0) >= cfg.validate.min_quality_score)
    print(f"Validation done in {elapsed:.0f}s: kept {len(kept)}/{total} ({rate:.1%}); "
          f"answerable {kept_answerable}/{len(answerable)} passed >= {cfg.validate.min_quality_score}/6 "
          f"with floors {floors} ({total_only - kept_answerable} more would pass on the total alone); "
          f"{failures} judge failures. Tokens used: {usage.total_tokens:,}.")
    if report is not None:
        report.update({"judged": total, "kept": len(kept), "rejected": len(rejected),
                       "judge_failures": failures, "groups_resumed": resumed,
                       "seconds": round(elapsed, 1), "usage": usage.to_dict()})
    # Pairs are returned in their original order so downstream output is stable.
    order = {p["id"]: i for i, p in enumerate(pairs)}
    kept.sort(key=lambda p: order.get(p["id"], 0))
    rejected.sort(key=lambda p: order.get(p["id"], 0))
    return kept, rejected


if __name__ == "__main__":
    from src.generate import generate_from_chunks
    from src.ingest import chunk_documents, load_pdfs, select_chunks

    smoke_cache = resolve_path(CONFIG.paths.cache_dir) / "smoke"
    docs = load_pdfs(CONFIG.paths.raw_dir)
    sample = select_chunks(chunk_documents(docs), 5)
    generated = generate_from_chunks(sample, CONFIG, cache_dir=smoke_cache, resume=True)
    kept_pairs, rejected_pairs = validate_all(generated, CONFIG, cache_dir=smoke_cache,
                                              output_dir=smoke_cache / "output")
    for p in kept_pairs + rejected_pairs:
        status = "KEEP" if p in kept_pairs else "REJECT"
        print(f"\n[{status}] score={p.get('quality_score')} {p.get('scores') or p.get('unanswerable_check')}")
        print(f"Q: {p['question']}\nA: {p['answer']}")
        if p.get("reject_reason"):
            print(f"Reason: {p['reject_reason']}")
