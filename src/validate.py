"""
Step 3: LLM-as-judge scoring.

A second Groq call grades every generated pair against a published rubric.
This is the step that separates a dataset from a pile of model output: without
it there is no evidence the data is any good, only the hope that it is.

Rubric - three criteria, 0-2 each, total 0-6:
  groundedness  Is every claim in the answer traceable to the passage?
  specificity   Is the question specific rather than vague?
  completeness  Does the answer stand on its own?

Pairs scoring >= 4 are kept. Rejected pairs are written to
data/output/rejected_pairs.jsonl with their sub-scores and a reason - they are
evidence for the dashboard, not waste.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from pathlib import Path

from groq import Groq

try:
    import checkpoint
    from config import (
        ABSTENTION_ANSWER,
        GRADED_FILE,
        GROQ_MODEL,
        JUDGE_MODEL,
        JUDGE_CHECKPOINT_FILE,
        JUDGE_MAX_TOKENS,
        JUDGE_TEMPERATURE,
        REJECTED_FILE,
        REQUEST_DELAY,
        ensure_output_dir,
        get_groq_key,
        model_kwargs,
        use_utf8_console,
    )
    from generate import extract_json_array, extract_json_object
except ImportError:  # running as `python src/validate.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import checkpoint
    from config import (
        ABSTENTION_ANSWER,
        GRADED_FILE,
        GROQ_MODEL,
        JUDGE_MODEL,
        JUDGE_CHECKPOINT_FILE,
        JUDGE_MAX_TOKENS,
        JUDGE_TEMPERATURE,
        REJECTED_FILE,
        REQUEST_DELAY,
        ensure_output_dir,
        get_groq_key,
        model_kwargs,
        use_utf8_console,
    )
    from generate import extract_json_array, extract_json_object

# Why 800: the judge only needs enough passage to verify the answer. Sending
# the full chunk costs tokens without improving the grade, and a longer
# context makes the model more forgiving on groundedness - in a wall of text
# almost anything looks supported.
# Bumped whenever a judge prompt changes, so scores cached under the old
# rubric are never reused against the new one.
JUDGE_PROMPT_VERSION = "judge-v1"

CONTEXT_LIMIT = 800

MIN_QUALITY_SCORE = 4  # of 6
MAX_SCORE = 6

# Hard gate, independent of the total. A pair scoring 0 on groundedness is
# rejected even if it totals 4+.
#
# Why: the total alone is not safe for medical data. Measured against planted
# test pairs, an answer that invented a drug dose ("300 mg once daily", never
# stated in the passage) scored groundedness 0 + specificity 2 + completeness
# 2 = 4/6, and an answer that flatly contradicted the passage's threshold
# scored the same. Both would have passed a total-only filter and entered the
# clean dataset. A fabricated dose in a clinical training set is the single
# worst output this pipeline could produce, so groundedness gets a veto.
MIN_GROUNDEDNESS = 2

# A floor on every criterion, not just groundedness. The total alone lets
# 2+2+0 = 4 through - a specific, complete answer that is also wrong - and
# 2+0+2 through, a precise answer to a question too vague to be useful. Each
# criterion therefore has a minimum of its own.
#
# Groundedness is the strict one at 2 of 2, because for clinical material an
# answer that is "mostly supported, adds a small unstated detail" is exactly
# the failure that matters: the unstated detail is the invented dose. The
# other two sit at 1, which drops the unusable without being precious.
MIN_CRITERION_SCORES = {
    "groundedness": MIN_GROUNDEDNESS,
    "specificity": 1,
    "completeness": 1,
}


def keep_decision(
    scores: dict, min_score: int = MIN_QUALITY_SCORE, floors: dict | None = None
) -> bool:
    """
    The keep rule for one set of rubric scores: total met AND every floor met.

    The single place this rule is written down. validate_pairs, the threshold
    experiment and the human-vs-judge comparison all call it, so a change here
    cannot leave the three disagreeing about what "kept" means - which is how
    a sensitivity study ends up measuring the wrong thing.
    """
    if not scores:
        return False
    if sum(scores.values()) < min_score:
        return False
    floors = MIN_CRITERION_SCORES if floors is None else floors
    return all(scores.get(name, 0) >= floor for name, floor in floors.items())


JUDGE_PROMPT = """You are a strict data-quality judge for a medical training dataset.

Score the QUESTION and ANSWER against the PASSAGE on three criteria.
Each criterion scores 0, 1 or 2. Be strict - most pairs should not get 6.

groundedness
  2 = every claim in the answer appears in the passage
  1 = mostly supported, but adds a small detail the passage does not state
  0 = contradicts the passage, or invents information

specificity
  2 = a precise question about a named topic, threshold, term or procedure
  1 = somewhat general but still answerable on its own
  0 = vague, or refers to "the passage"/"the text"/"the document"

completeness
  2 = the answer fully answers the question and stands alone
  1 = partially answers it, or needs the question for context
  0 = does not answer the question, or is truncated

Also give "reason": one short clause (under 12 words) naming the WEAKEST
criterion and why. Example: "answer adds a dose the passage never states".

OUTPUT - a JSON object and nothing else. No markdown fences:
{{"groundedness": 0-2, "specificity": 0-2, "completeness": 0-2, "reason": "..."}}

PASSAGE:
\"\"\"
{context}
\"\"\"

QUESTION: {question}

ANSWER: {answer}
"""


# Compressed for the same reason as the generation prompt: this template was
# ~298 tokens on every judging call, and the pipeline is token-rate bound.
# The rubric is unchanged in substance - the same three criteria, the same
# 0/1/2 anchors, the same strictness instruction - only stated tersely.
#
# "reason" is now requested only for imperfect scores. It was costing output
# tokens on every pair, and a reason for a 2/2/2 pair explains nothing: the
# dashboard only ever displays reasons for pairs that were marked down.
JUDGE_BATCH_PROMPT = """Score each numbered Q&A pair against the PASSAGE.
Three criteria, each 0/1/2. Be strict - most pairs should not score 6.

groundedness: 2 = every claim is in the passage | 1 = adds a small unstated
  detail | 0 = contradicts the passage or invents information
specificity: 2 = precise, names a topic/threshold/term | 1 = general but
  answerable alone | 0 = vague, or refers to "the passage"
completeness: 2 = fully answers, stands alone | 1 = partial, or needs the
  question for context | 0 = does not answer, or is truncated

Add "reason" (under 12 words, names the weakest criterion) ONLY when a pair
scores below 2 on something. Omit it for perfect pairs.

Output ONLY a JSON array, one object per pair, same order, with "index":
[{{"index":1,"groundedness":2,"specificity":2,"completeness":2}}]

PASSAGE:
\"\"\"
{context}
\"\"\"

PAIRS:
{pairs_block}
"""

# Abstention pairs cannot be scored on the rubric above and it is important to
# see why. "This information is not available in the provided document" has no
# groundedness - it makes no claim to trace - and no completeness, because its
# whole point is not answering. Grading it 0-6 would either punish a correct
# refusal or hand it a meaningless 6 that drags the dataset average up.
#
# So they get their own two-question check, and no quality score at all. What
# matters is different: is the question really unanswerable from this passage,
# and would anyone actually ask it.
UNANSWERABLE_JUDGE_PROMPT = """The QUESTION below was written so that the
PASSAGE does NOT answer it. Verify that.

unanswerable: true if the passage contains nothing that answers the question,
  fully or in part; false if it answers it even partly.
plausible: true if the question is on the passage's topic and is one a reader
  of this material might really ask; false if it is off-topic, nonsensical, or
  about the document itself.
reason: one short sentence explaining any false verdict; "" if both are true.

Output ONLY a JSON object, no fences:
{{"unanswerable":true,"plausible":true,"reason":""}}

PASSAGE:
\"\"\"
{context}
\"\"\"

QUESTION: {question}
"""


def _coerce_score(value) -> int:
    """Clamp a model-supplied score into 0-2, defaulting to 0 if unusable."""
    try:
        return max(0, min(2, int(value)))
    except (TypeError, ValueError):
        return 0


def judge_pair(
    client: Groq, pair: dict, retries: int = 1, meter=None
) -> dict | None:
    """
    Score one pair. Returns a dict of sub-scores, or None if the judge failed.

    None is distinct from a zero score: a failed API call means "not assessed",
    which is reported separately so it never masquerades as a quality signal.
    """
    prompt = JUDGE_PROMPT.format(
        context=pair["chunk_text"][:CONTEXT_LIMIT],
        question=pair["question"],
        answer=pair["answer"],
    )

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=JUDGE_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
                **model_kwargs(JUDGE_MODEL, JUDGE_MAX_TOKENS),
            )
            if meter is not None:
                meter.record(response.usage, "judge")
            raw = response.choices[0].message.content

            # extract_json_array handles fences and stray prose; the judge
            # returns an object, so parse it directly after the same cleaning.
            from generate import strip_fences

            text = strip_fences(raw)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", text, re.S)
                parsed = json.loads(match.group(0)) if match else None

            if isinstance(parsed, dict):
                scores = {
                    "groundedness": _coerce_score(parsed.get("groundedness")),
                    "specificity": _coerce_score(parsed.get("specificity")),
                    "completeness": _coerce_score(parsed.get("completeness")),
                }
                scores["reason"] = str(parsed.get("reason", "")).strip()[:150]
                scores["total"] = (
                    scores["groundedness"]
                    + scores["specificity"]
                    + scores["completeness"]
                )
                return scores
        except Exception as exc:
            name = type(exc).__name__
            message = str(exc)
            if "RateLimit" in name or "429" in message:
                wait = _retry_wait(message, attempt)
                print(f"      rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"      judge error ({name}): {message[:110]}")

        if attempt < retries:
            time.sleep(2.0)

    return None


def _scores_from(obj: dict) -> dict:
    """Build a normalised score dict from one judge response object."""
    scores = {
        "groundedness": _coerce_score(obj.get("groundedness")),
        "specificity": _coerce_score(obj.get("specificity")),
        "completeness": _coerce_score(obj.get("completeness")),
    }
    scores["reason"] = str(obj.get("reason", "")).strip()[:150]
    scores["total"] = (
        scores["groundedness"] + scores["specificity"] + scores["completeness"]
    )
    return scores


def judge_batch(
    client: Groq, pairs: list[dict], retries: int = 1, meter=None
) -> list[dict | None]:
    """
    Score several pairs that share one passage in a single API call.

    Why batch: the free Groq tier allows 200,000 tokens per DAY. Judging each
    pair alone re-sends the 800-character passage and the ~350-token rubric
    every time, so three pairs from one chunk cost roughly 2,070 tokens.
    Sending the passage and rubric once costs about 970 - a 53% saving that
    is the difference between 70 and 105 chunks per day of quota.

    Returns a list aligned with `pairs`; an entry is None if that pair could
    not be scored.
    """
    if not pairs:
        return []

    pairs_block = "\n\n".join(
        f"[{i}]\nQUESTION: {p['question']}\nANSWER: {p['answer']}"
        for i, p in enumerate(pairs, 1)
    )
    prompt = JUDGE_BATCH_PROMPT.format(
        context=pairs[0]["chunk_text"][:CONTEXT_LIMIT],
        pairs_block=pairs_block,
    )

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=JUDGE_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
                **model_kwargs(JUDGE_MODEL, JUDGE_MAX_TOKENS),
            )
            if meter is not None:
                meter.record(response.usage, "judge")
            parsed = extract_json_array(response.choices[0].message.content)
            if parsed:
                out: list[dict | None] = [None] * len(pairs)
                for position, obj in enumerate(parsed):
                    if not isinstance(obj, dict):
                        continue
                    # Trust "index" when present, fall back to position - a
                    # mis-ordered response must not attach a score to the
                    # wrong pair.
                    try:
                        idx = int(obj.get("index", position + 1)) - 1
                    except (TypeError, ValueError):
                        idx = position
                    if 0 <= idx < len(pairs):
                        out[idx] = _scores_from(obj)
                if any(s is not None for s in out):
                    return out
        except Exception as exc:
            name = type(exc).__name__
            message = str(exc)
            if "RateLimit" in name or "429" in message:
                wait = _retry_wait(message, attempt)
                print(f"      rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"      judge error ({name}): {message[:110]}")

        if attempt < retries:
            time.sleep(2.0)

    return [None] * len(pairs)


def _retry_wait(message: str, attempt: int) -> float:
    """
    Seconds to wait after a 429, read from Groq's own message when possible.

    Groq says "Please try again in 4m24.383s". Honouring that is far better
    than a fixed backoff: guessing too short burns requests against the daily
    cap, guessing too long wastes wall-clock time.
    """
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
    if match:
        minutes = float(match.group(1) or 0)
        seconds = float(match.group(2))
        return min(minutes * 60 + seconds + 1.0, 300.0)
    return min(20.0 * (attempt + 1), 120.0)


def check_unanswerable(
    client: Groq, pair: dict, retries: int = 1, meter=None
) -> dict:
    """
    Verify one abstention pair. Returns {"passed", "check", "note"}.

    Judged one at a time rather than batched: at a 10% ratio there are only a
    handful per run, so the batching that pays for itself on answerable pairs
    would add complexity for nothing.
    """
    prompt = UNANSWERABLE_JUDGE_PROMPT.format(
        context=pair["chunk_text"][:CONTEXT_LIMIT], question=pair["question"]
    )

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=JUDGE_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
                **model_kwargs(JUDGE_MODEL, JUDGE_MAX_TOKENS),
            )
            if meter is not None:
                meter.record(response.usage, "judge")
            parsed = extract_json_object(response.choices[0].message.content)
            if parsed is not None:
                unanswerable = parsed.get("unanswerable")
                plausible = parsed.get("plausible")
                # Demand real booleans. A model that replies "maybe" has not
                # answered the question, and coercing that to True would wave
                # through exactly the pairs this check exists to catch.
                if isinstance(unanswerable, bool) and isinstance(plausible, bool):
                    # Checked in code, not by the model: the answer is a
                    # constant this pipeline wrote itself, so paying an LLM to
                    # confirm our own string literal would be absurd.
                    fixed_answer = pair.get("answer", "").strip() == ABSTENTION_ANSWER
                    passed = unanswerable and plausible and fixed_answer

                    if passed:
                        note = ""
                    elif not fixed_answer:
                        note = "answer is not the fixed abstention string"
                    elif not unanswerable:
                        note = str(parsed.get("reason") or "").strip() or (
                            "passage does answer the question"
                        )
                    else:
                        note = str(parsed.get("reason") or "").strip() or (
                            "question is off-topic or implausible"
                        )

                    return {
                        "passed": passed,
                        "check": {
                            "unanswerable": unanswerable,
                            "plausible": plausible,
                            "abstention_answer": fixed_answer,
                        },
                        "note": note,
                    }
        except Exception as exc:
            name = type(exc).__name__
            message = str(exc)
            if "RateLimit" in name or "429" in message:
                wait = _retry_wait(message, attempt)
                print(f"      rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"      abstention judge error ({name}): {message[:110]}")

        if attempt < retries:
            time.sleep(2.0)

    # Unverifiable, so not counted as a quality rejection - same treatment as
    # a failed judge call on an answerable pair.
    return {"passed": None, "check": None, "note": "judge call failed"}


def validate_pairs(
    pairs: list[dict],
    min_score: int = MIN_QUALITY_SCORE,
    delay: float = REQUEST_DELAY,
    write_rejected: bool = True,
    floors: dict | None = None,
    meter=None,
    progress_cb=None,
    bucket=None,
    workers: int | None = None,
    resume: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Judge every pair. Returns (kept, rejected).

    A pair is kept only if BOTH hold:
      total >= min_score (default 4/6), and
      every criterion meets its own floor (see MIN_CRITERION_SCORES).

    Why 4 of 6: it keeps pairs that are solid on two criteria and merely
    imperfect on a third, which is where most usable data sits. Dropping to 3
    admits pairs weak on every criterion; raising to 5 costs roughly a third
    of the data. The threshold-sensitivity experiment measures that trade-off
    rather than asserting it.

    Why floors on top of the total: see MIN_CRITERION_SCORES above - the total
    alone lets a hallucinated dose through at exactly 4/6.
    """
    client = Groq(api_key=get_groq_key())
    kept: list[dict] = []
    rejected: list[dict] = []
    judge_failed = 0
    started = time.time()

    ungrounded = 0

    print(f"Judging {len(pairs)} pairs using {JUDGE_MODEL}")
    print(f"  temperature={JUDGE_TEMPERATURE} (reproducible), threshold >= {min_score}/6")
    floors = MIN_CRITERION_SCORES if floors is None else floors
    print("  floors: " + ", ".join(f"{k} >= {v}" for k, v in floors.items()))

    # Abstention pairs go down a different path - see
    # UNANSWERABLE_JUDGE_PROMPT for why the 0-6 rubric cannot apply to them.
    answerable = [p for p in pairs if p.get("answerable", True)]
    abstentions = [p for p in pairs if not p.get("answerable", True)]
    if abstentions:
        print(
            f"  {len(answerable)} answerable on the 0-6 rubric, "
            f"{len(abstentions)} unanswerable on the abstention check"
        )

    # Scores already paid for in an earlier run of this exact judge setup.
    # Deliberately NOT keyed on the keep threshold or the floors: those are
    # applied to the scores afterwards, so re-running at a different threshold
    # costs nothing and the sensitivity experiments stay free.
    judge_fp = checkpoint.fingerprint(
        stage="judge",
        prompt=JUDGE_PROMPT_VERSION,
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
        context_limit=CONTEXT_LIMIT,
    )
    cached_scores = checkpoint.load(JUDGE_CHECKPOINT_FILE, judge_fp) if resume else {}
    reused = 0

    # Group answerable pairs by the passage they came from, so each group can
    # be judged in one call. dict preserves insertion order, so output order
    # is stable. Anything already scored is left out of the grouping entirely,
    # so a resumed run spends nothing on it.
    groups: dict[str, list[dict]] = {}
    for p in answerable:
        if p["question"] in cached_scores:
            reused += 1
            continue
        groups.setdefault(p["chunk_text"], []).append(p)

    pending_abstentions = [
        p for p in abstentions if p["question"] not in cached_scores
    ]
    reused += len(abstentions) - len(pending_abstentions)
    if reused:
        print(
            f"  resuming: {reused} pair(s) already judged in an earlier run, "
            f"no API calls needed for them"
        )

    from concurrent.futures import ThreadPoolExecutor

    from budget import (
        EST_JUDGE_FIXED,
        EST_JUDGE_PER_PAIR,
        MAX_CONCURRENCY,
        TokenBucket,
    )

    bucket = bucket or TokenBucket()
    workers = workers or MAX_CONCURRENCY
    group_list = list(groups.values())

    # Pre-size the results so each group writes into its own slot: order must
    # survive concurrency, because deduplication keeps the FIRST of every
    # duplicate group and a reshuffle would change which version survives.
    per_group: list[list[dict | None]] = [[None] * len(g) for g in group_list]
    judged_count = 0
    stopped = False
    lock = threading.Lock()

    print(
        f"  batching into {len(group_list)} calls (one per source passage), "
        f"{workers} workers, paced to {bucket.capacity:.0f} tokens/min"
    )

    def judge_group(index: int, group: list[dict]) -> None:
        nonlocal judged_count, stopped

        if stopped:
            return
        if meter is not None and meter.exhausted():
            with lock:
                if not stopped:
                    stopped = True
                    print(f"  STOPPING: token budget spent ({meter.report()}).")
            return

        bucket.acquire(EST_JUDGE_FIXED + EST_JUDGE_PER_PAIR * len(group))
        results = judge_batch(client, group, meter=meter)

        # A batch call that comes back entirely empty usually means one
        # malformed response, not a real problem with the pairs. Falling back
        # to judging them one at a time costs a few hundred extra tokens on a
        # rare failure, and is much better than silently leaving pairs
        # ungraded - which is what used to happen, and which shows up in the
        # statistics as an unexplained "not assessed".
        if all(s is None for s in results):
            for position, pair in enumerate(group):
                bucket.acquire(EST_JUDGE_FIXED)
                results[position] = judge_pair(client, pair, meter=meter)
            recovered = sum(1 for s in results if s is not None)
            if recovered:
                print(f"      batch failed; recovered {recovered}/{len(group)} singly")

        per_group[index] = results

        with lock:
            # Written as soon as the call returns, under the lock so workers
            # cannot interleave half-lines. This is the stage most likely to
            # meet the daily token wall, so anything already graded must
            # survive the run that dies.
            for pair, scored in zip(group, results):
                if scored is not None:
                    checkpoint.append(
                        JUDGE_CHECKPOINT_FILE, judge_fp, pair["question"], scored
                    )
            judged_count += 1
            done = sum(
                1 for g in per_group for s in g if s is not None
            )
            if progress_cb is not None:
                progress_cb(
                    judged_count, len(group_list), f"{done} of {len(pairs)} pairs judged"
                )
            if judged_count % 10 == 0 or judged_count == len(group_list):
                elapsed = time.time() - started
                remaining = (elapsed / judged_count) * (len(group_list) - judged_count)
                print(
                    f"  [{judged_count}/{len(group_list)} calls, {done}/{len(pairs)} "
                    f"pairs] {elapsed / 60:.1f}m elapsed, ~{remaining / 60:.1f}m left"
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda item: judge_group(*item), enumerate(group_list)))

    # Now the abstention pairs, one call each, same concurrency and pacing.
    abstain_results: list[dict | None] = [None] * len(pending_abstentions)

    def check_one(index: int, pair: dict) -> None:
        if stopped or (meter is not None and meter.exhausted()):
            return
        bucket.acquire(EST_JUDGE_FIXED)
        result = check_unanswerable(client, pair, meter=meter)
        abstain_results[index] = result
        if result.get("passed") is not None:
            with lock:
                checkpoint.append(
                    JUDGE_CHECKPOINT_FILE, judge_fp, pair["question"], result
                )

    if pending_abstentions:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda item: check_one(*item), enumerate(pending_abstentions)))

    # Anything left unscored - because the budget ran out - stays None, which
    # is recorded below as "not assessed" rather than counted as a quality
    # rejection. That distinction is what keeps the pass rate honest.
    # Looked up by object identity so the merge below can walk `pairs` in the
    # order they were generated. Order matters downstream: deduplication keeps
    # the FIRST of each duplicate group, so a reshuffle here would silently
    # change which version of a near-duplicate survives.
    score_of: dict[int, dict | None] = {}
    check_of: dict[int, dict | None] = {}

    # Cached results first, so a resumed run treats them exactly like fresh
    # ones from here on.
    for p in pairs:
        record = cached_scores.get(p["question"])
        if record is None:
            continue
        if p.get("answerable", True):
            score_of[id(p)] = record["payload"]
        else:
            check_of[id(p)] = record["payload"]

    for group, results in zip(group_list, per_group):
        for pair, result in zip(group, results):
            score_of[id(pair)] = result

    for pair, result in zip(pending_abstentions, abstain_results):
        check_of[id(pair)] = result

    abstain_kept = 0
    abstain_rejected = 0

    # One pass over `pairs` in generation order, handling both kinds, so that
    # `kept` comes out in exactly the input order. Two passes would group all
    # the abstention pairs at the front, and since deduplication keeps the
    # FIRST of a duplicate group, a refusal could then beat a good answerable
    # pair it happened to collide with.
    for pair in pairs:
        if not pair.get("answerable", True):
            result = check_of.get(id(pair)) or {
                "passed": None,
                "check": None,
                "note": "judge call failed",
            }
            graded = dict(pair)
            # No quality_score, deliberately: these were never graded 0-6, and
            # a number here would end up in the dataset average.
            graded["quality_score"] = None
            graded["scores"] = None
            graded["unanswerable_check"] = result["check"]

            if result["passed"] is None:
                judge_failed += 1
                graded["reject_reason"] = "judge call failed - not assessed"
                rejected.append(graded)
            elif result["passed"]:
                graded["reject_reason"] = None
                kept.append(graded)
                abstain_kept += 1
            else:
                graded["reject_reason"] = f"abstention check failed: {result['note']}"
                rejected.append(graded)
                abstain_rejected += 1
            continue

        scores = score_of.get(id(pair))
        if scores is None:
            judge_failed += 1
            graded = dict(pair)
            graded.update(
                {
                    "quality_score": None,
                    "scores": None,
                    "reject_reason": "judge call failed - not assessed",
                }
            )
            rejected.append(graded)
        else:
            graded = dict(pair)
            graded["quality_score"] = scores["total"]
            graded["scores"] = {
                k: scores[k] for k in ("groundedness", "specificity", "completeness")
            }
            sub = graded["scores"]
            fails_groundedness = sub["groundedness"] < floors.get("groundedness", 0)
            failed_floors = [
                f"{name} {sub.get(name, 0)}/{floor}"
                for name, floor in floors.items()
                if sub.get(name, 0) < floor
            ]
            if keep_decision(sub, min_score, floors):
                graded["reject_reason"] = None
                kept.append(graded)
            else:
                if fails_groundedness:
                    ungrounded += 1
                    # Say plainly why this one died, even at a passing total -
                    # this is the line the dashboard shows as evidence.
                    graded["reject_reason"] = (
                        f"UNGROUNDED (groundedness {sub['groundedness']}/2): "
                        + (scores["reason"] or "claims not traceable to passage")
                    )
                elif failed_floors:
                    graded["reject_reason"] = (
                        f"below floor on {', '.join(failed_floors)}"
                        + (f": {scores['reason']}" if scores["reason"] else "")
                    )
                else:
                    graded["reject_reason"] = (
                        scores["reason"]
                        or f"scored {scores['total']}/6, below threshold {min_score}"
                    )
                rejected.append(graded)

    # No sleep or progress print here: every API call already happened in the
    # batching loop above, so this pass is pure CPU and runs instantly.
    print(f"  Sorted {len(pairs)} pairs: {len(kept)} kept, {len(rejected)} rejected")

    if write_rejected:
        ensure_output_dir()
        with open(REJECTED_FILE, "w", encoding="utf-8") as fh:
            for r in rejected:
                out = {k: v for k, v in r.items() if k != "chunk_text"}
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(rejected)} rejected pairs -> {REJECTED_FILE.name}")

        # Every graded pair, kept and rejected, WITH chunk_text. The threshold
        # and dedup experiments re-filter this file offline; the judge
        # reliability study needs the passage to score pairs by hand.
        with open(GRADED_FILE, "w", encoding="utf-8") as fh:
            for p in kept + rejected:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Wrote {len(kept) + len(rejected)} graded pairs -> {GRADED_FILE.name}")

    assessed = len(pairs) - judge_failed
    pass_rate = len(kept) / assessed * 100 if assessed else 0.0
    # The headline pass rate mixes two different tests, so the answerable-only
    # rate is reported alongside it - that is the one the rubric describes and
    # the one the report should quote.
    answerable_assessed = sum(
        1
        for p in kept + rejected
        if p.get("answerable", True) and p.get("quality_score") is not None
    )
    answerable_kept = sum(1 for p in kept if p.get("answerable", True))
    scored = [k["quality_score"] for k in kept if k["quality_score"] is not None]

    print(f"\nValidation complete:")
    print(f"  Kept     : {len(kept)}")
    print(f"  Rejected : {len(rejected)}")
    if judge_failed:
        print(f"    (of which {judge_failed} were judge failures, not quality rejections)")
    if ungrounded:
        print(
            f"    (of which {ungrounded} were vetoed for groundedness 0 - "
            f"ungrounded claims, some at a passing total)"
        )
    print(f"  Pass rate: {pass_rate:.1f}% of {assessed} assessed pairs")
    if abstentions:
        answerable_rate = (
            answerable_kept / answerable_assessed * 100 if answerable_assessed else 0.0
        )
        print(
            f"    answerable only: {answerable_rate:.1f}% "
            f"({answerable_kept} of {answerable_assessed} graded on the rubric)"
        )
        print(
            f"    abstention examples: {abstain_kept} verified, "
            f"{abstain_rejected} failed the check"
        )
    if scored:
        print(f"  Avg score of kept: {sum(scored) / len(scored):.2f}/6")
        if abstentions:
            print("    (answerable pairs only - abstention pairs carry no score)")

    graded_all = [p for p in kept + rejected if p.get("quality_score") is not None]
    if graded_all:
        dist = Counter(p["quality_score"] for p in graded_all)
        print("  Score distribution:", {s: dist[s] for s in sorted(dist)})

    return kept, rejected


if __name__ == "__main__":
    use_utf8_console()
    from generate import generate_pairs
    from ingest import chunk_documents, load_pdfs

    print("=" * 70)
    print("STEP 3: VALIDATE - smoke test")
    print("=" * 70)

    chunks = chunk_documents(load_pdfs())
    start = len(chunks) // 2
    pairs = generate_pairs(chunks[start : start + 4], questions_per_chunk=3)

    kept, rejected = validate_pairs(pairs, write_rejected=False)

    print("\n--- Kept examples ---")
    for p in kept[:3]:
        print(f"\n  [{p['quality_score']}/6] {p['scores']}")
        print(f"  Q: {p['question']}")
        print(f"  A: {p['answer'][:150]}")

    print("\n--- Rejected examples ---")
    for p in rejected[:3]:
        print(f"\n  [{p['quality_score']}/6] {p['scores']}")
        print(f"  Q: {p['question']}")
        print(f"  reason: {p['reject_reason']}")
