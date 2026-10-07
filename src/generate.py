"""
Step 2: Generate Q&A pairs from each chunk with the Groq API.

One API call per chunk returns a JSON array of pairs. Each pair carries its
source document, page number and question type so the dataset stays traceable,
plus `chunk_text` - the passage it came from - which the judge in validate.py
needs and export.py strips before writing the final dataset.
"""

from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from pathlib import Path

from groq import Groq

try:
    import checkpoint
    from config import (
        ABSTENTION_ANSWER,
        GEN_CHECKPOINT_FILE,
        GEN_MAX_TOKENS,
        GEN_TEMPERATURE,
        GROQ_MODEL,
        REASONING_EFFORT,
        REQUEST_DELAY,
        RUN_SEED,
        UNANSWERABLE_RATIO,
        get_groq_key,
        model_kwargs,
        use_utf8_console,
    )
except ImportError:  # running as `python src/generate.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import checkpoint
    from config import (
        ABSTENTION_ANSWER,
        GEN_CHECKPOINT_FILE,
        GEN_MAX_TOKENS,
        GEN_TEMPERATURE,
        GROQ_MODEL,
        REASONING_EFFORT,
        REQUEST_DELAY,
        RUN_SEED,
        UNANSWERABLE_RATIO,
        get_groq_key,
        model_kwargs,
        use_utf8_console,
    )

# Bumped whenever PROMPT_TEMPLATE or UNANSWERABLE_PROMPT changes, so a
# checkpoint written by an older prompt is never reused for a newer one.
PROMPT_VERSION = "gen-v1"


def chunk_key(chunk) -> str:
    """
    A stable identifier for one chunk, for checkpointing.

    Source file, page and character offset - not the text itself and not the
    list position, because a run over a different subset of the corpus must
    still recognise a chunk it has already paid to generate from.
    """
    md = getattr(chunk, "metadata", {}) or {}
    return "{}|{}|{}".format(
        md.get("source_file", "?"), md.get("page", 0), md.get("start_index", 0)
    )


# The three types the brief asks for. Mixing them stops the dataset becoming
# 300 variations of "What is X?", which is the default failure mode of naive
# synthetic generation and shows up immediately in the diversity statistic.
QUESTION_TYPES = ("factual", "definitional", "procedural")

MIN_QUESTION_CHARS = 15
MIN_ANSWER_CHARS = 25

# Compressed deliberately. The verbose original cost ~373 tokens of template on
# every one of hundreds of calls, and the whole pipeline is token-rate bound
# (8,000/min), so template size translates directly into wall-clock time.
# Every rule from the long version survives here in shorter form - the
# constraints are the same, only the prose is gone.
PROMPT_TEMPLATE = """Write exactly {n} Q&A pairs from the PASSAGE below.

RULES:
1. Answers must come only from the PASSAGE. Add nothing, infer nothing.
2. Fewer pairs is fine if the passage is thin.
3. Questions must stand alone - name the subject, never say "this passage",
   "the text", "the document" or "the above".
4. No questions about the document itself (ISBN, page/figure numbers, authors,
   copyright, contact details, references).
5. Answers: 1-3 sentences, self-contained.
6. One of each type where possible - factual (a specific fact/number/
   threshold), definitional (what a term means), procedural (how/when to act).

Output ONLY a JSON array, no fences:
[{{"question":"...","answer":"...","question_type":"factual"}}]

PASSAGE:
\"\"\"
{passage}
\"\"\"
"""

# Abstention examples. A dataset of nothing but answerable questions teaches a
# model that every question has an answer in the document, so when it does not
# know it invents something. Including questions the passage genuinely cannot
# answer, paired with a fixed refusal, teaches it to say so instead.
#
# The hard part is making the question plausible: it has to sound like
# something a reader would really ask about this material, not a non-sequitur,
# or the model learns to refuse anything unfamiliar rather than anything
# unsupported. Hence the insistence on staying on topic.
UNANSWERABLE_PROMPT = """Read the PASSAGE below. Write ONE question a reader
might plausibly ask about this topic, but which the PASSAGE does NOT answer.

RULES:
1. Stay on the passage's topic, so the question sounds natural for this
   material.
2. The PASSAGE must contain nothing that answers it, not even partially. Good
   choices: a dose, duration, age group, side effect, cost or comparison the
   passage never mentions.
3. The question must stand alone - name the subject, never say "this passage",
   "the text" or "the document".
4. Nothing about the document itself (authors, dates, page numbers).

Output ONLY a JSON object, no fences:
{{"question":"..."}}

PASSAGE:
\"\"\"
{passage}
\"\"\"
"""

# Questions containing these refer to the passage rather than the subject, so
# they are useless as standalone training data. The prompt forbids them; this
# catches the cases where the model does it anyway.
_CONTEXT_DEPENDENT = re.compile(
    r"\b(this|the)\s+(passage|text|document|excerpt|section|article|guideline\s+text)\b"
    r"|\baccording to the (passage|text|document|above)\b"
    r"|\bthe above\b|\bmentioned above\b",
    re.I,
)

# Questions about the publication rather than its clinical content. The
# boilerplate filter in ingest.py removes most of the source chunks; this is
# the second line of defence for the ones that slip through, because a
# question about map border lines or copyright is worthless training data
# however well formed it is.
_ABOUT_THE_DOCUMENT = re.compile(
    r"\b(disclaimer|copyright|ISBN|licence|license|publisher|book-?orders)\b"
    r"|\b(dotted|dashed)\b.{0,20}\blines?\b"
    r"|\bborder lines\b|\blegal status\b|\bfrontiers\b"
    r"|\binitial capital letters\b|\bthird-party\b"
    r"|\bfigure \d|\btable \d|\bpage \d|\breference list\b"
    # Document navigation. A run produced "On which page does the
    # Abbreviations section start according to the contents list?" - the
    # earlier `page \d` pattern missed it because no digit follows "page".
    r"|\b(which|what) page\b|\bpage number\b|\bcontents list\b"
    r"|\btable of contents\b|\bsection (start|begin)\b"
    r"|\blisted in the contents\b|\bwhich section\b",
    re.I,
)


def retry_wait(message: str, attempt: int) -> float:
    """
    Seconds to wait after a 429, read from Groq's own message when possible.

    Groq replies "Please try again in 4m24.383s". Honouring that beats a fixed
    backoff: too short burns requests against the daily cap, too long wastes
    wall-clock time.
    """
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
    if match:
        minutes = float(match.group(1) or 0)
        seconds = float(match.group(2))
        return min(minutes * 60 + seconds + 1.0, 300.0)
    return min(20.0 * (attempt + 1), 120.0)


def strip_fences(text: str) -> str:
    """
    Remove ```json ... ``` wrappers.

    Instruction-tuned models add markdown fences even when told not to, and
    json.loads dies on the backticks. Cheapest possible fix, applied always.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_array(text: str) -> list | None:
    """
    Parse a JSON array from a model response, tolerating minor sloppiness.

    Tries the whole string first, then falls back to the outermost [...] span
    for responses that wrap the array in a sentence of commentary.
    """
    text = strip_fences(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    # Some responses come back as {"pairs": [...]} despite the instruction.
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                parsed = value
                break
    return parsed if isinstance(parsed, list) else None


def extract_json_object(text: str) -> dict | None:
    """
    Parse a single JSON object from a model response.

    The abstention prompt asks for one object, not an array, so it needs its
    own parser; same tolerance for fences and surrounding commentary.
    """
    text = strip_fences(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def clean_pairs(raw: list, chunk) -> list[dict]:
    """Validate the model's output and attach provenance to each good pair."""
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        qtype = str(item.get("question_type", "")).strip().lower()

        # Too short to be a real Q&A; usually a truncated response. The
        # answer floor is higher than the question's because a 15-character
        # answer ("Yes", "At six weeks") is not self-contained training data,
        # whereas a 15-character question can be perfectly good.
        if len(question) < MIN_QUESTION_CHARS or len(answer) < MIN_ANSWER_CHARS:
            continue
        # Rule 3 enforcement - drop questions that lean on the passage.
        if _CONTEXT_DEPENDENT.search(question):
            continue
        # Rule 4 enforcement - drop questions about the document itself.
        if _ABOUT_THE_DOCUMENT.search(question):
            continue
        if qtype not in QUESTION_TYPES:
            qtype = "factual"

        out.append(
            {
                "question": question,
                "answer": answer,
                "question_type": qtype,
                # Every pair carries this so the judge knows which rubric to
                # apply and the dashboard can separate the two kinds.
                "answerable": True,
                "source": chunk.metadata.get("source_file", "unknown"),
                # PyPDF pages are 0-indexed; +1 matches what a human sees.
                "page": int(chunk.metadata.get("page", 0)) + 1,
                # Kept for the judge, stripped by export.py.
                "chunk_text": chunk.page_content,
            }
        )
    return out


def generate_for_chunk(
    client: Groq, chunk, n: int = 3, retries: int = 1, meter=None
) -> list[dict]:
    """
    Generate pairs for one chunk. Returns [] if the chunk cannot be parsed.

    Retries once on a parse failure (the brief's rule) and backs off on rate
    limits, because one bad response must not kill a 20-minute run.
    """
    prompt = PROMPT_TEMPLATE.format(n=n, passage=chunk.page_content)

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=GEN_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
                **model_kwargs(GROQ_MODEL, GEN_MAX_TOKENS),
            )
            if meter is not None:
                meter.record(response.usage, "generate")
            parsed = extract_json_array(response.choices[0].message.content)
            if parsed is not None:
                return clean_pairs(parsed, chunk)
        except Exception as exc:
            name = type(exc).__name__
            message = str(exc)
            # 429 / rate limit - honour the wait Groq asks for.
            if "RateLimit" in name or "429" in message:
                wait = retry_wait(message, attempt)
                print(f"      rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"      API error ({name}): {message[:110]}")

        if attempt < retries:
            time.sleep(2.0)

    return []


def unanswerable_chunk_ids(
    count: int, ratio: float = UNANSWERABLE_RATIO, seed: int = RUN_SEED
) -> set[int]:
    """
    Choose which chunk positions also get an abstention example.

    Seeded from config.RUN_SEED, not left to chance: a dataset whose contents
    shift between runs cannot be reported on, and the threshold experiments
    compare runs against each other. `ceil` rather than `round` so that even a
    small test run gets at least one abstention example - at ratio 0.10 a
    20-chunk run would otherwise be a coin toss between 2 and 0.
    """
    if ratio <= 0 or count <= 0:
        return set()
    k = min(count, math.ceil(count * ratio))
    return set(random.Random(seed).sample(range(count), k))


def generate_unanswerable(client: Groq, chunk, retries: int = 1, meter=None):
    """
    Ask for one on-topic question this passage cannot answer.

    The answer is not generated. It is the fixed ABSTENTION_ANSWER from
    config, identical on every abstention example, so the model learns one
    refusal phrase rather than fifty paraphrases of "I don't know".

    Returns None rather than raising if the model will not cooperate: a
    missing abstention example costs the dataset one row, and the alternative
    - a bad one - costs it credibility.
    """
    prompt = UNANSWERABLE_PROMPT.format(passage=chunk.page_content)

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=GEN_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
                **model_kwargs(GROQ_MODEL, GEN_MAX_TOKENS),
            )
            if meter is not None:
                meter.record(response.usage, "generate")
            parsed = extract_json_object(response.choices[0].message.content)
            if parsed:
                question = str(parsed.get("question", "")).strip()
                # Same three gates as a normal pair: long enough to be real,
                # standing on its own, and not about the publication.
                if (
                    len(question) >= MIN_QUESTION_CHARS
                    and not _CONTEXT_DEPENDENT.search(question)
                    and not _ABOUT_THE_DOCUMENT.search(question)
                ):
                    return {
                        "question": question,
                        "answer": ABSTENTION_ANSWER,
                        "question_type": "unanswerable",
                        "answerable": False,
                        "source": chunk.metadata.get("source_file", "unknown"),
                        "page": int(chunk.metadata.get("page", 0)) + 1,
                        "chunk_text": chunk.page_content,
                    }
        except Exception as exc:
            name = type(exc).__name__
            message = str(exc)
            if "RateLimit" in name or "429" in message:
                wait = retry_wait(message, attempt)
                print(f"      rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"      abstention error ({name}): {message[:110]}")

        if attempt < retries:
            time.sleep(2.0)

    return None


def generate_pairs(
    chunks: list,
    questions_per_chunk: int = 3,
    delay: float = REQUEST_DELAY,
    meter=None,
    progress_cb=None,
    bucket=None,
    workers: int | None = None,
    resume: bool = True,
) -> list[dict]:
    """
    Run generation across all chunks, several requests at a time.

    Pacing is by token budget, not by a fixed sleep. The ceiling that actually
    binds is 8,000 tokens per minute, so a shared TokenBucket meters spending
    and the workers simply wait their turn on it. `delay` is accepted for
    backwards compatibility and ignored - a fixed sleep was both too slow when
    the budget had headroom and too fast when it did not.

    Concurrency does not raise the ceiling; it stops request latency (~3s per
    call) from leaving the token budget idle while one worker waits for a
    reply. Results are reassembled in input order so deduplication's
    first-wins rule stays deterministic.
    """
    from concurrent.futures import ThreadPoolExecutor

    from budget import (
        EST_TOKENS_PER_GENERATION,
        EST_TOKENS_PER_UNANSWERABLE,
        MAX_CONCURRENCY,
        TokenBucket,
    )

    client = Groq(api_key=get_groq_key())
    bucket = bucket or TokenBucket()
    workers = workers or MAX_CONCURRENCY

    # Work already paid for in an earlier run of this exact configuration.
    # The fingerprint covers everything that would change the output, so a
    # different model, temperature or prompt silently starts from scratch
    # rather than blending two generators' output into one dataset.
    fp = checkpoint.fingerprint(
        stage="generate",
        prompt=PROMPT_VERSION,
        model=GROQ_MODEL,
        temperature=GEN_TEMPERATURE,
        max_tokens=GEN_MAX_TOKENS,
        reasoning=REASONING_EFFORT,
        questions_per_chunk=questions_per_chunk,
        unanswerable_ratio=UNANSWERABLE_RATIO,
        abstention_answer=ABSTENTION_ANSWER,
        seed=RUN_SEED,
    )
    done = checkpoint.load(GEN_CHECKPOINT_FILE, fp) if resume else {}

    results: list[list[dict]] = [[] for _ in chunks]
    abstain_at = unanswerable_chunk_ids(len(chunks))
    reused = 0
    abstain_made = 0
    failed = 0
    completed = 0
    stopped = False
    started = time.time()
    lock = threading.Lock()

    print(f"Generating from {len(chunks)} chunks using {GROQ_MODEL}")
    print(
        f"  temperature={GEN_TEMPERATURE}, {questions_per_chunk} pairs/chunk, "
        f"{workers} workers, paced to {bucket.capacity:.0f} tokens/min"
    )
    if abstain_at:
        print(
            f"  plus 1 unanswerable question from {len(abstain_at)} of them "
            f"({UNANSWERABLE_RATIO:.0%}) to teach abstention"
        )
    if done:
        already = sum(1 for c in chunks if chunk_key(c) in done)
        if already:
            print(
                f"  resuming: {already} of {len(chunks)} chunks already "
                f"generated in an earlier run, no API calls needed for them"
            )

    def work(index: int, chunk) -> None:
        nonlocal failed, completed, stopped, abstain_made, reused

        # Leave roughly half the budget for judging: pairs that can never be
        # graded are worse than pairs that were never generated.
        if meter is not None and meter.exhausted(reserve=meter.budget * 0.45):
            with lock:
                if not stopped:
                    stopped = True
                    print(
                        f"  STOPPING: token budget nearly spent "
                        f"({meter.report()}). Reserving the rest for judging."
                    )
            return
        if stopped:
            return

        key = chunk_key(chunk)
        cached = done.get(key)
        if cached is not None:
            with lock:
                completed += 1
                reused += 1
                results[index] = cached["payload"]
                if progress_cb is not None:
                    progress_cb(
                        completed,
                        len(chunks),
                        f"{sum(len(r) for r in results)} pairs from "
                        f"{completed} chunks",
                    )
            return

        bucket.acquire(EST_TOKENS_PER_GENERATION)
        got = generate_for_chunk(client, chunk, questions_per_chunk, meter=meter)

        # A second, different call on the chunks picked for abstention. Kept
        # separate from the one above so that a chunk whose normal generation
        # failed can still contribute its abstention example, and vice versa.
        abstain = None
        if index in abstain_at and not stopped:
            bucket.acquire(EST_TOKENS_PER_UNANSWERABLE)
            abstain = generate_unanswerable(client, chunk, meter=meter)

        with lock:
            completed += 1
            if got:
                results[index] = got
            else:
                failed += 1
                print(f"  [{completed}/{len(chunks)}] FAILED - skipped a chunk")
            if abstain is not None:
                results[index] = results[index] + [abstain]
                abstain_made += 1

            # Written inside the lock, so concurrent workers cannot interleave
            # half-lines into the file.
            if results[index]:
                checkpoint.append(GEN_CHECKPOINT_FILE, fp, key, results[index])

            total_pairs = sum(len(r) for r in results)
            if progress_cb is not None:
                progress_cb(
                    completed, len(chunks), f"{total_pairs} pairs from {completed} chunks"
                )
            if completed % 10 == 0 or completed == len(chunks):
                elapsed = time.time() - started
                remaining = (elapsed / completed) * (len(chunks) - completed)
                print(
                    f"  [{completed}/{len(chunks)}] {total_pairs} pairs, "
                    f"{failed} failed, {elapsed / 60:.1f}m elapsed, "
                    f"~{remaining / 60:.1f}m left"
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda pair: work(*pair), enumerate(chunks)))

    pairs = [p for chunk_pairs in results for p in chunk_pairs]

    answerable = sum(1 for p in pairs if p.get("answerable", True))
    print(f"\nGenerated {len(pairs)} raw pairs from {len(chunks)} chunks")
    print(f"  Answerable: {answerable}   Unanswerable: {len(pairs) - answerable}")
    if abstain_at:
        print(
            f"  Abstention examples: {abstain_made} of {len(abstain_at)} chunks asked"
        )
    if reused:
        print(f"  Reused from checkpoint: {reused} chunk(s) - no API cost")
    print(f"  Failed chunks: {failed} ({failed / max(len(chunks),1) * 100:.1f}%)")
    print(f"  Elapsed: {(time.time() - started) / 60:.1f} min")
    if pairs:
        from collections import Counter

        dist = Counter(p["question_type"] for p in pairs)
        print("  Question types:", dict(dist))
    return pairs


if __name__ == "__main__":
    use_utf8_console()
    from ingest import chunk_documents, load_pdfs

    print("=" * 70)
    print("STEP 2: GENERATE - smoke test on 5 chunks")
    print("=" * 70)

    chunks = chunk_documents(load_pdfs())

    # Sample from the middle of the corpus: the first chunks of any guideline
    # are title pages and contents, which are not representative.
    start = len(chunks) // 2
    sample = chunks[start : start + 5]

    pairs = generate_pairs(sample, questions_per_chunk=3)

    print("\n--- Sample output ---")
    for p in pairs[:6]:
        print(f"\n[{p['question_type']}] {p['source']} p.{p['page']}")
        print(f"  Q: {p['question']}")
        print(f"  A: {p['answer']}")
