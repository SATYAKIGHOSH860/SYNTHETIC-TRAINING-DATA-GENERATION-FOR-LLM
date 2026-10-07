"""
Step 4: Semantic deduplication with sentence embeddings.

Chunks overlap by 80 characters and neighbouring passages restate the same
recommendation, so the generator inevitably produces the same question in
different words. String matching cannot see this:

    "What is the CD4 threshold for advanced HIV disease?"
    "At what CD4 count is a patient considered to have advanced HIV disease?"

share few exact tokens but mean the same thing. Embeddings put them at ~0.9
cosine similarity, so they are caught. That difference is the whole point of
this step.

Every removal is recorded with the question it duplicated and the similarity
score, so the dashboard can show the pair side by side as evidence.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from config import DUPLICATE_FILE, ensure_output_dir, use_utf8_console
except ImportError:  # running as `python src/deduplicate.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import DUPLICATE_FILE, ensure_output_dir, use_utf8_console

# Why all-MiniLM-L6-v2: 90 MB, runs in seconds on CPU (this project has no
# GPU), and is the standard baseline for sentence similarity. A larger model
# would shift scores by a couple of points and cost minutes per run.
EMBED_MODEL = "all-MiniLM-L6-v2"

# Why 0.85: tuned on this corpus. Below ~0.80 genuinely different questions
# about the same drug start colliding; above ~0.90 obvious rewordings survive.
# The dedup-sensitivity experiment reports survivor counts at 0.75/0.85/0.95
# instead of just asserting this number.
# Raised from 0.85 after measuring what 0.85 actually removed on this corpus.
# At 0.85, embedding "At what age is the first dose given?" against "When is
# the second dose given?" scored 0.896 and deleted one of them - two different
# facts merged into one. 0.90 keeps that pair apart while still catching real
# rewordings.
SIMILARITY_THRESHOLD = 0.90

# Embed the question AND its answer, not the question alone. Two questions can
# be worded almost identically and still be about different populations, doses
# or time frames; the answer is where that difference lives, so leaving it out
# throws away the information that tells them apart.
EMBED_QUESTION_AND_ANSWER = True

# A second, non-semantic guard. Embeddings barely distinguish "6 months" from
# "9 months", so two answers stating different numbers are never treated as
# duplicates however similar the wording. Only applies when both answers
# contain numbers - prose answers fall back to similarity alone.
REQUIRE_SAME_NUMBERS = True

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def dedup_text(pair: dict) -> str:
    """What gets embedded for one pair."""
    if EMBED_QUESTION_AND_ANSWER:
        return f"{pair['question']} {pair['answer']}"
    return pair["question"]


def answer_numbers(answer: str) -> frozenset[str]:
    """
    Numbers stated in an answer, normalised ("1,500" and "1500" are the same).

    Used as a veto, not a score: see REQUIRE_SAME_NUMBERS.
    """
    return frozenset(n.replace(",", "") for n in _NUMBER.findall(answer or ""))


def lexical_overlap(a: str, b: str) -> float:
    """
    How similar two questions look as plain strings, 0-1.

    Recorded next to the cosine score on every removed duplicate because it is
    the evidence for using embeddings at all: a pair at 0.93 cosine and 0.40
    lexical is one that string matching would have missed completely.
    """
    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def embed_questions(questions: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    """
    Encode questions to unit-length vectors.

    normalize_embeddings=True makes every vector length 1, so a plain dot
    product IS cosine similarity - no separate normalisation step, and the
    matrix multiply below stays a single fast operation.
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading {model_name} (downloads ~90 MB on first run)...")
    model = SentenceTransformer(model_name, device="cpu")

    print(f"  Encoding {len(questions)} questions on CPU...")
    return model.encode(
        questions,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def question_diversity(pairs: list[dict], top: int = 12) -> dict:
    """
    Distribution of question opening words.

    Why it matters: a dataset can pass dedup and still be monotonous - 80%
    starting "What is" means the model trained on it never learns to handle
    "How should" or "When must". This is the cheapest possible check for that
    failure, and it is a number worth reporting.
    """
    firsts = Counter()
    for p in pairs:
        words = re.findall(r"[A-Za-z]+", p.get("question", ""))
        if words:
            # Two words carries far more signal than one: "What is" vs
            # "What dose" are different question shapes.
            firsts[" ".join(w.lower() for w in words[:2])] += 1

    total = sum(firsts.values()) or 1
    return {
        "unique_openers": len(firsts),
        "top_openers": [
            {"opener": k, "count": v, "pct": round(v / total * 100, 1)}
            for k, v in firsts.most_common(top)
        ],
        "most_common_pct": round(firsts.most_common(1)[0][1] / total * 100, 1)
        if firsts
        else 0.0,
    }


def deduplicate_pairs(
    pairs: list[dict],
    threshold: float = SIMILARITY_THRESHOLD,
    write_duplicates: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Remove near-duplicate questions. Returns (unique, duplicates).

    Greedy first-wins: walk the list in order and keep a pair only if it is
    below `threshold` against everything kept so far. The survivor is the
    earliest occurrence, which keeps the dataset stable - re-running does not
    reshuffle which version of a question survives.
    """
    if not pairs:
        return [], []

    embeddings = embed_questions([dedup_text(p) for p in pairs])
    numbers = (
        [answer_numbers(p.get("answer", "")) for p in pairs]
        if REQUIRE_SAME_NUMBERS
        else None
    )

    kept_idx: list[int] = []
    unique: list[dict] = []
    duplicates: list[dict] = []
    # Rows of `embeddings` for the pairs kept so far, as one array so each
    # comparison is a single matrix-vector product rather than a Python loop.
    kept_matrix = np.zeros((0, embeddings.shape[1]), dtype=embeddings.dtype)

    print(
        f"  Comparing at cosine similarity >= {threshold} "
        f"({'question + answer' if EMBED_QUESTION_AND_ANSWER else 'question only'}"
        f"{', number guard on' if REQUIRE_SAME_NUMBERS else ''})..."
    )
    for i, pair in enumerate(pairs):
        if kept_matrix.shape[0]:
            # Unit vectors, so dot product == cosine similarity.
            sims = kept_matrix @ embeddings[i]
            # Veto before picking the best match, not after: a pair whose
            # answer states different numbers is not a duplicate at any
            # similarity, so it must not be allowed to win the argmax.
            if numbers is not None and numbers[i]:
                for position, kept_index in enumerate(kept_idx):
                    other = numbers[kept_index]
                    if other and other != numbers[i]:
                        sims[position] = -1.0
            best = int(np.argmax(sims))
            best_sim = float(sims[best])
        else:
            best, best_sim = -1, 0.0

        if best_sim >= threshold:
            original = pairs[kept_idx[best]]
            duplicates.append(
                {
                    "removed_question": pair["question"],
                    "removed_answer": pair["answer"],
                    "duplicate_of_question": original["question"],
                    "duplicate_of_answer": original["answer"],
                    "similarity": round(best_sim, 4),
                    # The evidence for embeddings over string matching: a high
                    # cosine beside a low lexical score is a duplicate that
                    # `==` or fuzzy string matching would never have caught.
                    "lexical_overlap": round(
                        lexical_overlap(pair["question"], original["question"]), 4
                    ),
                    "source": pair.get("source"),
                    "page": pair.get("page"),
                    "question_type": pair.get("question_type"),
                    "quality_score": pair.get("quality_score"),
                }
            )
        else:
            kept_idx.append(i)
            unique.append(pair)
            kept_matrix = np.vstack([kept_matrix, embeddings[i][None, :]])

    removed = len(duplicates)
    pct = removed / max(len(pairs), 1) * 100
    print(f"\nDeduplication complete:")
    print(f"  In       : {len(pairs)}")
    print(f"  Unique   : {len(unique)}")
    print(f"  Removed  : {removed} ({pct:.1f}%)")

    if duplicates:
        sims = [d["similarity"] for d in duplicates]
        lex = [d["lexical_overlap"] for d in duplicates]
        print(f"  Similarity of removed: min {min(sims):.3f}, max {max(sims):.3f}")
        # The headline claim of this stage, stated as a measurement.
        caught = sum(1 for value in lex if value < 0.6)
        print(
            f"  Lexical overlap of removed: median {sorted(lex)[len(lex) // 2]:.2f}"
            f"  ({caught} of {len(lex)} below 0.60 - string matching would have "
            f"missed them)"
        )

    if write_duplicates:
        ensure_output_dir()
        with open(DUPLICATE_FILE, "w", encoding="utf-8") as fh:
            for d in duplicates:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  Wrote {removed} duplicate records -> {DUPLICATE_FILE.name}")

    div = question_diversity(unique)
    print(f"\n  Question diversity: {div['unique_openers']} distinct openers")
    print(f"  Most common opener is {div['most_common_pct']}% of the set")
    for row in div["top_openers"][:6]:
        print(f"    {row['opener']:<22} {row['count']:>5}  ({row['pct']}%)")

    return unique, duplicates


if __name__ == "__main__":
    use_utf8_console()

    print("=" * 70)
    print("STEP 4: DEDUPLICATE - self-test on planted near-duplicates")
    print("=" * 70)

    demo = [
        "What is the CD4 threshold for advanced HIV disease?",
        "At what CD4 count is a patient considered to have advanced HIV disease?",
        "How should cryptococcal meningitis be treated in adults living with HIV?",
        "What is the recommended treatment for cryptococcal meningitis in HIV-positive adults?",
        "Which HPV genotypes are targeted by DNA testing?",
    ]
    pairs = [
        {"question": q, "answer": f"Answer {i}", "source": "demo.pdf", "page": i}
        for i, q in enumerate(demo)
    ]

    unique, dupes = deduplicate_pairs(pairs, write_duplicates=False)

    print("\n--- Duplicates caught (string matching would miss these) ---")
    for d in dupes:
        print(f"\n  similarity {d['similarity']}")
        print(f"    removed : {d['removed_question']}")
        print(f"    kept    : {d['duplicate_of_question']}")

    print("\n--- Survivors ---")
    for u in unique:
        print(f"  {u['question']}")
