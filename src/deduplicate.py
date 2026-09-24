"""Semantic deduplication with sentence embeddings.

Why not string matching: "What is the intensive phase duration?" and "How
long does the intensive phase last?" share almost no words (string similarity
around 40%), yet their embeddings sit near 0.90 cosine similarity. Only
embeddings catch rewordings like these.

What counts as a duplicate: a pair that teaches the same fact. By default the
question AND answer are embedded (``deduplicate.embed``), and two pairs whose
answers state different numbers are never merged (``require_same_numbers``).
Both choices come from measurement on the WHO guideline data, where
question-only similarity merged distinct facts (for example, fibre intake for
children aged 2-5 years vs adults, or the strength vs the evidence quality of
one recommendation). The question-only behaviour of the original spec remains
available through config.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.config import CONFIG, Config, resolve_path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

Encoder = Callable[[list[str]], np.ndarray]

# Loaded once per process and reused. In the web interface deduplicate() runs
# once per job; reloading a 90 MB model every time would waste seconds per job.
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def get_embedding_model(name: str | None = None, cfg: Config = CONFIG):
    """Return the cached SentenceTransformer, loading it on CPU on first use.

    all-MiniLM-L6-v2: ~90 MB, 384-dimensional, fast on CPU, trained for
    semantic similarity of short sentences, which is exactly what questions are.
    """
    name = name or cfg.deduplicate.embedding_model
    with _MODEL_LOCK:
        if name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer  # heavy import, done lazily

            try:
                from transformers.utils import logging as hf_logging
                hf_logging.disable_progress_bar()
            except Exception:
                pass
            try:
                # Cached copy first: no network call, so dedup and the tests work offline.
                model = SentenceTransformer(name, device="cpu", local_files_only=True)
            except Exception:
                print(f"Downloading embedding model '{name}' (~90 MB, first run only)...")
                model = SentenceTransformer(name, device="cpu")
            _MODEL_CACHE[name] = model
        return _MODEL_CACHE[name]


def embed_questions(questions: list[str], cfg: Config = CONFIG) -> np.ndarray:
    """Encode with ``normalize_embeddings=True`` so a dot product equals cosine similarity."""
    model = get_embedding_model(cfg=cfg)
    return model.encode(questions, batch_size=cfg.deduplicate.batch_size,
                        normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True)


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def dedup_texts(pairs: list[dict[str, Any]], cfg: Config = CONFIG) -> list[str]:
    """The text embedded for each pair: the question, or question + answer."""
    if cfg.deduplicate.embed == "question_answer":
        return [f"{p['question']} {p['answer']}" for p in pairs]
    return [p["question"] for p in pairs]


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def answer_numbers(answer: str) -> frozenset[str]:
    """Numbers stated in an answer ("25 g", "HR 3.4", "aged 6-9"), normalised."""
    return frozenset(n.replace(",", "") for n in _NUMBER.findall(answer))


def find_duplicates(
    embeddings: np.ndarray,
    threshold: float,
    order: list[int] | None = None,
    numbers: list[frozenset[str]] | None = None,
) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Greedy single pass: keep the first occurrence, drop anything too similar to a kept item.

    ``order`` is the visiting order (default: as given). Returns
    ``(kept_indices, [(removed_index, kept_index_it_duplicates, similarity)])``.
    If ``numbers`` is given, two items whose number sets are both non-empty
    and differ are never treated as duplicates, however similar their text.
    """
    n = len(embeddings)
    order = list(range(n)) if order is None else order
    sim = embeddings @ embeddings.T
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
    return kept, removed


def deduplicate(
    pairs: list[dict[str, Any]],
    cfg: Config = CONFIG,
    *,
    output_dir: str | Path | None = None,
    write: bool = True,
    encoder: Encoder | None = None,
    threshold: float | None = None,
    report: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove semantic duplicates; return ``(unique, duplicate_records)``.

    Visiting order is highest quality score first (ties keep original order),
    so when two pairs say the same thing the better-scored one survives.
    Every removal is written to ``duplicate_pairs.jsonl`` with the pair it
    duplicated and the similarity, so the evidence stays visible.
    ``encoder`` lets tests inject embeddings; ``threshold`` overrides config.
    Why 0.90 with question+answer embeddings: on this data, restatements of the
    same fact scored 0.90-0.99 and distinct facts about the same topic 0.65-0.87.
    """
    threshold = cfg.deduplicate.similarity_threshold if threshold is None else threshold
    if len(pairs) < 2:
        unique, records = list(pairs), []
    else:
        texts = dedup_texts(pairs, cfg)
        emb = encoder(texts) if encoder else embed_questions(texts, cfg)
        emb = _normalise_rows(np.asarray(emb, dtype=np.float32))
        order = sorted(range(len(pairs)), key=lambda i: (-(pairs[i].get("quality_score") or 0), i))
        numbers = [answer_numbers(p["answer"]) for p in pairs] if cfg.deduplicate.require_same_numbers else None
        kept_idx, removed = find_duplicates(emb, threshold, order, numbers)
        unique = [pairs[i] for i in sorted(kept_idx)]
        records = []
        for r_idx, k_idx, score in removed:
            r, k = pairs[r_idx], pairs[k_idx]
            records.append({
                "removed_question": r["question"],
                "removed_answer": r["answer"],
                "removed_source": r.get("source"),
                "removed_page": r.get("page"),
                "removed_quality_score": r.get("quality_score"),
                "duplicate_of_question": k["question"],
                "duplicate_of_answer": k["answer"],
                "duplicate_of_source": k.get("source"),
                "duplicate_of_page": k.get("page"),
                "duplicate_of_quality_score": k.get("quality_score"),
                "similarity": round(score, 4),
                "compared": cfg.deduplicate.embed,
                "removed_id": r.get("id"),
                "duplicate_of_id": k.get("id"),
            })
        records.sort(key=lambda rec: -rec["similarity"])

    if write:
        path = resolve_path(output_dir or cfg.paths.output_dir) / "duplicate_pairs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(pairs)
    pct = len(records) / n if n else 0.0
    print(f"Deduplication ({cfg.deduplicate.embed} embeddings, threshold {threshold}"
          f"{', number guard' if cfg.deduplicate.require_same_numbers else ''}): removed {len(records)} of {n} "
          f"({pct:.1%}); {len(unique)} unique pairs remain.")
    if report is not None:
        report.update({"input": n, "duplicates_removed": len(records), "unique": len(unique),
                       "threshold": threshold})
    return unique, records


_WORD = re.compile(r"[a-z0-9']+")


def diversity_stats(pairs: list[dict[str, Any]], top: int = 15) -> dict[str, Any]:
    """Distribution of question opening words.

    Why: a dataset where 80% of questions begin "What is" is low quality even
    when every pair scores well individually; it teaches one question shape.
    """
    first, first_two = Counter(), Counter()
    for p in pairs:
        words = _WORD.findall(p["question"].lower())
        if words:
            first[words[0]] += 1
            first_two[" ".join(words[:2])] += 1
    n = sum(first.values())
    top_word, top_count = first.most_common(1)[0] if first else ("", 0)
    return {
        "questions": n,
        "unique_first_words": len(first),
        "unique_first_two_words": len(first_two),
        "top_first_word": top_word,
        "top_first_word_share": round(top_count / n, 4) if n else 0.0,
        "first_word_distribution": dict(first.most_common(top)),
        "first_two_words_distribution": dict(first_two.most_common(top)),
    }


if __name__ == "__main__":
    demo = [
        {"id": "a", "question": "What is the intensive phase duration?",
         "answer": "The intensive phase lasts two months.", "quality_score": 6},
        {"id": "b", "question": "How long does the intensive phase last?", "answer": "It lasts two months.",
         "quality_score": 5},
        {"id": "c", "question": "Which drugs are used in the continuation phase?",
         "answer": "Isoniazid and rifampicin.", "quality_score": 6},
        {"id": "d", "question": "What daily fibre intake does WHO recommend for adults?",
         "answer": "At least 25 g per day.", "quality_score": 6},
        {"id": "e", "question": "What daily fibre intake does WHO recommend for children aged 2 to 5 years?",
         "answer": "At least 15 g per day.", "quality_score": 6},
    ]
    for basis in ("question", "question_answer"):
        texts = [p["question"] + (" " + p["answer"] if basis == "question_answer" else "") for p in demo]
        e = embed_questions(texts)
        print(f"Cosine similarity matrix ({basis}):")
        print(np.round(e @ e.T, 3))
    uniq, dups = deduplicate(demo, write=False)
    for d in dups:
        print(f"REMOVED: {d['removed_question']!r}\n   DUPLICATE OF: {d['duplicate_of_question']!r} "
              f"(similarity {d['similarity']})")
    print("Diversity:", json.dumps(diversity_stats(demo), indent=1))
