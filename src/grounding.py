"""
Independent, non-LLM grounding check - a second opinion on hallucination.

The judge in validate.py scores groundedness, but the judge is itself a
language model: asking a model to mark its own family's homework is exactly
the weakness an examiner will press on. This module checks the same thing
mechanically, with no model involved at all.

Two signals, both deterministic and reproducible:

  lexical overlap  What share of the answer's content words appear in the
                   source passage? A grounded answer mostly reuses the
                   passage's vocabulary.

  unsupported numbers  Numbers in the answer that appear nowhere in the
                   passage. This is the sharp one. A fabricated dose ("300 mg
                   once daily") or an invented threshold is almost always a
                   number the source never contained, and in a clinical
                   dataset a wrong number is the most dangerous error there
                   is. Prose can be paraphrased; a dose cannot.

Agreement between this and the LLM judge is itself a result worth reporting:
where the two disagree is where the dataset needs a human eye.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

try:
    from config import OUTPUT_DIR, use_utf8_console
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import OUTPUT_DIR, use_utf8_console

ANALYSIS_FILE = OUTPUT_DIR / "grounding_analysis.jsonl"

# Deliberately a small hand-written list rather than an NLTK download: these
# are the words whose presence says nothing about grounding, and keeping the
# list here means the check has no external dependency and cannot drift.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "have", "has", "had", "having", "will",
    "would", "shall", "should", "can", "could", "may", "might", "must", "to",
    "of", "in", "on", "at", "by", "for", "with", "from", "as", "into", "about",
    "it", "its", "they", "them", "their", "there", "which", "who", "whom",
    "what", "when", "where", "how", "why", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "too", "very", "s", "t", "just", "also", "you", "your", "we",
    "our", "he", "she", "his", "her", "him", "i", "me", "my", "one", "two",
    "up", "out", "over", "under", "between", "during", "before", "after",
    "above", "below", "again", "further", "once", "here", "because", "while",
}

_WORD = re.compile(r"[a-z][a-z\-']+")
# Matches 200, 3.5, 1,000 and 12% - the shapes clinical thresholds arrive in.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def content_words(text: str) -> set[str]:
    """Lowercase content words, stopwords and very short tokens removed."""
    return {
        w for w in _WORD.findall(text.lower())
        if w not in STOPWORDS and len(w) > 2
    }


def numbers_in(text: str) -> set[str]:
    """
    Numbers, normalised so 1,000 and 1000 compare equal.

    Trailing zeros after a decimal point are stripped too, so "3.0" in an
    answer matches "3" in the passage rather than being reported as invented.
    """
    out = set()
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        out.add(str(int(value)) if value.is_integer() else str(value))
    return out


def analyse_pair(answer: str, passage: str) -> dict:
    """Compare one answer against its source passage."""
    a_words = content_words(answer)
    p_words = content_words(passage)
    a_nums = numbers_in(answer)
    p_nums = numbers_in(passage)

    novel_words = sorted(a_words - p_words)
    novel_numbers = sorted(a_nums - p_nums)

    overlap = len(a_words & p_words) / len(a_words) if a_words else 1.0

    return {
        "lexical_overlap": round(overlap, 3),
        "answer_words": len(a_words),
        "novel_words": novel_words[:20],
        "novel_word_count": len(novel_words),
        "numbers_in_answer": sorted(a_nums),
        "unsupported_numbers": novel_numbers,
        "has_unsupported_number": bool(novel_numbers),
        # Bands chosen from the observed distribution on this corpus: a
        # grounded answer paraphrases lightly and lands well above 0.7.
        "grounding_band": (
            "strong" if overlap >= 0.7 else "moderate" if overlap >= 0.5 else "weak"
        ),
    }


def build_page_lookup(pdf_dir=None) -> dict[tuple[str, int], str]:
    """
    Map (source file, page number) to that page's full text.

    Pages rather than chunks: a pair's answer may draw on a sentence that the
    600-character splitter cut in half, and judging it against only one half
    would report grounded content as invented.
    """
    import warnings

    warnings.filterwarnings("ignore")
    from config import RAW_DIR
    from ingest import load_pdfs

    lookup: dict[tuple[str, int], str] = {}
    for page in load_pdfs(pdf_dir or RAW_DIR):
        source = page.metadata.get("source_file", "unknown")
        # generate.py stores page as metadata["page"] + 1, so match that.
        number = int(page.metadata.get("page", 0)) + 1
        key = (source, number)
        lookup[key] = lookup.get(key, "") + "\n" + page.page_content
    return lookup


def analyse_dataset(
    pairs: list[dict], lookup: dict | None = None, pdf_dir=None
) -> list[dict]:
    """Run the check over every pair, returning one analysis record each."""
    if lookup is None:
        # Pairs still carrying chunk_text need no PDF at all - that is the
        # normal path, called from export.py while the pipeline is running.
        lookup = {} if all(p.get("chunk_text") for p in pairs) else build_page_lookup(pdf_dir)

    out = []
    for p in pairs:
        # Prefer the exact passage when the pipeline still has it; fall back to
        # the page for records that have already been stripped for export.
        passage = p.get("chunk_text") or lookup.get((p.get("source"), p.get("page")), "")
        record = {
            "question": p.get("question"),
            "answer": p.get("answer"),
            "source": p.get("source"),
            "page": p.get("page"),
            "question_type": p.get("question_type"),
            "quality_score": p.get("quality_score"),
        }
        if p.get("scores"):
            record["judge_groundedness"] = p["scores"].get("groundedness")
            record["judge_specificity"] = p["scores"].get("specificity")
            record["judge_completeness"] = p["scores"].get("completeness")
        if not passage:
            record["passage_found"] = False
            out.append(record)
            continue
        record["passage_found"] = True
        record.update(analyse_pair(p.get("answer", ""), passage))
        out.append(record)
    return out


def summarise(records: list[dict]) -> dict:
    """Aggregate the per-pair analysis into dashboard-ready numbers."""
    scored = [r for r in records if r.get("passage_found")]
    if not scored:
        return {"analysed": 0}

    overlaps = [r["lexical_overlap"] for r in scored]
    flagged = [r for r in scored if r.get("has_unsupported_number")]
    bands = Counter(r["grounding_band"] for r in scored)

    by_source: dict[str, list[float]] = {}
    by_type: dict[str, list[float]] = {}
    for r in scored:
        by_source.setdefault(r.get("source", "?"), []).append(r["lexical_overlap"])
        by_type.setdefault(r.get("question_type", "?"), []).append(r["lexical_overlap"])

    summary = {
        "analysed": len(scored),
        "not_matched": len(records) - len(scored),
        "avg_lexical_overlap": round(sum(overlaps) / len(overlaps), 3),
        "min_lexical_overlap": round(min(overlaps), 3),
        "bands": {k: bands.get(k, 0) for k in ("strong", "moderate", "weak")},
        "band_pct": {
            k: round(bands.get(k, 0) / len(scored) * 100, 1)
            for k in ("strong", "moderate", "weak")
        },
        "unsupported_number_pairs": len(flagged),
        "unsupported_number_pct": round(len(flagged) / len(scored) * 100, 1),
        "by_source": {
            k: round(sum(v) / len(v), 3) for k, v in sorted(by_source.items())
        },
        "by_question_type": {
            k: round(sum(v) / len(v), 3) for k, v in sorted(by_type.items())
        },
        "overlap_histogram": _histogram(overlaps),
    }

    # Where the mechanical check and the LLM judge disagree is the interesting
    # part - it is the only honest way to say whether the judge can be trusted.
    judged = [r for r in scored if r.get("judge_groundedness") is not None]
    if judged:
        agree = sum(
            1 for r in judged
            if (r["judge_groundedness"] == 2) == (r["lexical_overlap"] >= 0.7)
        )
        summary["judge_comparison"] = {
            "pairs": len(judged),
            "agreement_pct": round(agree / len(judged) * 100, 1),
            "judge_clean_but_low_overlap": sum(
                1 for r in judged
                if r["judge_groundedness"] == 2 and r["lexical_overlap"] < 0.5
            ),
            "judge_flagged_but_high_overlap": sum(
                1 for r in judged
                if r["judge_groundedness"] < 2 and r["lexical_overlap"] >= 0.7
            ),
        }
    return summary


def _histogram(values: list[float], bins: int = 10) -> list[dict]:
    counts = [0] * bins
    for v in values:
        idx = min(int(v * bins), bins - 1)
        counts[idx] += 1
    return [
        {"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "count": c}
        for i, c in enumerate(counts)
    ]


def write_analysis(
    pairs: list[dict], lookup: dict | None = None, pdf_dir=None
) -> dict:
    """Analyse, write the per-pair file, and return the summary."""
    records = analyse_dataset(pairs, lookup, pdf_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_FILE, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return summarise(records)


if __name__ == "__main__":
    use_utf8_console()
    from config import DATASET_FILE

    print("=" * 70)
    print("GROUNDING ANALYSIS - independent hallucination check")
    print("=" * 70)

    if not DATASET_FILE.exists():
        raise SystemExit(f"{DATASET_FILE} not found. Run the pipeline first.")

    pairs = [
        json.loads(line)
        for line in DATASET_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pdf-dir",
        default=None,
        help="where the source PDFs live, if they are no longer in data/raw",
    )
    args = ap.parse_args()

    print(f"Analysing {len(pairs)} pairs against their source pages...")

    summary = write_analysis(pairs, pdf_dir=args.pdf_dir)

    if not summary.get("analysed"):
        sources = sorted({p.get("source") for p in pairs})
        print(
            "\n  Could not match any pair to its source page.\n"
            "  The PDFs in data/raw are not the ones this dataset came from.\n"
            f"    dataset was built from : {sources}\n"
            "  Point at the originals with --pdf-dir, or simply run the pipeline\n"
            "  again - grounding is computed automatically during export."
        )
        raise SystemExit(1)

    print(f"\n  Analysed              : {summary['analysed']}")
    print(f"  Average word overlap  : {summary['avg_lexical_overlap']}")
    print(f"  Strongly grounded     : {summary['band_pct']['strong']}%")
    print(f"  Moderately grounded   : {summary['band_pct']['moderate']}%")
    print(f"  Weakly grounded       : {summary['band_pct']['weak']}%")
    print(
        f"  Unsupported numbers   : {summary['unsupported_number_pairs']} pairs "
        f"({summary['unsupported_number_pct']}%)"
    )
    if "judge_comparison" in summary:
        jc = summary["judge_comparison"]
        print(f"  Agreement with judge  : {jc['agreement_pct']}% over {jc['pairs']} pairs")

    print(f"\n  Written -> {ANALYSIS_FILE.name}")
