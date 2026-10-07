"""
Tests for the rules that decide what ends up in the dataset.

Run with:  python -m pytest tests -q        (or: python tests/test_pipeline.py)

Scope: the pure, offline decisions - the keep rule, deduplication, the
abstention plumbing, checkpoint reuse and the export shape. Nothing here calls
the Groq API, so the suite runs in seconds and costs nothing.

These are the parts where a silent change is most expensive: a wrong keep rule
quietly admits fabricated answers, and a wrong dedup rule quietly deletes
distinct facts. Both look like a working pipeline from the outside.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import checkpoint  # noqa: E402
import config  # noqa: E402
from deduplicate import (  # noqa: E402
    answer_numbers,
    dedup_text,
    deduplicate_pairs,
    lexical_overlap,
)
from export import to_chatml, to_clean_records  # noqa: E402
from generate import (  # noqa: E402
    MIN_ANSWER_CHARS,
    clean_pairs,
    extract_json_array,
    extract_json_object,
    unanswerable_chunk_ids,
)
from validate import MIN_CRITERION_SCORES, keep_decision  # noqa: E402


class FakeChunk:
    """Stands in for a LangChain Document."""

    def __init__(self, text="A passage about dosing schedules.", page=0):
        self.page_content = text
        self.metadata = {"source_file": "who.pdf", "page": page, "start_index": 0}


def pair(question, answer, **extra):
    row = {
        "question": question,
        "answer": answer,
        "question_type": "factual",
        "answerable": True,
        "source": "who.pdf",
        "page": 1,
        "chunk_text": "ctx",
    }
    row.update(extra)
    return row


# --- the keep rule ---------------------------------------------------------


@pytest.mark.parametrize(
    "scores, expected, why",
    [
        ({"groundedness": 2, "specificity": 2, "completeness": 2}, True, "perfect"),
        ({"groundedness": 2, "specificity": 1, "completeness": 1}, True, "on every floor"),
        ({"groundedness": 1, "specificity": 2, "completeness": 2}, False, "unstated detail"),
        ({"groundedness": 0, "specificity": 2, "completeness": 2}, False, "fabricated"),
        ({"groundedness": 2, "specificity": 2, "completeness": 0}, False, "2+2+0 = 4"),
        ({"groundedness": 2, "specificity": 0, "completeness": 2}, False, "vague question"),
        ({}, False, "no scores at all"),
    ],
)
def test_keep_decision(scores, expected, why):
    assert keep_decision(scores) is expected, why


def test_groundedness_floor_is_strict():
    """The whole point of the floor: a passing total cannot rescue a 1."""
    scores = {"groundedness": 1, "specificity": 2, "completeness": 2}
    assert sum(scores.values()) >= 4
    assert keep_decision(scores) is False
    assert MIN_CRITERION_SCORES["groundedness"] == 2


def test_floors_can_be_overridden_for_experiments():
    scores = {"groundedness": 1, "specificity": 2, "completeness": 2}
    assert keep_decision(scores, 4, {}) is True


# --- deduplication ---------------------------------------------------------


def test_answer_numbers_normalises():
    assert answer_numbers("Give 1,500 units") == answer_numbers("Give 1500 units")
    assert answer_numbers("no numbers here") == frozenset()


def test_dedup_text_includes_the_answer():
    assert dedup_text({"question": "Q?", "answer": "A."}) == "Q? A."


def test_lexical_overlap_bounds():
    assert lexical_overlap("same", "same") == pytest.approx(1.0)
    assert 0.0 <= lexical_overlap("abc", "xyz") < 0.5


@pytest.mark.slow
def test_dedup_keeps_facts_that_differ_only_by_number():
    """The regression that motivated the number guard."""
    pairs = [
        pair("What daily fibre does WHO recommend for adults?",
             "At least 25 g of dietary fibre per day."),
        pair("What daily fibre does WHO recommend for children?",
             "At least 15 g of dietary fibre per day."),
    ]
    unique, dupes = deduplicate_pairs(pairs, write_duplicates=False)
    assert len(unique) == 2, "answers stating different numbers must not merge"
    assert dupes == []


@pytest.mark.slow
def test_dedup_still_removes_a_true_reword():
    pairs = [
        pair("At what age is the first measles dose given?",
             "The first dose is given at nine months of age."),
        pair("What age is the first measles dose administered?",
             "The first dose is given at nine months of age."),
    ]
    unique, dupes = deduplicate_pairs(pairs, write_duplicates=False)
    assert len(unique) == 1
    assert len(dupes) == 1
    assert "lexical_overlap" in dupes[0]
    assert dupes[0]["similarity"] >= 0.90


# --- generation ------------------------------------------------------------


def test_short_answers_are_dropped():
    raw = [
        {"question": "What is the first dose of the vaccine?", "answer": "Nine.",
         "question_type": "factual"},
        {"question": "What is the first dose of the vaccine?",
         "answer": "The first dose is given at nine months of age.",
         "question_type": "factual"},
    ]
    kept = clean_pairs(raw, FakeChunk())
    assert len(kept) == 1
    assert len(kept[0]["answer"]) >= MIN_ANSWER_CHARS


def test_questions_about_the_passage_are_dropped():
    raw = [{"question": "What does this passage describe about dosing?",
            "answer": "It describes the dosing schedule in detail.",
            "question_type": "factual"}]
    assert clean_pairs(raw, FakeChunk()) == []


def test_generated_pairs_are_marked_answerable():
    raw = [{"question": "What is the recommended first dose?",
            "answer": "The first dose is given at nine months of age.",
            "question_type": "factual"}]
    assert clean_pairs(raw, FakeChunk())[0]["answerable"] is True


def test_unanswerable_selection_is_seeded_and_proportional():
    first = unanswerable_chunk_ids(100)
    assert first == unanswerable_chunk_ids(100), "selection must be reproducible"
    assert len(first) == 10
    assert unanswerable_chunk_ids(0) == set()
    assert unanswerable_chunk_ids(20, ratio=0) == set()
    assert len(unanswerable_chunk_ids(3)) == 1, "a small run still gets one"


def test_json_extraction_tolerates_model_sloppiness():
    assert extract_json_array('```json\n[{"a":1}]\n```') == [{"a": 1}]
    assert extract_json_array('Sure! [{"a":1}] hope that helps') == [{"a": 1}]
    assert extract_json_array("not json at all") is None
    assert extract_json_object('{"question":"Q?"}') == {"question": "Q?"}
    assert extract_json_object("[1,2]") is None


# --- export ----------------------------------------------------------------


def test_export_strips_internals_but_keeps_answerable():
    rows = to_clean_records([pair("Q long enough to be real?", "A long enough answer.",
                                  quality_score=6, scores={"groundedness": 2})])
    assert "chunk_text" not in rows[0]
    assert "scores" not in rows[0]
    assert rows[0]["answerable"] is True
    assert rows[0]["quality_score"] == 6


def test_chatml_shape_and_abstention_flag():
    rows = to_chatml([
        pair("Q one?", "A one.", quality_score=6),
        pair("Q two?", config.ABSTENTION_ANSWER, answerable=False,
             question_type="unanswerable", quality_score=None),
    ])
    assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]
    assert rows[1]["answerable"] is False
    assert rows[1]["quality_score"] is None
    assert rows[1]["messages"][-1]["content"] == config.ABSTENTION_ANSWER


# --- checkpoints -----------------------------------------------------------


def test_checkpoint_round_trip(tmp_path):
    f = tmp_path / "cp.jsonl"
    fp = checkpoint.fingerprint(model="m", temperature=0.4)
    checkpoint.append(f, fp, "k1", [{"question": "Q1"}])
    assert checkpoint.load(f, fp)["k1"]["payload"][0]["question"] == "Q1"


def test_checkpoint_ignores_a_different_configuration(tmp_path):
    f = tmp_path / "cp.jsonl"
    a = checkpoint.fingerprint(model="m1", temperature=0.4)
    b = checkpoint.fingerprint(model="m2", temperature=0.4)
    checkpoint.append(f, a, "k1", ["from m1"])
    assert checkpoint.load(f, b) == {}, "scores from another model must not be reused"
    assert len(checkpoint.load(f, a)) == 1


def test_checkpoint_survives_a_truncated_final_line(tmp_path):
    f = tmp_path / "cp.jsonl"
    fp = checkpoint.fingerprint(model="m")
    checkpoint.append(f, fp, "k1", ["ok"])
    with open(f, "a", encoding="utf-8") as fh:
        fh.write('{"fingerprint": "' + fp + '", "key": "k2", "payl')
    assert len(checkpoint.load(f, fp)) == 1, "a crash mid-write must not poison the file"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
