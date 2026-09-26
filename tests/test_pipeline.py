"""Unit tests. No API calls: every LLM interaction is stubbed, and any attempt
to reach the real API fails the test. Runs offline with no GROQ_API_KEY set.

    pytest -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Offline for Hugging Face before anything imports it; the key is removed per test below.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from src import agreement, generate, llm, validate  # noqa: E402
from src.config import (  # noqa: E402
    CONFIG, PROJECT_ROOT, ConfigError, is_public, project_relative, require_api_key, with_overrides,
)
from src.deduplicate import deduplicate, diversity_stats, find_duplicates  # noqa: E402
from src.export import (  # noqa: E402
    EXPORT_FIELDS, archive_stale_files, clean_pairs, load_jsonl, save_jsonl, to_chatml_format,
)
from src.ingest import boilerplate_reason, chunk_documents, clean_page_text, select_chunks  # noqa: E402
from src.uploads import UploadedPDF, replace_pdfs, safe_filename, validate_uploads  # noqa: E402

FLOORS = {"groundedness": 2, "specificity": 1, "completeness": 1}


# ---------------------------------------------------------------------------
# Fixtures: no key, no network, no sleeping
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)

    def forbid(*args, **kwargs):
        raise AssertionError("a test tried to create a real LLM client")

    monkeypatch.setattr(llm, "get_chat_model", forbid)
    # No pacing in tests: the real limiter would spin for request_delay_seconds per call.
    monkeypatch.setattr(llm, "get_limiter", lambda model_id, cfg=None: llm.RateLimiter(10**6, 10**9, 0.0))
    yield


class FakeLLM:
    """Stands in for ChatGroq: returns canned replies in order and counts calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        reply = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        return SimpleNamespace(content=reply, usage_metadata={"input_tokens": 100, "output_tokens": 50})


def use_fake(monkeypatch, replies) -> FakeLLM:
    fake = FakeLLM(replies)
    monkeypatch.setattr(llm, "get_chat_model", lambda *a, **k: fake)
    return fake


GOOD_PAIRS = json.dumps([
    {"question": "What minimum daily fibre intake does WHO recommend for adults?",
     "answer": "WHO recommends that adults consume at least 25 g of naturally occurring dietary fibre per day.",
     "question_type": "factual"},
    {"question": "How does WHO define moderate wasting in children aged 6-59 months?",
     "answer": "Moderate wasting is a weight-for-height between -3 and -2 standard deviations of the median.",
     "question_type": "definitional"},
])


# ---------------------------------------------------------------------------
# Tests: no API key needed
# ---------------------------------------------------------------------------
def test_no_api_key_is_set_and_missing_key_error_is_clear():
    assert "GROQ_API_KEY" not in os.environ
    with pytest.raises(ConfigError, match="GROQ_API_KEY is not set"):
        require_api_key()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def _long_text(n_sentences: int = 60) -> str:
    return " ".join(f"Sentence number {i} explains a separate clinical recommendation in plain words."
                    for i in range(n_sentences))


def test_chunking_respects_size_and_overlap_and_drops_short_chunks():
    cfg = with_overrides(CONFIG, {"ingest.filter_boilerplate": False})
    docs = [
        Document(page_content=_long_text(), metadata={"source": "a.pdf", "page": 1, "page_label": "1"}),
        Document(page_content="Too short to use.", metadata={"source": "a.pdf", "page": 2, "page_label": "2"}),
    ]
    chunks = chunk_documents(docs, cfg)
    assert len(chunks) > 3
    assert all(len(c.page_content) <= cfg.ingest.chunk_size for c in chunks)
    assert all(len(c.page_content) >= cfg.ingest.min_chunk_chars for c in chunks)
    assert all(c.metadata["page"] == 1 for c in chunks), "the short page-2 fragment must be dropped"
    for a, b in zip(chunks, chunks[1:]):
        end_a = a.metadata["start_index"] + len(a.page_content)
        overlap = end_a - b.metadata["start_index"]
        assert 0 < overlap <= cfg.ingest.chunk_overlap, "consecutive chunks must overlap, by at most chunk_overlap"
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_clean_page_text_removes_headers_page_numbers_and_joins_lines():
    raw = ("17\nGUIDELINE: CARBOHYDRATE INTAKE\nWHO recommends that adults eat at least \n400 g of vegetables "
           "and fruits per day.\nIntake of carbo - \nhydrate should come from whole grains.\n" + chr(0xE021) + " FOOTER")
    text = clean_page_text(raw, running_headers={"guideline: carbohydrate intake", "footer"})
    assert "GUIDELINE" not in text and not text.startswith("17") and "FOOTER" not in text
    assert "at least 400 g of vegetables" in text
    assert "carbohydrate should come" in text
    assert chr(0xE021) not in text


def test_boilerplate_filter_flags_references_and_keeps_recommendations():
    refs = ("14. Schwingshackl L, Bogensberger B, Hoffmann G. Diet quality. J Acad Nutr Diet. 2018;118(1):74-100. "
            "15. Erhardt L, Hobbs FD. Public perceptions. Int J Clin Pract. 2002;56(9):638-44.")
    prose = ("WHO recommends that carbohydrate intake should come primarily from whole grains, vegetables, "
             "fruits and pulses. In adults, WHO recommends an intake of at least 25 g per day of dietary fibre.")
    assert boilerplate_reason(refs) == "references"
    assert boilerplate_reason(prose) is None


def test_select_chunks_is_deterministic_and_covers_every_source():
    chunks = [Document(page_content="x" * 200, metadata={"source": s, "chunk_index": i, "chunk_id": f"{s}-{i}"})
              for i, s in enumerate(["a.pdf"] * 60 + ["b.pdf"] * 30 + ["c.pdf"] * 10)]
    first = select_chunks(chunks, 10)
    assert [c.metadata["chunk_id"] for c in first] == [c.metadata["chunk_id"] for c in select_chunks(chunks, 10)]
    counts = {s: sum(1 for c in first if c.metadata["source"] == s) for s in ("a.pdf", "b.pdf", "c.pdf")}
    assert counts == {"a.pdf": 6, "b.pdf": 3, "c.pdf": 1}
    assert len(select_chunks(chunks, None)) == 100


# ---------------------------------------------------------------------------
# JSON handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    '```json\n[{"a": 1}]\n```',
    '```\n[{"a": 1}]\n```',
    '[{"a": 1}]',
    'Here are the pairs:\n[{"a": 1}]\nHope this helps.',
    '{"pairs": [{"a": 1}]}',
])
def test_parse_json_handles_fenced_unfenced_and_wrapped_output(raw):
    assert llm.parse_json(raw, list) == [{"a": 1}]


def test_strip_code_fences():
    assert llm.strip_code_fences('```json\n{"x": 2}\n```') == '{"x": 2}'
    assert llm.strip_code_fences('{"x": 2}') == '{"x": 2}'


@pytest.mark.parametrize("raw", ["", "not json at all", "[{'single': 'quotes'}]", '[{"a": 1'])
def test_parse_json_rejects_malformed_output(raw):
    with pytest.raises(ValueError):
        llm.parse_json(raw, list)


def test_malformed_llm_output_returns_empty_list_not_exception(monkeypatch):
    fake = use_fake(monkeypatch, ["Sorry, I cannot produce JSON."])
    errors: list[str] = []
    assert generate.generate_qa_pairs("Some passage text about fibre intake.", CONFIG, errors=errors) == []
    assert fake.calls == CONFIG.model.max_retries + 1, "one attempt plus max_retries retries"
    assert errors and "no JSON" in errors[0]


def test_fenced_llm_output_is_parsed_and_filtered(monkeypatch):
    use_fake(monkeypatch, ["```json\n" + GOOD_PAIRS + "\n```"])
    pairs = generate.generate_qa_pairs("passage", CONFIG)
    assert [p["question_type"] for p in pairs] == ["factual", "definitional"]


def test_filter_pairs_drops_short_and_generic_and_rewrites_passage_references():
    raw = [
        {"question": "What is this passage about?", "answer": "It is about dietary fibre and carbohydrate quality.",
         "question_type": "factual"},
        {"question": "According to the passage, what is the fibre target for adults?",
         "answer": "The passage states at least 25 g of dietary fibre per day.", "question_type": "factual"},
        {"question": "Short?", "answer": "Too short.", "question_type": "factual"},
        {"question": "What minimum daily fibre intake does WHO recommend for adults?",
         "answer": "At least 25 g per day of naturally occurring dietary fibre.", "question_type": "weird-type"},
        "not a dict",
    ]
    kept = generate.filter_pairs(raw, CONFIG)
    ref = CONFIG.generate.source_reference
    assert [p["question"] for p in kept] == [
        f"According to {ref}, what is the fibre target for adults?",
        "What minimum daily fibre intake does WHO recommend for adults?",
    ]
    assert kept[0]["answer"].startswith(ref[:1].upper() + ref[1:] + " states")
    assert kept[1]["question_type"] == "other"


def test_refill_keeps_complete_legacy_chunks_and_regenerates_the_rest(monkeypatch, tmp_path):
    chunks = [Document(page_content=f"Passage {i} about WHO fibre recommendations for adults and children.",
                       metadata={"source": "a.pdf", "page": i + 1, "chunk_id": f"a-p{i + 1}-0", "chunk_index": i})
              for i in range(3)]
    cfg = with_overrides(CONFIG, {"generate.unanswerable_ratio": 0.0})
    three = json.dumps(json.loads(GOOD_PAIRS) + [{
        "question": "What vegetable and fruit intake does WHO recommend for adults per day?",
        "answer": "WHO recommends at least 400 g of vegetables and fruits per day for adults.",
        "question_type": "factual"}])
    # Write a checkpoint as the legacy prompt would have: chunk 0 complete, chunk 1 short, chunk 2 empty.
    legacy = "gen-v2"
    fp = generate.run_fingerprint(chunks, cfg, "generate", legacy, generate.LEGACY_PROMPT_VERSIONS[legacy])
    recs = []
    for i, n in enumerate((3, 1, 0)):
        pairs = [{"question": f"legacy q{i}-{k} about fibre?", "answer": "legacy answer long enough here.",
                  "id": f"L{i}{k}", "answerable": True} for k in range(n)]
        recs.append({"fingerprint": fp, "key": chunks[i].metadata["chunk_id"], "status": "ok" if n else "skipped",
                     "pairs": pairs, "unanswerable": None, "usage": {}})
    generate.append_checkpoint(tmp_path / "generate_checkpoint.jsonl", recs)
    fake = use_fake(monkeypatch, [three])
    out = generate.generate_from_chunks(chunks, cfg, cache_dir=tmp_path, refill=True)
    assert fake.calls == 2, "only the incomplete chunks are regenerated"
    assert sum(1 for p in out if p["id"].startswith("L")) == 3
    assert sum(1 for p in out if p.get("prompt_version") == generate.PROMPT_VERSION) == 6
    fake.calls = 0
    generate.generate_from_chunks(chunks, cfg, cache_dir=tmp_path, resume=True)
    assert fake.calls == 0, "a later plain --resume finds everything"


def test_generation_checkpoint_resume_makes_no_new_calls(monkeypatch, tmp_path):
    chunks = [Document(page_content=f"Passage {i} about WHO fibre recommendations for adults and children.",
                       metadata={"source": "a.pdf", "page": i + 1, "chunk_id": f"a-p{i + 1}-0", "chunk_index": i})
              for i in range(4)]
    cfg = with_overrides(CONFIG, {"generate.unanswerable_ratio": 0.0, "run.checkpoint_every": 2})
    fake = use_fake(monkeypatch, [GOOD_PAIRS])
    first = generate.generate_from_chunks(chunks, cfg, cache_dir=tmp_path)
    assert fake.calls == 4 and len(first) == 8
    assert (tmp_path / "generate_checkpoint.jsonl").is_file()
    fake.calls = 0
    resumed = generate.generate_from_chunks(chunks, cfg, cache_dir=tmp_path, resume=True)
    assert fake.calls == 0
    assert [p["id"] for p in resumed] == [p["id"] for p in first]


def test_stop_request_checkpoints_finished_chunks_and_resume_continues(monkeypatch, tmp_path):
    from src.pipeline import PipelineCancelled

    chunks = [Document(page_content=f"Passage {i} about WHO fibre recommendations for adults and children.",
                       metadata={"source": "a.pdf", "page": i + 1, "chunk_id": f"a-p{i + 1}-0", "chunk_index": i})
              for i in range(5)]
    cfg = with_overrides(CONFIG, {"generate.unanswerable_ratio": 0.0, "run.checkpoint_every": 25})
    fake = use_fake(monkeypatch, [GOOD_PAIRS])

    def stop_after_two(current, total, stage):
        if current == 2:
            raise PipelineCancelled("Stopped by user")

    with pytest.raises(PipelineCancelled):
        generate.generate_from_chunks(chunks, cfg, stop_after_two, cache_dir=tmp_path)
    assert fake.calls == 2
    assert len(load_jsonl(tmp_path / "generate_checkpoint.jsonl")) == 2, "finished chunks are saved on stop"
    fake.calls = 0
    generate.generate_from_chunks(chunks, cfg, cache_dir=tmp_path, resume=True)
    assert fake.calls == 3, "resume only generates the chunks the stopped run had not finished"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _judged(score_triplet, answerable=True, judge_ok=True):
    g, s, c = score_triplet
    return {"id": f"id{g}{s}{c}", "question": "q", "answer": "a", "answerable": answerable, "judge_ok": judge_ok,
            "scores": {"groundedness": g, "specificity": s, "completeness": c}, "quality_score": g + s + c}


def test_score_filtering_keeps_exactly_pairs_at_or_above_threshold():
    pairs = [_judged((0, 0, 0)), _judged((1, 1, 1)), _judged((2, 1, 1)), _judged((2, 2, 1)), _judged((2, 2, 2))]
    for threshold in range(7):
        kept, rejected = validate.apply_threshold(pairs, threshold)
        assert sorted(p["quality_score"] for p in kept) == sorted(p["quality_score"] for p in pairs
                                                                  if p["quality_score"] >= threshold)
        assert len(kept) + len(rejected) == len(pairs)
        assert all(p["reject_reason"] for p in rejected) and all(p["reject_reason"] == "" for p in kept)


def test_criterion_floors_reject_hallucinated_or_bare_answers_that_pass_on_total():
    hallucinated, yes_no, good = _judged((0, 2, 2)), _judged((2, 2, 0)), _judged((2, 2, 2))
    kept_total_only, _ = validate.apply_threshold([hallucinated, yes_no, good], 4)
    kept_with_floors, _ = validate.apply_threshold([hallucinated, yes_no, good], 4, FLOORS)
    assert len(kept_total_only) == 3
    assert [p["id"] for p in kept_with_floors] == [good["id"]]


def test_judge_failure_is_rejected_never_kept():
    failed = _judged((2, 2, 2), judge_ok=False)
    kept, rejected = validate.apply_threshold([failed], 0)
    assert kept == [] and len(rejected) == 1


def test_parse_scores_recomputes_total_and_rejects_invalid_scores():
    parsed = validate.parse_scores({"groundedness": 2, "specificity": "1", "completeness": 2, "total": 6})
    assert parsed["quality_score"] == 5
    assert validate.parse_scores({"groundedness": 3, "specificity": 1, "completeness": 2}) is None
    assert validate.parse_scores({"groundedness": True, "specificity": 1, "completeness": 2}) is None


def test_grouped_judge_reply_missing_a_pair_is_rejudged(monkeypatch):
    first = json.dumps({"scores": [{"id": 1, "groundedness": 2, "specificity": 2, "completeness": 2,
                                    "total": 6, "reject_reason": ""}]})
    second = json.dumps({"scores": [{"id": 1, "groundedness": 0, "specificity": 2, "completeness": 2,
                                     "total": 4, "reject_reason": "adds a dosage"}]})
    fake = use_fake(monkeypatch, [first, second])
    pairs = [{"question": "q1", "answer": "a1", "source": "a.pdf", "page": 1},
             {"question": "q2", "answer": "a2", "source": "a.pdf", "page": 1}]
    results = validate.score_pairs("passage", pairs, CONFIG, usage=llm.Usage())
    assert fake.calls == 2
    assert results[0]["quality_score"] == 6 and results[1]["scores"]["groundedness"] == 0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def test_greedy_dedup_logic_with_injected_embeddings():
    emb = np.array([[1.0, 0.0], [0.99, 0.141], [0.0, 1.0]])
    kept, removed = find_duplicates(emb / np.linalg.norm(emb, axis=1, keepdims=True), 0.85)
    assert kept == [0, 2] and removed[0][:2] == (1, 0)


def test_dedup_keeps_higher_scored_copy_and_writes_evidence(tmp_path):
    pairs = [{"id": "low", "question": "q-low", "answer": "a", "quality_score": 4},
             {"id": "high", "question": "q-high", "answer": "a", "quality_score": 6}]
    unique, records = deduplicate(pairs, CONFIG, output_dir=tmp_path,
                                  encoder=lambda qs: np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert [p["id"] for p in unique] == ["high"]
    written = load_jsonl(tmp_path / "duplicate_pairs.jsonl")
    assert written[0]["removed_id"] == "low" and written[0]["similarity"] == pytest.approx(1.0)


def test_semantic_dedup_removes_rewording_and_keeps_distinct_facts(tmp_path):
    try:
        from src.deduplicate import get_embedding_model
        get_embedding_model()
    except Exception as exc:  # model not cached yet and we are offline
        pytest.skip(f"embedding model not available offline: {exc}")
    pairs = [
        {"id": "a", "question": "What is the intensive phase duration?",
         "answer": "The intensive phase lasts two months.", "quality_score": 6},
        {"id": "b", "question": "How long does the intensive phase last?", "answer": "It lasts two months.",
         "quality_score": 5},
        {"id": "c", "question": "What daily fibre intake does WHO recommend for adults?",
         "answer": "At least 25 g of dietary fibre per day.", "quality_score": 6},
        # Same wording, different population and number: a distinct fact that must survive.
        {"id": "d", "question": "What daily fibre intake does WHO recommend for children aged 2 to 5 years?",
         "answer": "At least 15 g of dietary fibre per day.", "quality_score": 6},
    ]
    unique, records = deduplicate(pairs, CONFIG, output_dir=tmp_path)
    assert [p["id"] for p in unique] == ["a", "c", "d"]
    assert records[0]["removed_id"] == "b" and records[0]["similarity"] > CONFIG.deduplicate.similarity_threshold


def test_number_guard_blocks_merging_pairs_that_state_different_numbers():
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    numbers = [frozenset({"3.4"}), frozenset({"12.3"}), frozenset({"3.4"})]
    kept, removed = find_duplicates(emb, 0.9, numbers=numbers)
    assert kept == [0, 1] and removed == [(2, 0, pytest.approx(1.0))]


def test_diversity_stats_counts_first_words():
    stats = diversity_stats([{"question": "What is X?"}, {"question": "What is Y?"}, {"question": "How is Z?"}])
    assert stats["unique_first_words"] == 2 and stats["top_first_word"] == "what"
    assert stats["top_first_word_share"] == pytest.approx(2 / 3, abs=1e-4)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def test_jsonl_round_trips_without_data_loss(tmp_path):
    rows = [{"question": "Is ≥ 25 g/day of fibre advised?", "answer": "Yes — at least 25 g (µ-level détail).",
             "page": 12, "answerable": True, "quality_score": None, "nested": {"list": [1, 2.5, "三"]}}]
    path = save_jsonl(rows, tmp_path / "out" / "data.jsonl")
    assert load_jsonl(path) == rows
    assert "µ" in path.read_text(encoding="utf-8"), "ensure_ascii=False keeps characters readable"


def test_clean_pairs_exports_only_public_fields():
    pair = {f: 1 for f in EXPORT_FIELDS} | {"chunk_text": "secret passage", "scores": {}, "judge_note": ""}
    assert list(clean_pairs([pair])[0]) == list(EXPORT_FIELDS)


def test_chatml_format_with_and_without_context():
    pair = {"question": "Q?", "answer": "A.", "chunk_text": "Passage."}
    plain = to_chatml_format([pair])[0]["messages"]
    assert plain == [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}]
    ctx = to_chatml_format([pair], include_context=True)[0]["messages"]
    assert ctx[0]["content"].startswith("Context:\nPassage.") and ctx[0]["content"].endswith("Question: Q?")


# ---------------------------------------------------------------------------
# Judge agreement
# ---------------------------------------------------------------------------
def test_cohen_kappa_matches_hand_computed_value():
    # groundedness: human [0,1,2,2,1,0] vs judge [0,1,2,1,1,0]
    # p_o = 5/6; marginals human (2,2,2), judge (2,3,1) -> p_e = (4+6+2)/36 = 1/3
    # kappa = (5/6 - 1/3) / (1 - 1/3) = 0.75
    human_g, judge_g = [0, 1, 2, 2, 1, 0], [0, 1, 2, 1, 1, 0]
    humans = [{"id": str(i), "groundedness": g, "specificity": 2, "completeness": 2} for i, g in enumerate(human_g)]
    judges = {str(i): {"groundedness": g, "specificity": 2, "completeness": 2} for i, g in enumerate(judge_g)}
    result = agreement.compute_kappa(humans, judges, min_score=4, floors=FLOORS)
    assert result["n"] == 6
    assert result["criteria"]["groundedness"]["kappa"] == pytest.approx(0.75, abs=1e-4)
    assert result["criteria"]["groundedness"]["agreement"] == pytest.approx(5 / 6, abs=1e-4)
    # keep decision (groundedness floor 2): human keep [0,0,1,1,0,0], judge keep [0,0,1,0,0,0]
    # p_o = 5/6; p_e = (4/6)(5/6) + (2/6)(1/6) = 22/36 -> kappa = (30/36 - 22/36) / (14/36) = 8/14
    assert result["decision"]["kappa"] == pytest.approx(8 / 14, abs=1e-4)
    assert result["decision"]["confusion_matrix"] == [[4, 0], [1, 1]]
    assert result["criteria"]["specificity"]["kappa"] is None  # identical constant labels: undefined


def test_constant_human_labels_are_flagged_as_uninformative():
    judges = {str(i): {"groundedness": 2, "specificity": 2, "completeness": 2 if i % 3 else 1} for i in range(10)}
    humans = [{"id": str(i), "groundedness": 1, "specificity": 1, "completeness": 1} for i in range(10)]
    result = agreement.compute_kappa(humans, judges, 4, FLOORS)
    assert result["warnings"] and "same human scores (1/1/1)" in result["warnings"][0]
    varied = [dict(h, groundedness=2 if i % 2 else 1) for i, h in enumerate(humans)]
    assert all("same human scores" not in w for w in agreement.compute_kappa(varied, judges, 4, FLOORS)["warnings"])


def test_interpret_kappa_bands():
    assert [agreement.interpret_kappa(k) for k in (0.9, 0.7, 0.5, 0.2, None)] == \
        ["strong", "substantial", "moderate", "weak", "undefined"]


def test_labelling_sample_is_stratified_blind_and_protects_existing_labels(tmp_path):
    pairs = [_judged(t) | {"id": f"p{i}", "chunk_text": "passage"}
             for i, t in enumerate([(0, 1, 1), (1, 1, 1), (2, 1, 0), (2, 1, 1), (2, 2, 1), (2, 2, 2)] * 5)]
    path = tmp_path / "human_labels.json"
    agreement.sample_for_labelling(pairs, n=10, seed=1, path=path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    items = doc["items"]
    assert len(items) == 10 and all(i["groundedness"] is None for i in items)
    assert "scores" not in items[0] and "quality_score" not in items[0], "labeller must not see judge scores"
    bands = {agreement._band(next(p["quality_score"] for p in pairs if p["id"] == i["id"])) for i in items}
    assert len(bands) >= 4, "sample must span score bands"
    items[0].update(groundedness=2, specificity=2, completeness=2)
    agreement.save_labels(doc, path)
    with pytest.raises(agreement.LabelsExistError):
        agreement.sample_for_labelling(pairs, n=10, seed=1, path=path)


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
def test_upload_validation_rejects_too_large_file_and_non_pdf():
    limit = CONFIG.web.max_upload_mb
    too_big = UploadedPDF("big.pdf", b"%PDF-1.7\n" + b"0" * int((limit + 1) * 1024 * 1024))
    not_pdf = UploadedPDF("notes.txt", b"plain text")
    renamed = UploadedPDF("fake.pdf", b"PK\x03\x04 this is really a zip/docx")
    ok = UploadedPDF("guideline.pdf", b"%PDF-1.7\n%...minimal...")
    assert validate_uploads([ok]) == []
    assert any("MB" in e for e in validate_uploads([too_big]))
    assert any("not a PDF" in e for e in validate_uploads([not_pdf]))
    assert any("content is not a PDF" in e for e in validate_uploads([renamed]))
    assert any("Too many files" in e for e in validate_uploads([ok] * (CONFIG.web.max_files_per_run + 1)))
    assert validate_uploads([]) == ["Please upload at least one PDF."]


def test_safe_filename_strips_paths():
    assert safe_filename("..\\..\\evil/../x.pdf") == "x.pdf"
    assert safe_filename("my file (1).pdf") == "my file _1_.pdf"


def test_upload_replaces_the_previous_runs_pdfs_only(tmp_path):
    # One pipeline: the input folder must hold exactly the latest upload, nothing from the run before.
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "old.pdf").write_bytes(b"%PDF-1.4 old")
    (folder / "notes.txt").write_text("not a PDF, left alone", encoding="utf-8")
    pdf = b"%PDF-1.4 new"
    replace_pdfs([UploadedPDF("a.pdf", pdf), UploadedPDF("A.pdf", pdf), UploadedPDF("../evil/b.pdf", pdf)], folder)
    assert {p.name for p in folder.glob("*.pdf")} == {"a.pdf", "A_2.pdf", "b.pdf"}
    assert (folder / "notes.txt").is_file() and not (folder / "old.pdf").exists()


def test_new_dataset_moves_old_labels_aside_but_identical_rerun_keeps_them(tmp_path):
    scores = {"groundedness": 2, "specificity": 2, "completeness": 2}
    old = [{"id": "p1", "answer": "A one.", "scores": scores}, {"id": "p2", "answer": "A two.", "scores": scores}]
    save_jsonl(old, tmp_path / "judged_pairs.jsonl")
    labels = {"items": [{"id": "p1", "groundedness": 2, "specificity": 1, "completeness": 2}]}
    (tmp_path / "human_labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (tmp_path / "experiments.md").write_text("tables", encoding="utf-8")

    assert archive_stale_files(tmp_path, old) is None, "an identical re-run must keep the labels in place"
    assert (tmp_path / "human_labels.json").is_file()

    # Shared result files record project-relative paths, never this machine's folder layout.
    assert project_relative(PROJECT_ROOT / "data" / "raw") == "data/raw"
    assert project_relative(tmp_path) == str(tmp_path)

    moved = archive_stale_files(tmp_path, [{"id": "p9", "answer": "New.", "scores": scores}])
    assert moved is not None
    folder, names = moved
    assert set(names) == {"human_labels.json", "experiments.md"}
    assert not (tmp_path / "human_labels.json").exists()
    assert json.loads((folder / "human_labels.json").read_text(encoding="utf-8")) == labels, "gold labels are kept"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_overrides_are_typed_and_typos_fail_loudly():
    cfg = with_overrides(CONFIG, ["validate.min_quality_score=5", "run.max_chunks=null"])
    assert cfg.validate.min_quality_score == 5 and cfg.run.max_chunks is None
    assert CONFIG.validate.min_quality_score == 4, "overrides must not mutate the base config"
    with pytest.raises(ConfigError):
        with_overrides(CONFIG, ["validate.min_qualty_score=5"])
    with pytest.raises(ConfigError):
        with_overrides(CONFIG, ["validate.min_quality_score=9"])


def test_public_mode_fails_safe_unless_local_server_and_local_visitor():
    # "auto": private only when the server listens on localhost AND the page was opened from localhost.
    assert CONFIG.web.public_mode == "auto"
    assert not is_public("auto", "localhost", "localhost:8501")
    assert not is_public("auto", "127.0.0.1", "[::1]:8501")
    assert is_public("auto", "", "localhost:8501") and is_public("auto", None, None)
    assert is_public("auto", "0.0.0.0", "localhost:8501")
    assert is_public("auto", "localhost", "my-app.streamlit.app"), "a proxy to a localhost-bound app is public"
    assert is_public("auto", "localhost", None)
    assert is_public(True, "localhost", "localhost:8501") and not is_public(False, "0.0.0.0", "example.org")
    assert with_overrides(CONFIG, ["web.public_mode=true"]).web.public_mode is True
    with pytest.raises(ConfigError):
        with_overrides(CONFIG, ["web.public_mode=sometimes"])
