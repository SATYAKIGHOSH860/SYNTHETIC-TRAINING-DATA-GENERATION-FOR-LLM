"""The one pipeline implementation shared by the CLI (main.py) and the web UI.

ingest -> generate -> validate -> deduplicate -> export

This module never calls ``sys.exit()`` and never imports streamlit, so it is
usable from both contexts. Expected failures (no PDFs, unreadable PDFs,
missing key, bad model id, exhausted quota) are raised as ``PipelineError``
with a message written for the person running it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from src.config import CONFIG, Config, ConfigError, require_api_key, resolve_path
from src.deduplicate import deduplicate
from src.export import build_stats, print_report, write_outputs
from src.generate import generate_from_chunks
from src.ingest import NoReadablePDFError, chunk_documents, list_pdfs, load_pdfs, select_chunks
from src.llm import FatalLLMError
from src.validate import validate_all

ProgressCallback = Callable[[int, int, str], None]


class PipelineError(RuntimeError):
    """A failure with a clear, user-facing explanation (safe to show in the UI)."""


class PipelineCancelled(RuntimeError):
    """Raised from a progress_callback to stop a run (the UI's Stop button).

    The callback runs after every chunk and every judged group, so a run stops
    within one API call of the request. Generation and validation flush their
    checkpoints on the way out, so the next run with resume=True continues
    where this one stopped. Output files are only written at the very end, so a
    stopped run never leaves a half-written dataset.
    """


def _header(title: str) -> None:
    print(f"\n{'-' * 60}\n {title}\n{'-' * 60}", flush=True)


def run_pipeline(
    pdf_dir: str | Path,
    output_dir: str | Path,
    cfg: Config = CONFIG,
    max_chunks: int | None = None,
    progress_callback: ProgressCallback | None = None,
    resume: bool = False,
    *,
    cache_dir: str | Path | None = None,
    api_key: str | None = None,
    warnings: list[str] | None = None,
    refill: bool = False,
) -> dict[str, Any]:
    """Run every stage and write all output files; return the stats dict.

    ``max_chunks`` overrides ``cfg.run.max_chunks`` (the web UI passes its demo
    cap). ``cache_dir`` holds checkpoints (default ``cfg.paths.cache_dir``; web
    jobs use their own folder so concurrent users never collide).
    ``api_key`` lets a web user supply their own key; it is held in memory only.
    ``warnings`` (optional list) receives non-fatal problems such as a
    scanned PDF that had to be skipped. ``progress_callback`` may raise
    PipelineCancelled to stop the run cleanly. ``refill`` (implies ``resume``) keeps
    complete chunks from an older prompt version and regenerates the rest
    (see generate_from_chunks).
    """
    resume = resume or refill
    started = time.time()
    warnings = warnings if warnings is not None else []
    notify = progress_callback or (lambda current, total, stage: None)
    pdf_dir, output_dir = resolve_path(pdf_dir), resolve_path(output_dir)
    cache_dir = resolve_path(cache_dir or cfg.paths.cache_dir)
    timings: dict[str, float] = {}

    try:
        key = require_api_key(api_key)
    except ConfigError as exc:
        raise PipelineError(str(exc)) from None

    # --- 1. Ingest -------------------------------------------------------------
    _header("1/5  INGEST: PDFs -> chunks")
    notify(0, 1, "ingesting")
    t = time.time()
    try:
        pdf_files = list_pdfs(pdf_dir)
        docs = load_pdfs(pdf_dir, cfg, warnings)
    except FileNotFoundError as exc:
        raise PipelineError(str(exc)) from None
    except NoReadablePDFError as exc:
        raise PipelineError(str(exc)) from None
    chunks = chunk_documents(docs, cfg)
    if not chunks:
        raise PipelineError("The PDFs contained no usable text passages after cleaning.")
    limit = max_chunks if max_chunks is not None else cfg.run.max_chunks
    selected = select_chunks(chunks, limit)
    print(f"Processing {len(selected)} of {len(chunks)} chunks"
          + (f" (cap {limit})" if limit and limit < len(chunks) else ""))
    timings["ingest"] = round(time.time() - t, 1)
    notify(1, 1, "ingesting")

    # --- 2. Generate -----------------------------------------------------------
    _header("2/5  GENERATE: chunks -> Q&A pairs")
    gen_report: dict[str, Any] = {}
    try:
        raw_pairs = generate_from_chunks(selected, cfg, progress_callback, cache_dir=cache_dir,
                                         resume=resume, refill=refill, api_key=key, report=gen_report)
    except FatalLLMError as exc:
        raise PipelineError(str(exc)) from None
    if not raw_pairs:
        raise PipelineError(
            "The model produced no usable question-answer pairs. Check the log above: every chunk "
            "either failed or was judged to contain no substantive content."
        )

    # --- 3. Validate -----------------------------------------------------------
    _header("3/5  VALIDATE: LLM-as-judge scoring")
    val_report: dict[str, Any] = {}
    try:
        kept, rejected = validate_all(raw_pairs, cfg, progress_callback, cache_dir=cache_dir,
                                      output_dir=output_dir, resume=resume, api_key=key, report=val_report)
    except FatalLLMError as exc:
        raise PipelineError(str(exc)) from None

    # --- 4. Deduplicate --------------------------------------------------------
    _header("4/5  DEDUPLICATE: semantic similarity")
    notify(0, 1, "deduplicating")
    t = time.time()
    dedup_report: dict[str, Any] = {}
    final, duplicates = deduplicate(kept, cfg, output_dir=output_dir, report=dedup_report)
    timings["deduplicate"] = round(time.time() - t, 1)
    notify(1, 1, "deduplicating")

    # --- 5. Export -------------------------------------------------------------
    _header("5/5  EXPORT: dataset, evidence files, statistics")
    notify(0, 1, "exporting")
    timings["generate"] = gen_report.get("seconds", 0.0)
    timings["validate"] = val_report.get("seconds", 0.0)
    gen_usage, val_usage = gen_report.get("usage", {}), val_report.get("usage", {})
    wall = round(time.time() - started, 1)
    run_info = {
        "run": {
            "pdf_dir": str(pdf_dir),
            "pdf_files": [p.name for p in pdf_files],
            "pages_loaded": len(docs),
            "chunks_total": len(chunks),
            "chunks_processed": len(selected),
            "max_chunks": limit,
            "chunks_failed": gen_report.get("chunks_failed", 0),
            "chunks_skipped_no_content": gen_report.get("chunks_empty", 0),
            "chunks_fully_filtered": gen_report.get("chunks_filtered", 0),
            "chunks_carried_over": gen_report.get("chunks_carried_over", 0),
            "pairs_by_prompt_version": gen_report.get("pairs_by_prompt_version", {}),
            "generator_model": cfg.model.id,
            "judge_model": cfg.model.judge_id,
            "api_calls": gen_usage.get("api_calls", 0) + val_usage.get("api_calls", 0),
            "tokens_generation": gen_usage.get("total_tokens", 0),
            "tokens_validation": val_usage.get("total_tokens", 0),
            "stage_seconds": timings,
            "wall_seconds": wall,
            "resumed": resume,
            "refilled": refill,
            "warnings": list(warnings),
        }
    }
    judged = kept + rejected
    stats = build_stats(raw_pairs=raw_pairs, judged=judged, kept=kept, final=final,
                        duplicates=duplicates, cfg=cfg, extra=run_info)
    paths = write_outputs(output_dir, kept=kept, rejected=rejected, final=final, stats=stats, cfg=cfg)
    for name, path in paths.items():
        print(f"  wrote {path}")
    print_report(stats)
    notify(1, 1, "done")
    return stats


if __name__ == "__main__":
    # Smallest end-to-end check: three chunks into a scratch folder.
    scratch = resolve_path(CONFIG.paths.cache_dir) / "pipeline_smoke"
    result = run_pipeline(CONFIG.paths.raw_dir, scratch / "output", CONFIG, max_chunks=3,
                          cache_dir=scratch / "cache")
    print(f"\nSmoke run finished: {result['final_count']} pairs in {result['run']['wall_seconds']}s")
