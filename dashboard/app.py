"""Streamlit dashboard: explore the generated dataset and generate new ones from uploaded PDFs.

Run from the project root:
    streamlit run dashboard/app.py

This file only presents results. Every piece of pipeline logic lives in src/
and is called from here, never reimplemented.
"""

from __future__ import annotations

import difflib
import html
import io
import json
import math
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Synthetic Training Data Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from src.config import CONFIG, has_api_key, is_public, resolve_path, with_overrides  # noqa: E402
except Exception as exc:  # config.yaml missing or invalid
    st.error(f"Configuration problem: {exc}")
    st.stop()

from src import agreement  # noqa: E402
from src.export import DISCLAIMER, load_jsonl  # noqa: E402
from src.llm import redact  # noqa: E402
from src.pipeline import PipelineCancelled, PipelineError, run_pipeline  # noqa: E402
from src.uploads import UploadedPDF, new_job_dir, save_uploads, validate_uploads  # noqa: E402

OUTPUT_DIR = resolve_path(CONFIG.paths.output_dir)
# Public mode (config web.public_mode): results are read-only, no pipeline runs, no labelling, and
# uploads use the visitor's own key only. "auto" turns it on unless the server listens only on
# localhost and this session's page was opened from localhost.
PUBLIC = is_public(CONFIG.web.public_mode, st.get_option("server.address"), st.context.headers.get("Host"))

# ---------------------------------------------------------------------------
# Visual system: validated categorical slots (dataviz reference palette),
# recessive chrome, one accent. Text never wears a series colour.
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DE_EMPHASIS = "#c3c2b7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD_TEXT, CRITICAL = "#006300", "#d03b3b"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

st.markdown(
    """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .app-title {font-size: 1.85rem; font-weight: 700; color: #0b0b0b; margin: 0; line-height: 1.2;}
  .app-sub {color: #52514e; font-size: 0.98rem; margin: 0.25rem 0 0.9rem;}
  .disclaimer {border: 1px solid #ecd9ae; background: #fdf8ec; color: #5a4512; border-radius: 10px;
               padding: 0.6rem 0.9rem; font-size: 0.86rem; margin-bottom: 0.4rem;}
  .disclaimer b {color: #3f300b;}
  .tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 0.75rem; margin: 0.4rem 0 1.1rem;}
  .tile {background: #ffffff; border: 1px solid #e1e0d9; border-radius: 12px; padding: 0.85rem 1rem 0.8rem;}
  .tile-label {color: #52514e; font-size: 0.8rem; font-weight: 500;}
  .tile-value {color: #0b0b0b; font-size: 1.7rem; font-weight: 650; line-height: 1.25; margin-top: 0.2rem;}
  .tile-sub {color: #898781; font-size: 0.76rem; margin-top: 0.1rem;}
  .section-title {font-size: 1.08rem; font-weight: 650; color: #0b0b0b; margin: 1.1rem 0 0.1rem;}
  .section-sub {color: #52514e; font-size: 0.88rem; margin: 0 0 0.6rem;}
  .card {border: 1px solid #e1e0d9; border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 0.65rem; background: #ffffff;}
  .card-q {font-weight: 600; color: #0b0b0b; margin-bottom: 0.3rem;}
  .card-a {color: #26251f; margin-bottom: 0.45rem;}
  .meta {color: #898781; font-size: 0.78rem;}
  .reason {color: #52514e; font-size: 0.84rem; margin-top: 0.35rem;}
  .reason b {color: #0b0b0b;}
  .pill {display: inline-block; padding: 0.05rem 0.55rem; border-radius: 999px; font-size: 0.74rem; font-weight: 600;
         border: 1px solid #e1e0d9; color: #52514e; background: #f6f5f2; margin: 0 0.3rem 0.25rem 0; white-space: nowrap;}
  .pill-good {color: #006300; background: #eef8ee; border-color: #cfe8cf;}
  .pill-bad {color: #9f2626; background: #fcefef; border-color: #f1d0d0;}
  .pill-blue {color: #1c5cab; background: #edf4fd; border-color: #cde2fb;}
  .passage {border-left: 3px solid #cde2fb; padding: 0.45rem 0.8rem; color: #52514e; font-size: 0.84rem;
            background: #f8fafd; border-radius: 0 8px 8px 0; white-space: pre-wrap; margin-top: 0.4rem;}
  .scores {display: flex; flex-wrap: wrap; gap: 0.35rem 1.1rem; margin-top: 0.2rem;}
  .score {display: flex; align-items: center; gap: 0.45rem; font-size: 0.78rem; color: #52514e;}
  .track {width: 64px; height: 6px; background: #e6eefa; border-radius: 3px; overflow: hidden;}
  .fill {height: 6px; background: #2a78d6; border-radius: 3px;}
  .vs {display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem;}
  .vs-label {font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: #898781; margin-bottom: 0.2rem;}
  .empty {border: 1px dashed #d6d4cc; border-radius: 12px; padding: 1.4rem; color: #52514e; background: #ffffff;}
  .empty code {background: #f3f2ee; padding: 0.1rem 0.35rem; border-radius: 5px; white-space: nowrap;}
  @media (max-width: 700px) {.vs {grid-template-columns: 1fr;} .app-title {font-size: 1.5rem;}}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------
def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def fmt_pct(value: float | None, digits: int = 0) -> str:
    return "–" if value is None else f"{value * 100:.{digits}f}%"


def fmt_score(value: float | None) -> str:
    """Two decimals, truncated rather than rounded, so 5.995 never displays as a perfect 6.00."""
    return "–" if value is None else f"{math.floor(value * 100) / 100:.2f}"


def tiles(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(
        f'<div class="tile"><div class="tile-label">{esc(label)}</div>'
        f'<div class="tile-value">{esc(value)}</div><div class="tile-sub">{esc(sub)}</div></div>'
        for label, value, sub in items
    )
    st.markdown(f'<div class="tiles">{cells}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{esc(title)}</div>'
                + (f'<div class="section-sub">{esc(subtitle)}</div>' if subtitle else ""),
                unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    return f'<span class="pill {kind}">{esc(text)}</span>'


def status_pill(status: str) -> str:
    return {
        "kept": pill("✓ Kept", "pill-good"),
        "rejected": pill("✕ Rejected", "pill-bad"),
        "duplicate": pill("⧉ Duplicate", ""),
    }.get(status, pill(status))


def score_bars(scores: dict[str, int] | None) -> str:
    if not scores:
        return ""
    rows = "".join(
        f'<div class="score">{esc(name.title())}<div class="track"><div class="fill" '
        f'style="width:{int(value) / 2 * 100:.0f}%"></div></div>{int(value)}/2</div>'
        for name, value in scores.items()
    )
    return f'<div class="scores">{rows}</div>'


def style_fig(fig: go.Figure, height: int = 300, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="none", height=height, margin=dict(l=8, r=16, t=16, b=8),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(family=FONT, color=INK, size=13)),
        showlegend=legend, bargap=0.4,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK_2)),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, linewidth=1, ticks="", tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2, size=12), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, linecolor=AXIS, ticks="", tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2, size=12), zeroline=False)
    return fig


def show_chart(fig: go.Figure, table: pd.DataFrame | None = None, key: str = "") -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"chart_{key}")
    if table is not None:
        with st.expander("View as table"):
            st.dataframe(table, hide_index=True, width="stretch")


def hbar(labels: list[str], values: list[float], *, color: str = BLUE, hover: str = "%{y}: %{x}",
         text: list[str] | None = None, xtitle: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=color, cornerradius=4),
        text=text, textposition="outside", cliponaxis=False, textfont=dict(color=INK_2, size=12),
        hovertemplate=hover + "<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK_2))
    # Headroom so the value labels outside the bar ends are never clipped.
    fig.update_xaxes(showgrid=True, gridcolor=GRID, title_text=xtitle,
                     range=[0, (max(values) if values else 1) * 1.22])
    return style_fig(fig, height=max(150, 46 * len(labels) + 50))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
OUTPUT_FILES = ("pipeline_stats.json", "synthetic_dataset.jsonl", "judged_pairs.jsonl", "rejected_pairs.jsonl",
                "duplicate_pairs.jsonl", "unanswerable_pairs.jsonl", "synthetic_dataset_chatml.jsonl")


def files_signature(folder: Path) -> tuple:
    """(name, mtime) of each output file, so the cache refreshes after a new run."""
    return tuple((name, (folder / name).stat().st_mtime if (folder / name).exists() else 0.0)
                 for name in OUTPUT_FILES)


@st.cache_data(show_spinner=False)
def load_results(folder: str, signature: tuple) -> dict[str, Any] | None:
    out = Path(folder)
    stats_path = out / "pipeline_stats.json"
    if not stats_path.is_file():
        return None
    with open(stats_path, encoding="utf-8") as fh:
        stats = json.load(fh)

    def rows(name: str) -> list[dict[str, Any]]:
        path = out / name
        return load_jsonl(path) if path.is_file() else []

    return {
        "stats": stats,
        "dataset": rows("synthetic_dataset.jsonl"),
        "judged": rows("judged_pairs.jsonl"),
        "rejected": rows("rejected_pairs.jsonl"),
        "duplicates": rows("duplicate_pairs.jsonl"),
        "unanswerable": rows("unanswerable_pairs.jsonl"),
    }


def read_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.is_file() else b""


# Every file a run can produce, in the order a reader needs them:
# (file name, title, what it contains, how it is created if missing)
OUTPUT_CATALOG = (
    ("synthetic_dataset.jsonl", "Final dataset (JSONL)",
     "The clean training pairs: question, answer, source, page, type, answerable, quality score.",
     "python main.py"),
    ("synthetic_dataset.csv", "Final dataset (CSV, opens in Excel)",
     "The same pairs as a spreadsheet, for reading and sharing.", "python main.py"),
    ("synthetic_dataset_chatml.jsonl", "Fine-tuning format (ChatML)",
     "Each pair as chat messages with its source passage, ready for TRL, Axolotl or Hugging Face trainers.",
     "python main.py"),
    ("judged_pairs.jsonl", "Full audit trail",
     "Every judged pair with its sub-scores, the judge's note and its status: kept, rejected or duplicate.",
     "python main.py"),
    ("rejected_pairs.jsonl", "Rejected pairs", "Pairs the judge rejected, each with the judge's reason.",
     "python main.py"),
    ("duplicate_pairs.jsonl", "Duplicates removed",
     "Each removed pair beside the pair it duplicated, with the similarity score.", "python main.py"),
    ("unanswerable_pairs.jsonl", "Unanswerable (abstention) pairs",
     "Questions the source does not answer, with the verification verdict.", "python main.py"),
    ("pipeline_stats.json", "Run statistics", "Every count and distribution, plus the exact settings used.",
     "python main.py"),
    ("human_labels.json", "Your human labels", "The 40-pair blind sample with the scores you entered.",
     "the Judge reliability tab"),
    ("judge_agreement.json", "Judge vs human agreement", "Cohen's kappa results, written when all pairs are labelled.",
     "labelling all pairs in the Judge reliability tab"),
    ("experiments.md", "Experiment tables (Markdown)", "The four experiment tables, ready to paste into the README.",
     "python -m src.experiments"),
    ("experiments.json", "Experiment data (JSON)", "The same experiment results in machine-readable form.",
     "python -m src.experiments"),
)
MIME = {".jsonl": "application/jsonl", ".json": "application/json", ".csv": "text/csv", ".md": "text/markdown"}


def dataset_csv(folder: Path) -> bytes:
    """The final dataset as CSV. UTF-8 with BOM so Excel shows µ, ≥ and accents correctly."""
    path = folder / "synthetic_dataset.jsonl"
    if not path.is_file():
        return b""
    return pd.DataFrame(load_jsonl(path)).to_csv(index=False).encode("utf-8-sig")


def output_bytes(folder: Path, name: str) -> bytes:
    return dataset_csv(folder) if name == "synthetic_dataset.csv" else read_bytes(folder / name)


def zip_outputs(folder: Path) -> bytes:
    """Every output file that exists (plus the CSV version of the dataset) in one ZIP.

    Empty files are included too: an empty rejected_pairs.jsonl is a result ("nothing rejected").
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, *_ in OUTPUT_CATALOG:
            source = "synthetic_dataset.jsonl" if name == "synthetic_dataset.csv" else name
            if (folder / source).is_file():
                zf.writestr(name, output_bytes(folder, name))
    return buffer.getvalue()


def human_size(n: int) -> str:
    return f"{n / 1024:.1f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


def row_count(folder: Path, name: str) -> str:
    """Records in a file: lines for JSONL, items for the labels file, blank otherwise."""
    if name == "synthetic_dataset.csv":
        name = "synthetic_dataset.jsonl"
    path = folder / name
    if name.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            return f"{sum(1 for line in fh if line.strip()):,} rows"
    if name == "human_labels.json":
        items = agreement.load_labels(path).get("items", [])
        return f"{sum(1 for i in items if agreement.is_complete(i))} of {len(items)} labelled"
    return ""


def no_results_message() -> None:
    st.markdown(
        '<div class="empty"><b>No results yet.</b> The pipeline has not been run, so there is nothing to show '
        'in this tab. From the project root run <code>python main.py</code> (add <code>--max-chunks 5</code> '
        'for a quick test), then refresh this page. You can also try the <b>Generate from your PDFs</b> tab.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-title">Synthetic Training Data Studio</div>'
    '<div class="app-sub">WHO guideline PDFs → LLM-generated Q&amp;A → LLM-judge validation → semantic '
    'deduplication → fine-tuning dataset, with every rejected and removed pair kept as evidence.</div>'
    f'<div class="disclaimer"><b>Not medical advice.</b> {esc(DISCLAIMER)}</div>',
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def warm_up() -> threading.Thread:
    """Load the embedding model (and the slow langchain/torch imports) once per server,
    in the background, so the page renders instantly and the first job starts warm."""
    def load() -> None:
        try:
            import langchain_text_splitters  # noqa: F401  (pulls in torch/transformers)
            from src.deduplicate import get_embedding_model
            get_embedding_model()
        except Exception:
            traceback.print_exc()

    thread = threading.Thread(target=load, name="warm-up", daemon=True)
    thread.start()
    return thread


warm_up()
results = load_results(str(OUTPUT_DIR), files_signature(OUTPUT_DIR))


# ---------------------------------------------------------------------------
# Background runs with a Stop button (shared by the full pipeline run and upload jobs)
# ---------------------------------------------------------------------------
STAGE_LABELS = {
    "queued": "Starting…", "ingesting": "Reading and chunking PDFs", "generating": "Generating questions",
    "validating": "Judging quality", "deduplicating": "Removing semantic duplicates",
    "exporting": "Writing dataset files", "done": "Finished",
}
STAGE_WEIGHTS = {"queued": (0, 0), "ingesting": (0, 3), "generating": (3, 50), "validating": (53, 42),
                 "deduplicating": (95, 3), "exporting": (98, 2), "done": (100, 0)}


def overall_progress(job: dict[str, Any]) -> float:
    base, span = STAGE_WEIGHTS.get(job["stage"], (0, 0))
    return min(1.0, (base + span * job["current"] / max(job["total"], 1)) / 100)


def stage_text(job: dict[str, Any]) -> str:
    stage = job["stage"]
    label = STAGE_LABELS.get(stage, stage)
    if stage == "generating":
        return f"{label}: chunk {job['current']}/{job['total']}"
    if stage == "validating":
        return f"{label}: pair {job['current']}/{job['total']}"
    return label


def elapsed_text(job: dict[str, Any]) -> str:
    seconds = int((job.get("finished") or time.time()) - job["started"])
    return f"{seconds // 60}m {seconds % 60:02d}s"


def make_progress(job: dict[str, Any]):
    """Progress callback for run_pipeline that also carries the Stop button's request.

    Stop is cooperative: the pipeline calls this after every chunk and judged
    group, so raising PipelineCancelled here ends the run within one API call,
    with finished work checkpointed and no partial dataset written.
    """
    def progress(current: int, total: int, stage: str) -> None:
        # "done" is reported after every file is written: too late to cancel, and nothing left to stop.
        if job["stop"].is_set() and stage != "done":
            raise PipelineCancelled("Stopped by user")
        job.update(stage=stage, current=current, total=max(total, 1))
    return progress


# launcher.py reads this file: its Stop asks for confirmation only while a run or upload job is in progress.
BUSY_FILE = resolve_path(CONFIG.paths.cache_dir) / "dashboard.busy"


@st.cache_resource
def work_tracker() -> dict[str, Any]:
    """Count of runs and upload jobs in progress in this server process."""
    BUSY_FILE.unlink(missing_ok=True)  # left behind if the previous server was killed mid-run
    return {"count": 0, "lock": threading.Lock()}


def work_started():
    """Mark one run or job as in progress; the worker calls the returned function when it ends.

    Called from the script thread, so the worker never touches st.cache_resource itself.
    """
    tracker = work_tracker()
    with tracker["lock"]:
        tracker["count"] += 1
        BUSY_FILE.parent.mkdir(parents=True, exist_ok=True)
        BUSY_FILE.touch()

    def finished() -> None:
        with tracker["lock"]:
            tracker["count"] -= 1
            if tracker["count"] <= 0:
                BUSY_FILE.unlink(missing_ok=True)
    return finished


work_tracker()


@st.cache_resource
def pipeline_runner() -> dict[str, Any]:
    """The single full-pipeline run (data/raw -> data/output), shared by every browser tab."""
    return {"status": "idle", "stage": "queued", "current": 0, "total": 1, "started": None,
            "finished": None, "error": None, "stats": None, "stop": threading.Event(), "chunks": None}


def start_pipeline_run(max_chunks: int | None, fresh: bool) -> None:
    """Only the worker thread changes status after this; the UI only sets the stop flag."""
    runner = pipeline_runner()
    if runner["status"] == "running":
        return
    runner.update(status="running", stage="queued", current=0, total=1, started=time.time(), finished=None,
                  error=None, stats=None, stop=threading.Event(), chunks=max_chunks)
    progress = make_progress(runner)
    finished = work_started()

    def work() -> None:
        try:
            runner["stats"] = run_pipeline(CONFIG.paths.raw_dir, CONFIG.paths.output_dir, CONFIG,
                                           max_chunks=max_chunks, progress_callback=progress, resume=not fresh)
            runner["status"] = "done"
        except PipelineCancelled:
            runner["status"] = "stopped"
        except PipelineError as exc:
            runner.update(status="error", error=str(exc))
        except Exception as exc:  # never show a raw traceback in the browser
            traceback.print_exc()
            runner.update(status="error", error=redact(f"Unexpected error ({type(exc).__name__}): {exc}", None))
        finally:
            runner["finished"] = time.time()
            finished()

    threading.Thread(target=work, name="pipeline-run", daemon=True).start()


run_active = pipeline_runner()["status"] == "running"


@st.fragment(run_every=1.5 if run_active else None)
def pipeline_control() -> None:
    """Sidebar Generate / Stop generating for the full pipeline, with live progress while it runs.

    "Start" and "Stop" are reserved for the desktop launcher (open / shut down the whole app).
    """
    runner = pipeline_runner()
    status = runner["status"]
    st.markdown("### Generate the dataset")
    if status == "running":
        stopping = runner["stop"].is_set()
        label = "Stopping after the current step…" if stopping else stage_text(runner) + "…"
        st.markdown(f":green[●] **Generating** · {elapsed_text(runner)}  \n{label}")
        st.progress(overall_progress(runner))
        if st.button("■ Stop generating", width="stretch", disabled=stopping, key="run_stop"):
            runner["stop"].set()
            st.rerun(scope="fragment")
        st.caption("Finished work is saved. Press Generate dataset again later to continue from where it stopped.")
        return
    if run_active:  # it finished since the page last drew: refresh every tab with the new results
        st.cache_data.clear()
        st.session_state.run_flash = {"done": "Generation finished: the tabs now show the new results.",
                                      "stopped": "Generation stopped. Press Generate dataset to continue it later.",
                                      "error": "Generation could not finish: see the sidebar."}.get(status)
        st.rerun(scope="app")
    if status == "done" and runner["stats"]:
        st.success(f"Last run finished in {elapsed_text(runner)}: {runner['stats']['final_count']} pairs.")
    elif status == "stopped":
        st.info(f"Last run was stopped after {elapsed_text(runner)}. "
                "Generate dataset continues from saved progress.")
    elif status == "error":
        st.error(runner["error"])
    chunks = st.number_input("Chunks to process", min_value=1, max_value=5000, step=10,
                             value=int(CONFIG.run.max_chunks or 100), key="run_chunks",
                             help="More chunks give more pairs but take longer: roughly 15 seconds per chunk "
                                  "on the free API tier (100 chunks ≈ 25 minutes).")
    fresh = st.checkbox("Start over (discard saved progress)", key="run_fresh",
                        help="By default a run reuses finished work from earlier runs with the same settings.")
    if not has_api_key():
        st.warning("Add GROQ_API_KEY to .env to run the pipeline.")
    if st.button("▶ Generate dataset", type="primary", width="stretch", disabled=not has_api_key(),
                 key="run_start"):
        start_pipeline_run(int(chunks), fresh)
        st.rerun(scope="app")
    st.caption(f"Processes the PDFs in `{CONFIG.paths.raw_dir}/` and replaces the results in "
               f"`{CONFIG.paths.output_dir}/` when it finishes.")


if st.session_state.get("run_flash"):
    st.toast(st.session_state.pop("run_flash"), icon="ℹ️")

with st.sidebar:
    if PUBLIC:
        st.markdown("### Public demo")
        st.caption("This page shows a finished run of the pipeline, which you can explore and download. "
                   "Re-running it is switched off here; to try your own PDFs, use the Generate from your PDFs "
                   "tab with your own free Groq API key.")
    else:
        pipeline_control()
    st.divider()
    st.markdown("### Run details")
    if results:
        s, run = results["stats"], results["stats"].get("run", {})
        cfg_used = s.get("config", {})
        st.caption(f"Generated {s.get('generated_at', '')[:19].replace('T', ' ')} UTC")
        st.markdown(
            f"**Generator** `{run.get('generator_model', '?')}`  \n"
            f"**Judge** `{run.get('judge_model', '?')}`  \n"
            f"**Chunks** {run.get('chunks_processed', '?')} of {run.get('chunks_total', '?')}  \n"
            f"**Keep threshold** ≥ {cfg_used.get('validate', {}).get('min_quality_score', '?')}/6  \n"
            f"**Dedup threshold** {cfg_used.get('deduplicate', {}).get('similarity_threshold', '?')}  \n"
            f"**API calls** {run.get('api_calls', 0):,}  \n"
            f"**Tokens** {run.get('tokens_generation', 0) + run.get('tokens_validation', 0):,}  \n"
            f"**Wall time** {run.get('wall_seconds', 0) / 60:.1f} min"
            + (" (resumed from checkpoints)" if run.get("resumed") else "")
        )
        st.caption("API calls and tokens cover every pair in this dataset, including pairs reused from checkpoints.")
        with st.expander("Source documents"):
            for name in run.get("pdf_files", []):
                st.markdown(f"- {name}")
    else:
        st.caption("No pipeline run found in data/output yet.")
    if st.button("Reload results", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption("Pipeline code lives in `src/`; this app only presents it. "
               + ("Public mode: read-only results; uploads use each visitor's own API key." if PUBLIC else
                  "Server API key: " + ("configured" if has_api_key() else "not set")))

tab_overview, tab_quality, tab_evidence, tab_judge, tab_generate, tab_output = st.tabs(
    ["Pipeline overview", "Quality explorer", "Evidence", "Judge reliability", "Generate from your PDFs", "Output"]
)


# ---------------------------------------------------------------------------
# Tab 1: overview
# ---------------------------------------------------------------------------
def render_stat_tiles(stats: dict[str, Any]) -> None:
    avg = stats.get("average_quality_score")
    tiles([
        ("Final pairs", f"{stats['final_count']:,}", f"from {stats['raw_count']:,} generated"),
        ("Average quality", "–" if avg is None else f"{fmt_score(avg)} / 6", "answerable pairs in the dataset"),
        ("Pass rate", fmt_pct(stats["pass_rate"]), f"{stats['validated_count']:,} passed the judge"),
        ("Duplicates removed", f"{stats['duplicates_removed']:,}", f"{fmt_pct(stats.get('duplicate_rate'), 1)} of validated"),
        ("Unanswerable", f"{stats['unanswerable_count']:,}", "abstention examples kept"),
    ])


with tab_overview:
    if not results:
        no_results_message()
    else:
        stats = results["stats"]
        render_stat_tiles(stats)

        left, right = st.columns([1.05, 1], gap="large")
        with left:
            section("From raw generations to the final dataset",
                    "Each stage removes pairs for a stated reason; nothing is silently dropped.")
            stages = ["Generated", "Passed judge", "After dedup"]
            values = [stats["raw_count"], stats["validated_count"], stats["final_count"]]
            fig = go.Figure(go.Funnel(
                y=stages, x=values, textinfo="value+percent initial",
                textfont=dict(color="#ffffff", size=13), marker=dict(color=[BLUE_RAMP[1], BLUE_RAMP[2], BLUE_RAMP[3]]),
                connector=dict(fillcolor="#eef3fb", line=dict(width=0)),
                hovertemplate="%{y}: %{x} pairs (%{percentInitial:.0%} of generated)<extra></extra>",
            ))
            fig.update_yaxes(tickfont=dict(color=INK_2, size=13), gridcolor="rgba(0,0,0,0)")
            show_chart(style_fig(fig, height=260),
                       pd.DataFrame({"Stage": stages, "Pairs": values,
                                     "Share of generated": [fmt_pct(v / max(values[0], 1), 1) for v in values]}),
                       key="funnel")
        with right:
            section("Pairs per source document", "Final dataset, after validation and deduplication.")
            per_source = stats.get("per_source_counts", {})
            if per_source:
                names, counts = list(per_source), list(per_source.values())
                show_chart(hbar(names, counts, text=[str(c) for c in counts], hover="%{y}: %{x} pairs"),
                           pd.DataFrame({"Source": names, "Pairs": counts}), key="sources")

        left, right = st.columns(2, gap="large")
        with left:
            section("Question types", "Spread across the configured types; unanswerable pairs teach abstention.")
            qtypes = stats.get("question_type_distribution", {})
            if qtypes:
                names, counts = list(qtypes), list(qtypes.values())
                total = sum(counts) or 1
                show_chart(hbar(names, counts, text=[f"{c} ({c / total:.0%})" for c in counts],
                                hover="%{y}: %{x} pairs"),
                           pd.DataFrame({"Type": names, "Pairs": counts}), key="types")
        with right:
            div = stats.get("diversity", {})
            section("How questions open",
                    f"{div.get('unique_first_words', 0)} distinct opening words; the most common, "
                    f"“{div.get('top_first_word', '')}”, starts {fmt_pct(div.get('top_first_word_share'))} of questions.")
            dist = div.get("first_word_distribution", {})
            if dist:
                n_q = div.get("questions", 1) or 1
                words = list(dist)[:8]
                shares = [dist[w] / n_q for w in words]
                show_chart(hbar(words, [round(v * 100, 1) for v in shares], text=[fmt_pct(v) for v in shares],
                                hover="“%{y}”: %{x}% of questions", xtitle="% of questions"),
                           pd.DataFrame({"First word": words, "Questions": [dist[w] for w in words],
                                         "Share": [fmt_pct(v, 1) for v in shares]}), key="firstwords")

        run = stats.get("run", {})
        with st.expander("Run log: timings, usage and warnings"):
            secs = run.get("stage_seconds", {})
            st.dataframe(pd.DataFrame([
                {"Item": "Chunks processed / available", "Value": f"{run.get('chunks_processed')} / {run.get('chunks_total')}"},
                {"Item": "Chunks failed / no substantive content / fully filtered",
                 "Value": f"{run.get('chunks_failed', 0)} / {run.get('chunks_skipped_no_content', 0)} / "
                          f"{run.get('chunks_fully_filtered', 0)}"},
                {"Item": "Pairs by generation prompt version",
                 "Value": ", ".join(f"{v}: {n}" for v, n in run.get("pairs_by_prompt_version", {}).items()) or "–"},
                {"Item": "Generation time", "Value": f"{secs.get('generate', 0) / 60:.1f} min"},
                {"Item": "Validation time", "Value": f"{secs.get('validate', 0) / 60:.1f} min"},
                {"Item": "Tokens (generation / validation)", "Value": f"{run.get('tokens_generation', 0):,} / {run.get('tokens_validation', 0):,}"},
                {"Item": "Judge failures", "Value": str(stats.get("judge_failures", 0))},
            ]), hide_index=True, width="stretch")
            for warning in run.get("warnings", []):
                st.warning(warning)


# ---------------------------------------------------------------------------
# Tab 2: quality explorer
# ---------------------------------------------------------------------------
def pair_card(p: dict[str, Any], show_passage: bool) -> None:
    status = p.get("status", "kept")
    score = p.get("quality_score")
    head = status_pill(status) + (pill(f"Score {score}/6", "pill-blue") if score is not None else "") \
        + pill(p.get("question_type", "")) + f'<span class="meta">{esc(p.get("source"))} · page {esc(p.get("page"))}</span>'
    body = (f'<div class="card">{head}<div class="card-q">{esc(p["question"])}</div>'
            f'<div class="card-a">{esc(p["answer"])}</div>{score_bars(p.get("scores"))}')
    note = p.get("reject_reason") or p.get("judge_note")
    if note:
        body += f'<div class="reason"><b>Judge:</b> {esc(note)}</div>'
    if show_passage and p.get("chunk_text"):
        body += f'<div class="passage">{esc(p["chunk_text"])}</div>'
    st.markdown(body + "</div>", unsafe_allow_html=True)


with tab_quality:
    if not results or not results["judged"]:
        no_results_message()
    else:
        judged = [p for p in results["judged"] if p.get("answerable", True)]
        pipeline_threshold = results["stats"]["config"]["validate"]["min_quality_score"]
        floors_used = results["stats"]["config"]["validate"].get("min_criterion_scores", {})

        c1, c2 = st.columns([1, 1.4], gap="large")
        with c1:
            min_score = st.slider("Minimum quality score", 0, 6, 2, key="min_score",
                                  help="Drag from 2 to 5: vague and partly unsupported pairs drop out, specific grounded ones remain.")
        with c2:
            query = st.text_input("Search questions and answers", placeholder="e.g. fibre, disclosure, BMI…")
        f1, f2, f3, f4 = st.columns([1.3, 1.1, 1.1, 0.8])
        sources = sorted({p["source"] for p in judged})
        with f1:
            chosen_sources = st.multiselect("Source", sources, default=sources)
        with f2:
            types = sorted({p["question_type"] for p in judged})
            chosen_types = st.multiselect("Question type", types, default=types)
        with f3:
            chosen_status = st.multiselect("Status", ["kept", "duplicate", "rejected"],
                                           default=["kept", "duplicate", "rejected"])
        with f4:
            show_passages = st.toggle("Passages", value=False, help="Show the source passage under each pair")

        q = query.strip().lower()
        filtered = [p for p in judged
                    if (p.get("quality_score") or 0) >= min_score
                    and p["source"] in chosen_sources and p["question_type"] in chosen_types
                    and p.get("status") in chosen_status
                    and (not q or q in p["question"].lower() or q in p["answer"].lower())]

        dist = [sum(1 for p in judged if (p.get("quality_score") or 0) == s) for s in range(7)]
        shown_mask = [s >= min_score for s in range(7)]
        fig = go.Figure()
        for is_shown, name, color in ((True, f"Shown (≥ {min_score})", BLUE),
                                      (False, f"Hidden (< {min_score})", DE_EMPHASIS)):
            xs = [s for s in range(7) if shown_mask[s] == is_shown]
            ys = [dist[s] for s in xs]
            fig.add_bar(x=xs, y=ys, name=name, marker=dict(color=color, cornerradius=4),
                        text=[str(v) if v else "" for v in ys], textposition="outside", cliponaxis=False,
                        textfont=dict(color=INK_2, size=12), hovertemplate="Score %{x}: %{y} pairs<extra></extra>")
        fig.update_yaxes(range=[0, max(dist or [1]) * 1.15])
        fig.add_vline(x=pipeline_threshold - 0.5, line=dict(color=INK_2, width=1))
        fig.add_annotation(x=pipeline_threshold - 0.5, y=1, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"pipeline keeps ≥ {pipeline_threshold}", font=dict(color=INK_2, size=11), xanchor="left")
        fig.update_xaxes(tickmode="array", tickvals=list(range(7)), title_text="Judge quality score (0–6)")
        fig.update_yaxes(title_text="Pairs")
        fig.update_layout(barmode="overlay")

        left, right = st.columns([1.5, 1], gap="large")
        with left:
            floor_txt = ", ".join(f"{c} ≥ {v}" for c, v in floors_used.items())
            section("Quality score distribution",
                    "All judged answerable pairs, before deduplication. The pipeline also requires "
                    f"{floor_txt}, so a few pairs at or above the line are still rejected (see Status)."
                    if floor_txt else "All judged answerable pairs, before deduplication.")
            show_chart(style_fig(fig, height=280, legend=True),
                       pd.DataFrame({"Score": list(range(7)), "Pairs": dist}), key="hist")
        with right:
            kept_n = sum(1 for p in judged if (p.get("quality_score") or 0) >= min_score)
            avg_shown = (sum(p["quality_score"] for p in filtered) / len(filtered)) if filtered else None
            tiles([
                ("Pairs at this threshold", f"{kept_n:,}", f"of {len(judged):,} judged ({fmt_pct(kept_n / max(len(judged), 1))})"),
                ("Matching all filters", f"{len(filtered):,}", "listed below"),
                ("Average score shown", fmt_score(avg_shown), "out of 6"),
            ])

        section(f"{len(filtered):,} pairs", "Sorted by score, highest first.")
        filtered.sort(key=lambda p: (-(p.get("quality_score") or 0), p["question"]))
        per_page = 20
        pages = max(1, (len(filtered) + per_page - 1) // per_page)
        if st.session_state.get("qe_page", 1) > pages:  # filters shrank the list
            st.session_state.qe_page = pages
        page = st.number_input("Page", 1, pages, key="qe_page") if pages > 1 else 1
        for p in filtered[(page - 1) * per_page: page * per_page]:
            pair_card(p, show_passages)
        if not filtered:
            st.info("No pairs match these filters. Lower the minimum score or clear the search.")


# ---------------------------------------------------------------------------
# Tab 3: evidence
# ---------------------------------------------------------------------------
with tab_evidence:
    if not results:
        no_results_message()
    else:
        stats = results["stats"]
        judged = results["judged"]
        rejected = sorted([p for p in judged if p.get("status") == "rejected" and p.get("answerable", True)],
                          key=lambda p: (p.get("quality_score") or 0))
        # A representative sample of kept pairs: top-scored, taken round-robin across source documents.
        kept_by_source: dict[str, list[dict[str, Any]]] = {}
        for p in sorted([p for p in judged if p.get("status") == "kept" and p.get("answerable", True)],
                        key=lambda p: (-(p.get("quality_score") or 0), p["id"])):
            kept_by_source.setdefault(p["source"], []).append(p)
        kept = [p for group in zip(*kept_by_source.values()) for p in group] if kept_by_source else []

        val_cfg = stats["config"]["validate"]
        floor_txt = ", ".join(f"{c} ≥ {v}" for c, v in val_cfg.get("min_criterion_scores", {}).items())
        section("What the judge rejected, and why",
                f"{len(rejected)} answerable pairs were rejected: scored below {val_cfg['min_quality_score']}/6"
                + (f", or missed a per-criterion floor ({floor_txt})" if floor_txt else "")
                + ". Each keeps the judge's stated reason.")
        weak = {c: sum(1 for p in rejected if (p.get("scores") or {}).get(c, 2) < 2)
                for c in ("groundedness", "specificity", "completeness")}
        if rejected:
            names = [c.title() for c in weak]
            show_chart(hbar(names, list(weak.values()), text=[str(v) for v in weak.values()],
                            hover="%{y} below 2/2 in %{x} rejected pairs", xtitle="Rejected pairs"),
                       pd.DataFrame({"Criterion": names, "Rejected pairs scoring < 2": list(weak.values())}),
                       key="weak")

        shown_n = 5
        col_r, col_k = st.columns(2, gap="large")
        with col_r:
            st.markdown("**Rejected** · lowest scores first")
            for p in rejected[:shown_n]:
                pair_card(p, show_passage=False)
            if len(rejected) > shown_n:
                with st.expander(f"Show all {len(rejected)} rejected pairs"):
                    for p in rejected[shown_n:]:
                        pair_card(p, show_passage=False)
            if not rejected:
                st.caption("No pairs were rejected in this run.")
        with col_k:
            st.markdown("**Kept** · top scores, across all documents")
            for p in kept[:shown_n]:
                pair_card(p, show_passage=False)

        dups = results["duplicates"]
        ded_cfg = stats["config"]["deduplicate"]
        basis = "question and answer" if ded_cfg.get("embed") == "question_answer" else "question"
        section("Semantic duplicates removed",
                f"{len(dups)} pairs restated a fact already kept (embedding similarity of the {basis} above "
                f"{ded_cfg['similarity_threshold']}). The lexical score compares the questions' wording: a low "
                "value marks a rewording that exact or fuzzy string matching would have missed."
                + (" Pairs whose answers state different numbers are never merged."
                   if ded_cfg.get("require_same_numbers") else ""))
        for d in dups[:12]:
            lexical = difflib.SequenceMatcher(None, d["removed_question"].lower(),
                                              d["duplicate_of_question"].lower()).ratio()
            st.markdown(
                f'<div class="card">{pill(f"Semantic similarity {d["similarity"]:.2f}", "pill-blue")}'
                f'{pill(f"Lexical overlap of questions {lexical:.0%}")}<div class="vs">'
                f'<div><div class="vs-label">Removed</div><div class="card-q">{esc(d["removed_question"])}</div>'
                f'<div class="card-a">{esc(d.get("removed_answer"))}</div>'
                f'<div class="meta">{esc(d.get("removed_source"))} · page {esc(d.get("removed_page"))}</div></div>'
                f'<div><div class="vs-label">Kept (duplicate of)</div><div class="card-q">{esc(d["duplicate_of_question"])}</div>'
                f'<div class="card-a">{esc(d.get("duplicate_of_answer"))}</div>'
                f'<div class="meta">{esc(d.get("duplicate_of_source"))} · page {esc(d.get("duplicate_of_page"))}</div></div>'
                f'</div></div>', unsafe_allow_html=True)
        if not dups:
            st.caption("No duplicates above the similarity threshold in this run.")

        unans = results["unanswerable"]
        section("Unanswerable questions: teaching the model to abstain",
                "A model trained only on answerable questions learns that every question has an answer, and "
                "hallucinates when it does not know. These questions are on-topic but not answered by their "
                "passage, so the correct answer is to say so.")
        for u in unans[:10]:
            verdict = status_pill("kept" if u.get("kept") else "rejected")
            reason = u.get("reason")
            st.markdown(
                f'<div class="card">{verdict}<span class="meta">{esc(u.get("source"))} · page {esc(u.get("page"))}</span>'
                f'<div class="card-q">{esc(u["question"])}</div><div class="card-a"><i>{esc(u["answer"])}</i></div>'
                + (f'<div class="reason"><b>Judge:</b> {esc(reason)}</div>' if reason else "")
                + "</div>", unsafe_allow_html=True)
        if not unans:
            st.caption("No unanswerable questions were generated in this run.")

        section("Download")
        d1, d2, d3 = st.columns(3)
        d1.download_button("Dataset (JSONL)", read_bytes(OUTPUT_DIR / "synthetic_dataset.jsonl"),
                           "synthetic_dataset.jsonl", "application/jsonl", width="stretch", type="primary")
        d2.download_button("ChatML for fine-tuning", read_bytes(OUTPUT_DIR / "synthetic_dataset_chatml.jsonl"),
                           "synthetic_dataset_chatml.jsonl", "application/jsonl", width="stretch")
        d3.download_button("All outputs + evidence (ZIP)", zip_outputs(OUTPUT_DIR), "synthetic_dataset_outputs.zip",
                           "application/zip", width="stretch")


# ---------------------------------------------------------------------------
# Tab 4: judge reliability
# ---------------------------------------------------------------------------
LEVELS = {
    "groundedness": ["0 · info absent from passage", "1 · mixes in outside knowledge", "2 · fully traceable"],
    "specificity": ["0 · vague / generic", "1 · somewhat specific", "2 · tests this content"],
    "completeness": ["0 · truncated or yes/no", "1 · partial", "2 · fully answers"],
}


HOW_TO_SCORE = """
**How to score** (about 30 seconds per pair). Read the passage first, then the question and answer.

- **Groundedness**: is everything the answer says stated in the passage?
  **2** = yes, all of it · **1** = mostly, but it adds something the passage does not say · **0** = it states something wrong or absent.
- **Specificity**: does the question test *this* passage's content?
  **2** = specific to it · **1** = somewhat · **0** = generic (e.g. "What is this about?").
- **Completeness**: does the answer fully answer the question?
  **2** = fully · **1** = only partly · **0** = truncated, or a bare yes/no.

A correct, complete answer to a specific question is **2 / 2 / 2**; most pairs will be. Judge each pair on its own
merits. The AI judge's scores are hidden on purpose, so you are not trying to agree or disagree with it.
"""


def kappa_statement(result: dict[str, Any], target: int) -> str:
    d = result["decision"]
    k, band, n = d["kappa"], d["interpretation"], result["n"]
    prelim = f" This is preliminary: {n} of {target} sampled pairs are labelled." if n < target else ""
    if result.get("warnings"):
        return ("These labels cannot measure agreement yet (see the warning above), so no conclusion about the "
                "judge can be drawn from them." + prelim)
    if k is None:
        return ("Kappa is undefined for the labels so far (every decision was identical), so agreement cannot "
                "yet be distinguished from chance." + prelim)
    if band in ("strong", "substantial"):
        claim = ("The judge's keep/reject decisions agree with a human reviewer well beyond chance, so the "
                 "reported pass rate and quality scores can be read as a reasonable proxy for human judgement.")
    elif band == "moderate":
        claim = ("Agreement is only moderate: treat the pass rate and average quality as indicative rather than "
                 "precise, and check the per-criterion numbers to see where the judge and the human diverge.")
    else:
        claim = ("Agreement is weak: the judge's scores should not be taken as evidence of quality on their own. "
                 "The reported quality metrics are down-weighted accordingly.")
    return f"Keep/reject agreement is κ = {k:.2f} ({band}) on {n} pairs. {claim}{prelim}"


def render_agreement(result: dict[str, Any], total_items: int) -> None:
    """Kappa tiles, plain-language verdict, confusion matrix and per-criterion table."""
    for warning in result.get("warnings", []):
        st.warning(warning)
    d = result["decision"]
    tiles([("Keep/reject κ", "–" if d["kappa"] is None else f"{d['kappa']:.2f}", d["interpretation"])]
          + [(f"{c.title()} κ", "–" if r["kappa"] is None else f"{r['kappa']:.2f}",
              f"{r['interpretation']} · {fmt_pct(r['agreement'])} exact agreement")
             for c, r in result["criteria"].items()])
    st.markdown(kappa_statement(result, total_items))
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        cm = d["confusion_matrix"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Judge: reject", "Judge: keep"], y=["Human: reject", "Human: keep"],
            colorscale=[[0, "#eef4fc"], [1, "#1c5cab"]], showscale=False, xgap=2, ygap=2,
            text=[[str(v) for v in row] for row in cm], texttemplate="%{text}",
            textfont=dict(size=18), hovertemplate="%{y} · %{x}: %{z} pairs<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK_2))
        fig.update_xaxes(side="top", tickfont=dict(color=INK_2))
        section("Keep/reject decisions", "Diagonal = agreement.")
        show_chart(style_fig(fig, height=250), key=f"cm_{total_items}_{result['n']}")
    with right:
        section("Per-criterion agreement", "Weighted κ gives partial credit for 1-point disagreements.")
        st.dataframe(pd.DataFrame([
            {"Criterion": c.title(),
             "Cohen's κ": "–" if r["kappa"] is None else f"{r['kappa']:.3f}",
             "Weighted κ": "–" if r["weighted_kappa"] is None else f"{r['weighted_kappa']:.3f}",
             "Exact agreement": fmt_pct(r["agreement"]), "Band": r["interpretation"]}
            for c, r in result["criteria"].items()
        ] + [{"Criterion": "Keep / reject",
              "Cohen's κ": "–" if d["kappa"] is None else f"{d['kappa']:.3f}",
              "Weighted κ": "", "Exact agreement": fmt_pct(d["agreement"]),
              "Band": d["interpretation"]}]), hide_index=True, width="stretch")
        st.caption(f"Mean total score: human {result['mean_total_human']}, judge {result['mean_total_judge']}.")


def render_label_form(doc: dict[str, Any], items: list[dict[str, Any]], labels_file: Path) -> None:
    """One pair at a time: passage, question, answer, three 0-2 scores. Saves to disk on every click."""
    done = sum(1 for i in items if agreement.is_complete(i))
    labeller = st.text_input("Your name (recorded in the labels file)", value=doc.get("labeller", ""))
    first_open = next((i for i, it in enumerate(items) if not agreement.is_complete(it)), 0)
    if "label_idx" not in st.session_state:
        st.session_state.label_idx = first_open
    idx = min(st.session_state.label_idx, len(items) - 1)
    options = [f"{i + 1}. {'✓' if agreement.is_complete(it) else '○'} {it['question'][:80]}"
               for i, it in enumerate(items)]
    picked = st.selectbox("Jump to a pair (✓ = labelled)", options, index=idx)
    idx = options.index(picked)
    st.session_state.label_idx = idx
    item = items[idx]
    status = "already labelled, you can change it" if agreement.is_complete(item) else "not labelled yet"
    st.markdown(f"#### Pair {idx + 1} of {len(items)} · {status}")
    st.progress(done / max(len(items), 1), text=f"{done} of {len(items)} labelled")
    st.markdown(f'<div class="card"><div class="meta">{esc(item.get("source"))} · page {esc(item.get("page"))}</div>'
                f'<div class="passage">{esc(item.get("passage"))}</div>'
                f'<div class="card-q" style="margin-top:.7rem">Q: {esc(item["question"])}</div>'
                f'<div class="card-a">A: {esc(item["answer"])}</div></div>', unsafe_allow_html=True)
    with st.form(key=f"label_form_{idx}"):
        chosen: dict[str, int | None] = {}
        cols = st.columns(3)
        for col, (crit, levels) in zip(cols, LEVELS.items()):
            current = item.get(crit)
            with col:
                choice = st.radio(crit.title(), levels, index=current if current in (0, 1, 2) else None,
                                  key=f"{crit}_{idx}")
                chosen[crit] = levels.index(choice) if choice else None
        notes = st.text_input("Notes (optional)", value=item.get("notes", ""))
        saved = st.form_submit_button("Save and go to next", type="primary")
    if saved:
        if any(v is None for v in chosen.values()):
            st.warning("Choose a score for all three criteria before saving.")
            return
        item.update(chosen, notes=notes)
        doc["labeller"] = labeller
        agreement.save_labels(doc, labels_file)
        now_done = sum(1 for i in items if agreement.is_complete(i))
        nxt = next((i for i in range(idx + 1, len(items)) if not agreement.is_complete(items[i])),
                   next((i for i, it in enumerate(items) if not agreement.is_complete(it)), None))
        if nxt is None:
            st.session_state.label_flash = (f"Saved. All {len(items)} pairs are labelled: "
                                            "scroll to the top of this tab for the results.")
            st.session_state.label_idx = idx
        else:
            st.session_state.label_flash = (f"Saved pair {idx + 1} ({now_done} of {len(items)} labelled). "
                                            f"Showing pair {nxt + 1}.")
            st.session_state.label_idx = nxt
        st.rerun()


def render_reset(labels_file: Path, labelled: int) -> None:
    with st.expander("Start again: clear my labels"):
        st.caption(f"Removes all {labelled} of your scores and notes. The same 40 pairs stay in the sample.")
        confirm = st.checkbox("Yes, delete my labels")
        if st.button("Clear my labels", disabled=not confirm):
            agreement.clear_labels(labels_file)
            st.session_state.label_idx = 0
            st.session_state.label_flash = "Your labels were cleared. Start again from pair 1."
            st.rerun()


def judge_agreement(items: list[dict[str, Any]]) -> dict[str, Any]:
    judge_scores = {p["id"]: p["scores"] for p in results["judged"] if p.get("judge_ok") and p.get("scores")}
    val_cfg = results["stats"]["config"]["validate"]
    return agreement.compute_kappa(items, judge_scores, val_cfg["min_quality_score"],
                                   val_cfg.get("min_criterion_scores"))


def render_public_agreement(labels_file: Path) -> None:
    """Public mode: read-only, and results only once a person has finished a usable labelling."""
    items = agreement.load_labels(labels_file).get("items", []) if labels_file.is_file() else []
    if items and all(agreement.is_complete(i) for i in items):
        result = judge_agreement(items)
        if not result.get("warnings"):
            render_agreement(result, len(items))
            return
    st.info(f"The human check is not finished yet: the project author is scoring a blind sample of "
            f"{CONFIG.agreement.sample_size} pairs without seeing the judge's scores. The agreement results will "
            "appear here when it is done; until then, treat the judge's quality scores as unverified.")


with tab_judge:
    if not results or not results["judged"]:
        no_results_message()
    else:
        labels_file = OUTPUT_DIR / "human_labels.json"
        target_n = CONFIG.agreement.sample_size
        if st.session_state.get("label_flash"):
            st.toast(st.session_state.pop("label_flash"), icon="✅")
        section("Is the automated judge trustworthy?",
                "The AI judge decided which pairs to keep. To check it, a person scores a sample of the same "
                "pairs without seeing the judge's scores, and Cohen's kappa measures how far the two agree "
                "beyond chance: > 0.8 strong, 0.6–0.8 substantial, 0.4–0.6 moderate, < 0.4 weak.")
        if PUBLIC:
            render_public_agreement(labels_file)
        elif not labels_file.is_file():
            st.info(f"No human labels yet. Create a blind sample of {target_n} pairs spread across score bands, "
                    "then score them here (or edit data/output/human_labels.json).")
            if st.button(f"Create labelling sample ({target_n} pairs)", type="primary"):
                try:
                    agreement.sample_for_labelling(results["judged"], target_n, CONFIG.agreement.seed, labels_file)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        else:
            doc = agreement.load_labels(labels_file)
            items = doc.get("items", [])
            done = sum(1 for i in items if agreement.is_complete(i))
            result = judge_agreement(items)
            if not items:
                st.warning("The labels file has no items.")
            elif done == len(items):
                agreement.agreement_report(OUTPUT_DIR)
                st.success(f"All {len(items)} pairs are labelled. The results are below.")
                st.markdown(
                    "**Where the outputs are**  \n"
                    "- Your labels: `data/output/human_labels.json` (saved after every click)  \n"
                    "- Agreement results: `data/output/judge_agreement.json`, also printed by "
                    "`python -m src.agreement score`  \n"
                    "- README table: run `python -m src.experiments`, then copy section 3 of "
                    "`data/output/experiments.md` into the README's *Judge reliability* section")
                render_agreement(result, len(items))
                with st.expander("Review or change your labels"):
                    render_label_form(doc, items, labels_file)
                render_reset(labels_file, done)
            else:
                st.markdown(f"**Your task:** score all {len(items)} pairs below. The sample holds every pair the "
                            "judge scored below 6/6 plus a random set of 6/6 pairs, so each one matters. Scores are "
                            "saved after every click, so you can stop and continue later. Results appear at the top "
                            "of this tab when all pairs are labelled.")
                with st.expander("How to score", expanded=done == 0):
                    st.markdown(HOW_TO_SCORE)
                if result["n"] >= 5 and result.get("warnings"):
                    for warning in result["warnings"]:
                        st.warning(warning)
                render_label_form(doc, items, labels_file)
                if result["n"] >= 2:
                    with st.expander(f"Preliminary agreement on the {result['n']} pairs labelled so far"):
                        render_agreement(result, len(items))
                if done:
                    render_reset(labels_file, done)


# ---------------------------------------------------------------------------
# Tab 5: generate from uploaded PDFs
# ---------------------------------------------------------------------------
@st.cache_resource
def job_registry() -> dict[str, dict[str, Any]]:
    """Process-wide job table. Jobs run in background threads, so a click elsewhere
    in the app (which reruns the script) never interrupts a generation run."""
    return {}


def start_job(files: list[UploadedPDF], cfg, api_key: str | None) -> str:
    job_id, job_dir = new_job_dir(cfg)
    pdf_dir = save_uploads(files, job_dir)
    job: dict[str, Any] = {
        "id": job_id, "dir": job_dir, "status": "running", "stage": "queued", "current": 0, "total": 1,
        "started": time.time(), "finished": None, "stats": None, "error": None, "warnings": [],
        "files": [f.name for f in files], "stop": threading.Event(),
    }
    progress = make_progress(job)
    finished = work_started()

    def work() -> None:
        try:
            job["stats"] = run_pipeline(pdf_dir, job_dir / "output", cfg, max_chunks=cfg.web.max_chunks_per_upload,
                                        progress_callback=progress, cache_dir=job_dir / "cache",
                                        api_key=api_key, warnings=job["warnings"])
            job["status"] = "done"
        except PipelineCancelled:
            job["status"] = "stopped"
        except PipelineError as exc:
            job.update(status="error", error=str(exc))
        except Exception as exc:  # never show a raw traceback in the browser
            traceback.print_exc()
            job.update(status="error",
                       error=redact(f"Unexpected error ({type(exc).__name__}): {exc}", api_key))
        finally:
            job["finished"] = time.time()
            finished()

    job_registry()[job_id] = job
    threading.Thread(target=work, name=f"job-{job_id[:8]}", daemon=True).start()
    return job_id


def render_job_result(job: dict[str, Any]) -> None:
    out = job["dir"] / "output"
    stats = job["stats"]
    elapsed = (job["finished"] or time.time()) - job["started"]
    st.success(f"Done in {elapsed / 60:.1f} min: {stats['final_count']} pairs from {', '.join(job['files'])}.")
    for warning in job.get("warnings", []):
        st.warning(warning)
    render_stat_tiles(stats)
    rows = load_jsonl(out / "synthetic_dataset.jsonl") if (out / "synthetic_dataset.jsonl").is_file() else []
    if rows:
        section("Sample of generated pairs")
        st.dataframe(pd.DataFrame(rows)[["question", "answer", "question_type", "quality_score", "source", "page"]].head(15),
                     hide_index=True, width="stretch",
                     column_config={"question": st.column_config.TextColumn("Question", width="large"),
                                    "answer": st.column_config.TextColumn("Answer", width="large"),
                                    "question_type": "Type", "quality_score": "Score", "source": "Source",
                                    "page": "Page"})
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download dataset (JSONL)", read_bytes(out / "synthetic_dataset.jsonl"),
                       f"dataset_{job['id'][:8]}.jsonl", "application/jsonl", type="primary", width="stretch")
    d2.download_button("ChatML for fine-tuning", read_bytes(out / "synthetic_dataset_chatml.jsonl"),
                       f"dataset_{job['id'][:8]}_chatml.jsonl", "application/jsonl", width="stretch")
    d3.download_button("All outputs + evidence (ZIP)", zip_outputs(out), f"outputs_{job['id'][:8]}.zip",
                       "application/zip", width="stretch")


with tab_generate:
    web = CONFIG.web
    section("Generate a dataset from your own PDFs",
            "Upload guideline PDFs with a text layer. The same pipeline as the command line runs on them: "
            "generation, LLM-judge validation and semantic deduplication.")
    st.markdown('<div class="disclaimer"><b>Before you upload:</b> use public guideline documents only, never '
                'patient data. The generated dataset is for education and research, not medical advice or '
                'clinical decision-making.</div>', unsafe_allow_html=True)
    st.info(f"Demo mode: processes up to {web.max_chunks_per_upload} text chunks, spread across your files "
            f"(about 5–8 minutes), because the free API tier limits tokens per minute. "
            f"Limits: {web.max_files_per_job} files, {web.max_upload_mb:g} MB each. Scanned PDFs without a "
            "text layer cannot be read.")

    job_id = st.session_state.get("job_id")
    job = job_registry().get(job_id) if job_id else None
    running = bool(job and job["status"] == "running")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        uploads = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, disabled=running)
    with right:
        min_q = st.slider("Keep pairs scoring at least", 3, 6, CONFIG.validate.min_quality_score, disabled=running,
                          help="Judge score out of 6. Higher means fewer but better pairs.")
        sim = st.slider("Duplicate similarity threshold", 0.75, 0.95, float(CONFIG.deduplicate.similarity_threshold),
                        0.01, disabled=running, help="Questions more similar than this are treated as duplicates.")
        floors_cfg = CONFIG.validate.min_criterion_scores.to_dict()
        st.caption("Always required as well: " + ", ".join(f"{c} ≥ {v}/2" for c, v in floors_cfg.items())
                   + ". This stops a fluent answer with an unsupported claim from passing on its total alone.")
        if PUBLIC:  # never spend the server's key on a visitor's job
            st.text_input("Your Groq API key", type="password", key="user_api_key", disabled=running,
                          help="Held in this browser session only: never written to disk or logged.")
            st.caption("Required here. A free key takes a minute at "
                       "[console.groq.com/keys](https://console.groq.com/keys).")
        else:
            with st.expander("Use your own Groq API key (optional)"):
                st.text_input("Groq API key", type="password", key="user_api_key",
                              help="Held in this browser session only: never written to disk or logged.")
                st.caption("If left empty, the server's key is used"
                           + ("." if has_api_key() else " (none is configured)."))

    files = [UploadedPDF(u.name, u.getvalue()) for u in (uploads or [])]
    problems = validate_uploads(files, CONFIG) if files else []
    for problem in problems:
        st.error(problem)
    user_key = (st.session_state.get("user_api_key") or "").strip() or None
    key_available = bool(user_key) or (has_api_key() and not PUBLIC)
    if not key_available:
        st.warning("Enter your Groq API key above to generate a dataset." if PUBLIC else
                   "No Groq API key is available. Add GROQ_API_KEY to .env or enter your own key above.")

    go_clicked = st.button("Generate dataset", type="primary", disabled=running or not files or bool(problems)
                           or not key_available)
    if go_clicked and key_available:
        cfg = with_overrides(CONFIG, {"validate.min_quality_score": min_q, "deduplicate.similarity_threshold": sim})
        st.session_state.job_id = start_job(files, cfg, user_key)
        st.rerun()

    @st.fragment(run_every=1.0 if running else None)
    def job_panel() -> None:
        current = job_registry().get(st.session_state.get("job_id") or "")
        if not current:
            return
        if current["status"] == "running":
            stopping = current["stop"].is_set()
            title = "Stopping after the current step" if stopping else stage_text(current)
            with st.status(f"{title}…", state="running", expanded=True):
                st.progress(overall_progress(current), text=f"{overall_progress(current):.0%} · "
                                                             f"{elapsed_text(current)} elapsed")
                order = ["ingesting", "generating", "validating", "deduplicating", "exporting"]
                reached = order.index(current["stage"]) if current["stage"] in order else -1
                st.markdown("  \n".join(
                    ("✓ " if i < reached else "▸ " if i == reached else "○ ") + STAGE_LABELS[s]
                    for i, s in enumerate(order)))
            if st.button("■ Stop this job", disabled=stopping, key=f"stop_{current['id']}"):
                current["stop"].set()
                st.rerun(scope="fragment")
        elif running:
            st.rerun()  # the job just finished: redraw the whole page once with results

    job_panel()
    if job and job["status"] == "done":
        render_job_result(job)
    elif job and job["status"] == "stopped":
        st.info(f"Job stopped after {elapsed_text(job)}. Upload the PDFs again and press Generate to start a "
                "new job.")
    elif job and job["status"] == "error":
        st.error(f"The job could not finish. {job['error']}")


# ---------------------------------------------------------------------------
# Tab 6: output (download every result file)
# ---------------------------------------------------------------------------
def file_available(folder: Path, name: str) -> bool:
    return (folder / ("synthetic_dataset.jsonl" if name == "synthetic_dataset.csv" else name)).is_file()


def file_meta(folder: Path, name: str) -> str:
    source = folder / ("synthetic_dataset.jsonl" if name == "synthetic_dataset.csv" else name)
    updated = time.strftime("%d %b %Y, %H:%M", time.localtime(source.stat().st_mtime))
    size = "" if name == "synthetic_dataset.csv" else human_size(source.stat().st_size)
    return " · ".join(part for part in (row_count(folder, name), size, f"updated {updated}") if part)


def download(label: str, folder: Path, name: str, file_name: str, key: str) -> None:
    """A download button whose bytes are read (or built) only when it is clicked."""
    st.download_button(label, data=lambda f=folder, n=name: output_bytes(f, n), file_name=file_name,
                       mime=MIME.get(Path(name).suffix, "application/octet-stream"), key=key,
                       on_click="ignore", width="stretch", icon=":material/download:")


SHOW_ALL_JOBS = CONFIG.web.show_all_upload_jobs and not PUBLIC  # visitors never see each other's jobs


def upload_jobs() -> list[tuple[str, Path, dict[str, Any]]]:
    """Finished upload jobs on disk, newest first: (job id, output folder, stats)."""
    root = resolve_path(CONFIG.paths.uploads_dir)
    session_job = st.session_state.get("job_id")
    jobs = []
    for stats_path in root.glob("*/output/pipeline_stats.json") if root.is_dir() else []:
        job_id = stats_path.parent.parent.name
        if not SHOW_ALL_JOBS and job_id != session_job:
            continue
        try:
            with open(stats_path, encoding="utf-8") as fh:
                jobs.append((job_id, stats_path.parent, json.load(fh)))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(jobs, key=lambda j: j[1].stat().st_mtime, reverse=True)


with tab_output:
    section("Download your results",
            "Every file the pipeline produced, ready to save to your computer. Each download is the file "
            "exactly as the pipeline wrote it; the CSV is converted from the final dataset for spreadsheets.")

    if not (OUTPUT_DIR / "pipeline_stats.json").is_file():
        no_results_message()
    else:
        s = results["stats"] if results else {}
        run = s.get("run", {})
        stamp = time.strftime("%Y%m%d", time.localtime((OUTPUT_DIR / "pipeline_stats.json").stat().st_mtime))
        st.markdown(f"**Main dataset:** {s.get('final_count', 0):,} pairs from "
                    f"{len(run.get('pdf_files', []))} PDFs · average quality "
                    f"{fmt_score(s.get('average_quality_score'))} / 6 · saved in `data/output/`")
        left, _ = st.columns([1, 2])
        with left:
            st.download_button("Download everything (ZIP)", data=lambda: zip_outputs(OUTPUT_DIR),
                               file_name=f"synthetic_dataset_outputs_{stamp}.zip", mime="application/zip",
                               key="dl_all", on_click="ignore", type="primary", width="stretch",
                               icon=":material/folder_zip:")

        section("Individual files")
        for name, title, desc, how in OUTPUT_CATALOG:
            with st.container(border=True):
                info, meta, button = st.columns([3.2, 1.6, 1.1], vertical_alignment="center")
                available = file_available(OUTPUT_DIR, name)
                with info:
                    st.markdown(f"**{title}**  \n{desc}")
                    st.caption(f"`{name}`")
                with meta:
                    st.caption(file_meta(OUTPUT_DIR, name) if available else f"Not created yet: run {how}.")
                with button:
                    if available:
                        download("Download", OUTPUT_DIR, name, name, key=f"dl_{name}")
                    else:
                        st.button("Not available", key=f"na_{name}", disabled=True, width="stretch")

        previewable = [n for n, *_ in OUTPUT_CATALOG if n.endswith(".jsonl") and file_available(OUTPUT_DIR, n)]
        if previewable:
            section("Preview a file", "The first 200 records, to check a file before downloading it.")
            chosen = st.selectbox("File", previewable, label_visibility="collapsed")
            rows = load_jsonl(OUTPUT_DIR / chosen)[:200]
            if chosen == "synthetic_dataset_chatml.jsonl":
                rows = [{"user": r["messages"][-2]["content"], "assistant": r["messages"][-1]["content"]} for r in rows]
            frame = pd.DataFrame(rows).drop(columns=["chunk_text"], errors="ignore")
            st.dataframe(frame, hide_index=True, width="stretch", height=380)

    jobs = upload_jobs()
    section("Your upload jobs",
            "Datasets generated from PDFs you uploaded in the Generate tab. Each job is kept in its own folder "
            "under data/uploads/." + ("" if SHOW_ALL_JOBS else " Showing this session's job only."))
    if not jobs:
        st.caption("No upload jobs yet. Generate one in the Generate from your PDFs tab.")
    for job_id, folder, job_stats in jobs:
        job_run = job_stats.get("run", {})
        when = time.strftime("%d %b %Y, %H:%M", time.localtime(folder.stat().st_mtime))
        this_session = " · this session" if job_id == st.session_state.get("job_id") else ""
        with st.container(border=True):
            st.markdown(f"**{', '.join(job_run.get('pdf_files', [])) or 'Upload job'}**{this_session}")
            st.caption(f"{when} · {job_stats.get('final_count', 0)} pairs · average quality "
                       f"{fmt_score(job_stats.get('average_quality_score'))} / 6 · job {job_id[:8]}")
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                download("Dataset (JSONL)", folder, "synthetic_dataset.jsonl",
                         f"job_{job_id[:8]}_synthetic_dataset.jsonl", key=f"job_{job_id}_jsonl")
            with b2:
                download("Dataset (CSV)", folder, "synthetic_dataset.csv",
                         f"job_{job_id[:8]}_synthetic_dataset.csv", key=f"job_{job_id}_csv")
            with b3:
                download("ChatML", folder, "synthetic_dataset_chatml.jsonl",
                         f"job_{job_id[:8]}_chatml.jsonl", key=f"job_{job_id}_chatml")
            with b4:
                st.download_button("Everything (ZIP)", data=lambda f=folder: zip_outputs(f),
                                   file_name=f"job_{job_id[:8]}_outputs.zip", mime="application/zip",
                                   key=f"job_{job_id}_zip", on_click="ignore", width="stretch",
                                   icon=":material/folder_zip:")
