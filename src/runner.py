"""
Launches and supervises pipeline runs started from the dashboard.

The pipeline takes ten to twenty minutes. Running it inside a Streamlit
callback would freeze the page for the whole time and lose everything on a
browser refresh, so it runs as a detached child process instead. The dashboard
polls `progress.read()` and the log file.

State lives in `data/output/run_state.json` rather than Streamlit's session, so
a refresh - or a second browser tab - still sees the run that is in flight.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

try:
    from config import OUTPUT_DIR, PROJECT_ROOT
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import OUTPUT_DIR, PROJECT_ROOT

import progress

STATE_FILE = OUTPUT_DIR / "run_state.json"
LOG_FILE = OUTPUT_DIR / "run.log"

# Uploads: enough to make a real corpus, few enough to stay inside one day of
# free-tier quota. The brief asked for 5-8; 8 is the ceiling.
MAX_FILES = 8
MAX_FILE_MB = 50

# Older name, kept so nothing that imported it breaks.
MAX_PDFS = MAX_FILES
MAX_PDF_MB = MAX_FILE_MB


def _python() -> str:
    """
    The interpreter to run the pipeline with.

    sys.executable is the one running Streamlit, which is the same venv - so
    the child inherits every dependency without hunting for a path.
    """
    return sys.executable


def _pid_alive(pid: int) -> bool:
    """Cross-platform liveness check without adding a psutil dependency."""
    if not pid:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            return str(pid) in out
        except Exception:
            return False
    try:
        import os

        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _read_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(data: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_running() -> bool:
    """True only if a process we started is genuinely still alive."""
    state = _read_state()
    if not state.get("pid"):
        return False
    if not _pid_alive(state["pid"]):
        return False
    # A live PID that stopped reporting is a stalled run, not a working one.
    prog = progress.read()
    if prog.get("stale"):
        return False
    return prog.get("status", "running") == "running"


# Read from deduplicate so the dashboard, the CLI and the module itself
# cannot disagree about what the default is.
try:
    from deduplicate import SIMILARITY_THRESHOLD as DEFAULT_SIMILARITY
except Exception:  # pragma: no cover - keeps the dashboard usable if torch is absent
    DEFAULT_SIMILARITY = 0.90


def start(
    max_chunks: int = 75,
    min_score: int = 4,
    similarity: float = DEFAULT_SIMILARITY,
    questions_per_chunk: int = 3,
    delay: float = 1.0,
    api_key: str | None = None,
) -> dict:
    """
    Launch a pipeline run in the background. Returns the new state.

    `api_key` lets the dashboard supply a key typed by the visitor instead of
    one in .env. It is passed to the child process through the environment
    only - never written to disk, never logged - so a public deployment can
    run on each visitor's own key without the server holding one.
    """
    if is_running():
        return {"error": "A run is already in progress."}

    progress.clear()

    # Lay down the timing plan before the child process starts. main.py
    # refines it once the real chunk count is known, but writing it here means
    # the bar and the ETA are meaningful from the first second instead of only
    # after ingest finishes.
    page_budget = max(8, int(max_chunks / 3 / 0.75 * 1.8)) if max_chunks else 0
    progress.set_plan(max_chunks, questions_per_chunk, pages=page_budget)

    # Python takes ~30-40s to import torch, langchain and the Groq client on a
    # cold start. Without this message the dashboard sits at 0% looking hung.
    progress.write(
        "ingest", 0, 0,
        "Starting Python and loading libraries (this takes about 30 seconds)...",
    )

    cmd = [
        _python(),
        str(PROJECT_ROOT / "main.py"),
        "--max-chunks", str(max_chunks),
        "--min-score", str(min_score),
        "--similarity", str(similarity),
        "--questions-per-chunk", str(questions_per_chunk),
        "--delay", str(delay),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w", encoding="utf-8")

    # Unbuffered, UTF-8: without -u the log stays empty until the process ends,
    # which would leave the dashboard with nothing to show for twenty minutes.
    env_flags = {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    import os

    env = {**os.environ, **env_flags}
    if api_key:
        env["GROQ_API_KEY"] = api_key.strip()

    creation = 0
    if sys.platform == "win32":
        # Detach from the console so closing Streamlit does not kill the run.
        creation = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=env,
        creationflags=creation,
    )

    state = {
        "pid": proc.pid,
        "started": time.time(),
        "params": {
            "max_chunks": max_chunks,
            "min_score": min_score,
            "similarity": similarity,
            "questions_per_chunk": questions_per_chunk,
        },
        "log": str(LOG_FILE),
    }
    _write_state(state)
    return state


def stop() -> bool:
    """Terminate the running pipeline, if any."""
    state = _read_state()
    pid = state.get("pid")
    if not pid or not _pid_alive(pid):
        return False
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=15)
        else:
            import os
            import signal

            os.kill(pid, signal.SIGTERM)
    except Exception:
        return False
    # "stopped" rather than "error": the user chose this, and the dashboard
    # should not report their own action back to them as a failure.
    progress.write("error", 0, 0, "Run stopped by user.", status="stopped")
    return True


def log_tail(lines: int = 25) -> list[str]:
    """Last N interesting lines of the run log, for the dashboard."""
    try:
        raw = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    skip = ("DeprecationWarning", "from langchain", "warnings.warn",
            "Loading weights", "HF_TOKEN", "huggingface")
    keep = [l for l in raw if l.strip() and not any(s in l for s in skip)]
    return keep[-lines:]


def status() -> dict:
    """Everything the dashboard needs to render the current run."""
    state = _read_state()
    prog = progress.read()
    running = is_running()
    elapsed = time.time() - state["started"] if state.get("started") else 0
    eta = None
    if running and prog:
        eta = progress.eta_seconds(
            prog.get("stage", ""), prog.get("current", 0), prog.get("total", 0), elapsed
        )
        # Fill in the opening phase, where the pipeline cannot yet report
        # anything because Python is still importing its libraries.
        prog = {**prog, "overall_pct": progress.interpolated_pct(prog, elapsed)}

    return {
        "running": running,
        "pid": state.get("pid"),
        "params": state.get("params", {}),
        "elapsed": elapsed,
        "eta": eta,
        "progress": prog,
        "log": log_tail(),
    }


# --- Uploaded files --------------------------------------------------------
# The pipeline works only on what the user uploads in this session. Nothing is
# carried over between sessions: on a fresh start the input folder is emptied,
# so reopening the project always asks for documents rather than silently
# reusing whatever happened to be left behind. That was a real trap - a run
# once produced a dataset from documents the user had long since replaced.


def supported_extensions() -> set[str]:
    """Extensions the ingest step can actually read, as the single source."""
    from ingest import SUPPORTED_EXTENSIONS

    return SUPPORTED_EXTENSIONS


def clear_inputs() -> int:
    """Delete every uploaded file. Returns how many were removed."""
    from config import RAW_DIR

    if not RAW_DIR.exists():
        return 0
    removed = 0
    for f in RAW_DIR.iterdir():
        if f.is_file():
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def current_files() -> list[dict]:
    """Whatever is queued for the next run, with per-file detail."""
    from config import RAW_DIR
    from ingest import FORMAT_NOTES

    if not RAW_DIR.exists():
        return []
    out = []
    for p in sorted(RAW_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in supported_extensions():
            continue
        out.append(
            {
                "name": p.name,
                "type": p.suffix.lower().lstrip("."),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                "note": FORMAT_NOTES.get(p.suffix.lower(), ""),
            }
        )
    return out


# Old name, kept so nothing that imported it breaks.
current_pdfs = current_files


# Pages sampled when checking an upload. Reading a 200-page PDF in full just
# to answer "does this have a text layer?" made uploads feel slow for no
# benefit - the answer is identical after three pages.
PROBE_PAGES = 3


def probe(path: Path) -> tuple[int, int]:
    """
    Return (page count, estimated total characters) for one file.

    Estimated, not exact: only the first PROBE_PAGES pages are read and the
    character count is scaled up. The figure is used to spot empty and scanned
    files and to size the run, neither of which needs precision - and the full
    parse happens during ingest anyway.
    """
    try:
        import warnings as w

        w.filterwarnings("ignore")

        if path.suffix.lower() == ".pdf":
            import pypdf

            reader = pypdf.PdfReader(str(path))
            total_pages = len(reader.pages)
            if total_pages == 0:
                return 0, 0
            sampled = min(PROBE_PAGES, total_pages)
            chars = sum(
                len((reader.pages[i].extract_text() or "").strip())
                for i in range(sampled)
            )
            return total_pages, int(chars / sampled * total_pages)

        # Text and Word files are small and cheap to read whole.
        from ingest import load_one

        pages = load_one(path)
        return len(pages), sum(len(p.page_content.strip()) for p in pages)
    except Exception:
        return 0, 0


def save_uploads(files, replace: bool = True, on_progress=None) -> dict:
    """
    Store uploaded documents, checking each one is actually readable.

    Checking on upload rather than at generation time matters: a scanned PDF
    with no text layer yields nothing, and discovering that after twenty
    minutes of API calls is a bad way to learn it. A file that cannot be read
    is deleted again rather than left to fail the run later.
    """
    from config import RAW_DIR

    if not files:
        return {"error": "No files selected."}
    if len(files) > MAX_FILES:
        return {
            "error": f"Too many files: {len(files)}. The maximum is {MAX_FILES}."
        }

    allowed = supported_extensions()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if replace:
        clear_inputs()

    # Progress is weighted by BYTES, not by file count. A 40 MB PDF and a 2 KB
    # text file are one file each, but one takes seconds and the other is
    # instant - counting files would make the bar leap then freeze. Each file
    # contributes two steps: writing it, then reading it back to check it has
    # extractable text, with the check given the larger share because that is
    # where the time actually goes.
    sizes = []
    for f in files:
        try:
            sizes.append(max(len(f.getbuffer()), 1))
        except Exception:
            sizes.append(1)
    total_bytes = sum(sizes) or 1
    done_bytes = 0.0

    def report(fraction: float, name: str, phase: str) -> None:
        if on_progress is not None:
            try:
                on_progress(min(fraction, 1.0), name, phase)
            except Exception:
                pass

    saved, warnings, errors = [], [], []
    for position, f in enumerate(files):
        weight = sizes[position]
        name = Path(f.name).name
        suffix = Path(name).suffix.lower()

        if suffix not in allowed:
            errors.append(
                f"{name}: {suffix or 'no extension'} is not supported "
                f"({', '.join(sorted(allowed))})"
            )
            done_bytes += weight
            report(done_bytes / total_bytes, name, "Skipped")
            continue

        data = f.getbuffer()
        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            errors.append(
                f"{name}: {size_mb:.0f} MB exceeds the {MAX_FILE_MB} MB limit"
            )
            done_bytes += weight
            report(done_bytes / total_bytes, name, "Skipped")
            continue

        report(done_bytes / total_bytes, name, "Saving")
        target = RAW_DIR / name
        target.write_bytes(data)

        # Writing is roughly a third of the work; reading the file back to
        # verify it has text is the rest.
        done_bytes += weight * 0.35
        report(done_bytes / total_bytes, name, "Checking")

        pages, chars = probe(target)
        done_bytes += weight * 0.65
        if pages == 0 or chars == 0:
            errors.append(f"{name}: no readable text could be extracted")
            target.unlink(missing_ok=True)
            report(done_bytes / total_bytes, name, "Rejected")
            continue

        per_page = chars // max(pages, 1)
        if per_page < 100:
            warnings.append(
                f"{name}: only {per_page} characters per page. "
                + (
                    "This looks like a scanned image with no text layer and "
                    "will produce few or no pairs - replace it with a "
                    "text-based PDF."
                    if suffix == ".pdf"
                    else "There is very little text here, so expect few pairs."
                )
            )

        saved.append(
            {
                "name": name,
                "type": suffix.lstrip("."),
                "pages": pages,
                "chars": chars,
                "chars_per_page": per_page,
                "size_mb": round(size_mb, 2),
            }
        )
        report(done_bytes / total_bytes, name, "Loaded")

    report(1.0, "", "Done")
    return {"saved": saved, "warnings": warnings, "errors": errors}


def estimate_chunks() -> int:
    """
    Roughly how many chunks the queued files will produce.

    Used to warn before a run that would exceed the daily token budget. It
    reads the files rather than guessing, but stays cheap: no model, no API.
    """
    total_chars = 0
    for info in current_files():
        from config import RAW_DIR

        _pages, chars = probe(RAW_DIR / info["name"])
        total_chars += chars
    if not total_chars:
        return 0
    # 600-char chunks with 80 overlap, less ~20% lost to boilerplate.
    return int(total_chars / 520 * 0.8)
