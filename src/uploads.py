"""Validation and storage for PDFs uploaded through the dashboard.

Uploaded PDFs are the pipeline's only input: each upload becomes the input
folder of the next run. Kept free of streamlit so it can be unit-tested and so
the UI holds no logic beyond presentation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.config import CONFIG, Config, resolve_path

PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class UploadedPDF:
    name: str
    data: bytes


def validate_uploads(files: list[UploadedPDF], cfg: Config = CONFIG) -> list[str]:
    """Return a list of human-readable problems; empty means the upload is acceptable.

    Checks: at least one file, at most ``web.max_files_per_run`` files, each
    under ``web.max_upload_mb``, a .pdf extension, and real PDF content (the
    file must start with the %PDF- signature, so a renamed .docx or image is
    rejected rather than failing obscurely inside the parser).
    """
    errors: list[str] = []
    limit_mb = cfg.web.max_upload_mb
    if not files:
        return ["Please upload at least one PDF."]
    if len(files) > cfg.web.max_files_per_run:
        errors.append(f"Too many files: {len(files)} uploaded, the limit is {cfg.web.max_files_per_run} per run.")
    for f in files:
        size_mb = len(f.data) / (1024 * 1024)
        if not f.name.lower().endswith(".pdf"):
            errors.append(f"'{f.name}' is not a PDF file.")
        elif not f.data.lstrip()[:5] == PDF_MAGIC:
            errors.append(f"'{f.name}' has a .pdf name but its content is not a PDF.")
        if size_mb > limit_mb:
            errors.append(f"'{f.name}' is {size_mb:.1f} MB; the limit is {limit_mb} MB per file.")
        elif len(f.data) == 0:
            errors.append(f"'{f.name}' is empty.")
    return errors


def safe_filename(name: str) -> str:
    """Strip any path components and unsafe characters from an uploaded file name."""
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" .")
    return base or "upload.pdf"


def replace_pdfs(files: list[UploadedPDF], pdf_dir: str | Path) -> Path:
    """Make ``pdf_dir`` hold exactly these PDFs (safe, unique names); return the folder.

    Why remove the PDFs already there: the pipeline processes every PDF in its
    input folder, and each upload starts a new run, so the previous run's PDFs
    must not slip into it. Only .pdf files are touched.
    """
    pdf_dir = resolve_path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for old in pdf_dir.iterdir():
        if old.is_file() and old.suffix.lower() == ".pdf":
            old.unlink()
    used: set[str] = set()
    for f in files:
        name = safe_filename(f.name)
        stem, suffix = Path(name).stem, Path(name).suffix or ".pdf"
        candidate, n = f"{stem}{suffix}", 1
        while candidate.lower() in used:
            n += 1
            candidate = f"{stem}_{n}{suffix}"
        used.add(candidate.lower())
        (pdf_dir / candidate).write_bytes(f.data)
    return pdf_dir
