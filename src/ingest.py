"""PDF -> clean page documents -> overlapping text chunks.

Every chunk keeps ``source`` (file name) and ``page`` (1-based PDF page) so each
generated Q&A pair can be cited back to the exact page it came from.
"""

from __future__ import annotations

import re
import warnings as _warnings
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from src.config import CONFIG, Config, resolve_path


def _pdf_loader_class():
    """Import PyPDFLoader on first use.

    langchain_community is slow to import (tens of seconds on a cold Windows
    start), and chunking, the tests and the dashboard do not need it. Its
    package-level "being sunset" DeprecationWarning is silenced here;
    PyPDFLoader itself is unaffected.
    """
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)
        from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader


class NoReadablePDFError(RuntimeError):
    """Raised when PDFs exist but none has an extractable text layer."""


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
# Private-use-area glyphs are icon fonts (arrows, logos) that extract as garbage.
_PUA = re.compile(r"[\ue000-\uf8ff]")
_BULLETS = re.compile(r"^\s*[▶►•●▪■◦○▸‣∙·]\s*")
_PAGE_NUMBER = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,7})\s*$", re.IGNORECASE)
# "carbo - \nhydrate": a spaced hyphen at a line end is a typesetting break.
_SOFT_HYPHEN_BREAK = re.compile(r"(\w) -\s*\n\s*(\w)")
_SENTENCE_END = (".", "?", "!", ":", ";")


def _normalise_line(line: str) -> str:
    return re.sub(r"\s+", " ", _PUA.sub("", line)).strip()


def _header_key(line: str) -> str:
    """Key for spotting running headers: ignore digits (page numbers) and case."""
    return re.sub(r"[\d\s]+", " ", _normalise_line(line)).strip().lower()


def _find_running_headers(pages: list[str], min_pages: int = 3) -> set[str]:
    """Lines that recur in the first/last two lines of several pages.

    Why: WHO guidelines repeat the document title and section name on every
    page. Left in, they are glued into the middle of sentences and questions.
    """
    counts: Counter[str] = Counter()
    for text in pages:
        lines = [ln for ln in text.split("\n") if _normalise_line(ln)]
        edge = set(_header_key(ln) for ln in lines[:2] + lines[-2:])
        counts.update(k for k in edge if len(k) > 3)
    threshold = max(min_pages, len(pages) // 20)
    return {key for key, n in counts.items() if n >= threshold}


def clean_page_text(text: str, running_headers: set[str] | frozenset[str] = frozenset()) -> str:
    """Rebuild readable prose from pypdf's line-per-visual-line output.

    - drop page numbers and running headers/footers at the page edges
    - rejoin lines broken mid-sentence; keep a newline after sentence-ending
      punctuation and before bullets, so the splitter still has natural breaks
    - repair typesetting hyphen breaks and strip icon-font glyphs
    """
    text = _SOFT_HYPHEN_BREAK.sub(r"\1\2", text.replace("\r", ""))
    lines = [ln for ln in text.split("\n")]
    # Strip edge lines (top two / bottom two) that are page numbers or headers.
    kept = [ln for ln in lines if _normalise_line(ln)]
    for _ in range(2):
        if kept and (_PAGE_NUMBER.match(kept[0]) or _header_key(kept[0]) in running_headers):
            kept.pop(0)
        if kept and (_PAGE_NUMBER.match(kept[-1]) or _header_key(kept[-1]) in running_headers):
            kept.pop()

    out: list[str] = []
    for raw in kept:
        is_bullet = bool(_BULLETS.match(raw))
        line = _normalise_line(_BULLETS.sub("", raw))
        if not line:
            continue
        if is_bullet:
            line = "- " + line
        if not out:
            out.append(line)
        elif is_bullet or out[-1].endswith(_SENTENCE_END):
            out.append("\n" + line)
        elif out[-1].endswith("-") and line[:1].isalpha():
            out.append(line)  # "non-\nconditional" -> "non-conditional"
        else:
            out.append(" " + line)
    return "".join(out).strip()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def list_pdfs(data_dir: str | Path) -> list[Path]:
    data_dir = resolve_path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"PDF folder does not exist: {data_dir}")
    return sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def load_pdfs(
    data_dir: str | Path,
    cfg: Config = CONFIG,
    warnings: list[str] | None = None,
) -> list[Document]:
    """Load every PDF in ``data_dir`` as one cleaned Document per page.

    Why per page: the page number is the citation. A chunk never spans two
    pages, so its ``page`` metadata is always exact.

    PDFs with almost no extractable text (under ``ingest.min_pdf_chars``) are
    scanned images with no text layer. pypdf cannot read them and OCR is out of
    scope, so they are skipped with an explicit warning instead of producing
    empty or garbage chunks. Warnings are printed and, if a ``warnings`` list is
    passed, appended to it so the web interface can show them.
    """
    pdfs = list_pdfs(data_dir)
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found in {resolve_path(data_dir)}. Add guideline PDFs there and re-run."
        )
    warn = warnings if warnings is not None else []
    docs: list[Document] = []
    print(f"Loading {len(pdfs)} PDF file(s) from {resolve_path(data_dir)}")
    loader_cls = _pdf_loader_class()
    for path in pdfs:
        try:
            pages = loader_cls(str(path)).load()
        except Exception as exc:  # corrupt, encrypted or not really a PDF
            msg = f"Could not open '{path.name}' ({type(exc).__name__}: {exc}). Skipped."
            print(f"  WARNING: {msg}")
            warn.append(msg)
            continue
        raw_chars = sum(len(p.page_content.strip()) for p in pages)
        if raw_chars < cfg.ingest.min_pdf_chars:
            msg = (
                f"'{path.name}' has {len(pages)} page(s) but only {raw_chars} characters of text. "
                "It is probably a scanned image without a text layer, which cannot be read "
                "without OCR. Skipped."
            )
            print(f"  WARNING: {msg}")
            warn.append(msg)
            continue
        headers = _find_running_headers([p.page_content for p in pages])
        kept_pages = 0
        for p in pages:
            text = clean_page_text(p.page_content, headers)
            if not text:
                continue
            page_no = int(p.metadata.get("page", 0)) + 1  # PyPDFLoader pages are 0-based
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "page": page_no,
                    "page_label": str(p.metadata.get("page_label", page_no)),
                },
            ))
            kept_pages += 1
        print(f"  {path.name}: {len(pages)} pages ({kept_pages} with text)")
    if not docs:
        raise NoReadablePDFError(
            "None of the PDFs contained extractable text. They may be scanned images; "
            "export them with a text layer (or OCR them first) and try again."
        )
    print(f"Total pages loaded: {len(docs)}")
    return docs


# ---------------------------------------------------------------------------
# Boilerplate detection
# ---------------------------------------------------------------------------
_CITATION_MARKERS = re.compile(
    r"\bet al\b|\bdoi\b|https?://|www\.|\bISBN\b|\bPMID\b|\bAccessed\b|"
    r"\b(?:19|20)\d{2}\s*;\s*\d+|\d+\s*\(\s*\d+\s*\)\s*:\s*\d+|\bVol\.\s*\d+",
    re.IGNORECASE,
)
# "Schwingshackl L," / "Hobbs FD." : a surname followed by initials, as in reference lists.
_AUTHOR_INITIALS = re.compile(r"\b[A-Z][a-zA-Z'’-]+ [A-Z]{1,3}(?=[,.])")
_LEGAL_MARKERS = re.compile(
    r"all rights reserved|some rights reserved|\bISBN\b|cataloguing|licen[cs]e|licensing|liable|"
    r"copyright|©|suggested citation|third-party materials|design and layout|printed in|warranty|"
    r"designations employed|expression of any opinion",
    re.IGNORECASE,
)
# A contents entry: a word, then a page number, then the next entry ("Background 15 2 Method 17").
_TOC_ENTRY = re.compile(r"[a-z)]\s(\d{1,3}|[ivx]{1,5})\s(?=[A-Z\d])")
_HAS_LETTER = re.compile(r"[A-Za-z]")
_NUMERIC_TOKEN = re.compile(r"[\d.,–%()/<>=:;+-]*\d[\d.,–%()/<>=:;+-]*")


def boilerplate_reason(text: str, cfg: Config = CONFIG) -> str | None:
    """Return why a chunk is boilerplate, or None if it is usable content.

    Why filter at all: a reference list, contents page or list of committee
    members is perfectly "grounded" text, so the judge would pass questions
    like "Which university is Paul Montgomery affiliated with?". Those teach a
    model nothing about the clinical domain and waste API budget.

    Each rule was calibrated on the 1,400 chunks of the three WHO guidelines
    (see README): none of them removes recommendation or evidence prose, which
    has a capitalised-word ratio near 0.07 and almost no author-initial patterns.
    """
    ing = cfg.ingest
    if (len(_CITATION_MARKERS.findall(text)) > ing.max_citation_markers
            or len(_AUTHOR_INITIALS.findall(text)) > ing.max_author_initials):
        return "references"
    if len(_LEGAL_MARKERS.findall(text)) > ing.max_legal_markers:
        return "front matter"
    if len(_TOC_ENTRY.findall(text)) > ing.max_toc_entries:
        return "contents page or table"  # "Title 15 Next" runs: contents pages and GRADE table rows
    tokens = text.split()
    words = [t for t in tokens if _HAS_LETTER.search(t)]
    if words and sum(1 for w in words if w[0].isupper()) / len(words) >= ing.max_capitalised_ratio:
        return "name list or table"  # committee lists, acronym glossaries, table headers
    if tokens and sum(1 for t in tokens if _NUMERIC_TOKEN.fullmatch(t)) / len(tokens) >= ing.max_numeric_ratio:
        return "data table"
    return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def make_splitter(cfg: Config = CONFIG):
    """600-char chunks with 80-char overlap.

    Why 600 characters: about 150 words, enough substance for two or three
    specific questions, small enough that questions stay precise rather than
    vague. Why 80 overlap: a sentence straddling a chunk boundary still appears
    whole in one of the two chunks. Separators prefer paragraph, then line,
    then sentence, then word breaks, so chunks rarely cut a sentence mid-word.

    Imported here, not at module level: the langchain_text_splitters package
    also imports sentence-transformers/torch, which is slow on a cold start.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.ingest.chunk_size,
        chunk_overlap=cfg.ingest.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        add_start_index=True,
    )


def chunk_documents(docs: list[Document], cfg: Config = CONFIG) -> list[Document]:
    """Split page documents into chunks; drop fragments and boilerplate.

    Chunks under ``ingest.min_chunk_chars`` are headers, page numbers or
    fragments with too little content to support a real question.
    """
    pieces = make_splitter(cfg).split_documents(docs)
    chunks: list[Document] = []
    dropped: Counter[str] = Counter()
    for piece in pieces:
        text = piece.page_content.strip()
        if len(text) < cfg.ingest.min_chunk_chars:
            dropped["too short"] += 1
            continue
        if cfg.ingest.filter_boilerplate:
            reason = boilerplate_reason(text, cfg)
            if reason:
                dropped[reason] += 1
                continue
        meta = dict(piece.metadata)
        stem = Path(meta["source"]).stem
        meta["chunk_id"] = f"{stem}-p{meta['page']}-{meta.get('start_index', 0)}"
        meta["chunk_index"] = len(chunks)
        chunks.append(Document(page_content=text, metadata=meta))
    detail = ", ".join(f"{n} {why}" for why, n in dropped.most_common()) or "none"
    print(f"Chunks created: {len(chunks)} (from {len(pieces)} raw splits; dropped {sum(dropped.values())}: {detail})")
    return chunks


def select_chunks(chunks: list[Document], max_chunks: int | None) -> list[Document]:
    """Pick ``max_chunks`` chunks spread evenly across every source document.

    Why not simply the first N: with files processed alphabetically, the first
    100 chunks would all come from one PDF and mostly from its front matter
    (title page, acknowledgements, contents). Allocating the budget to each
    source in proportion to its size, then taking evenly spaced chunks within
    it, gives a representative, deterministic sample of all the documents.
    """
    if max_chunks is None or max_chunks >= len(chunks):
        return list(chunks)
    by_source: dict[str, list[Document]] = {}
    for c in chunks:
        by_source.setdefault(c.metadata["source"], []).append(c)
    total = len(chunks)
    # Largest-remainder allocation so the shares sum exactly to max_chunks.
    raw = {s: max_chunks * len(cs) / total for s, cs in by_source.items()}
    alloc = {s: int(v) for s, v in raw.items()}
    for s in sorted(raw, key=lambda s: raw[s] - alloc[s], reverse=True)[: max_chunks - sum(alloc.values())]:
        alloc[s] += 1
    selected: list[Document] = []
    for source, cs in by_source.items():
        k = alloc[source]
        if k <= 0:
            continue
        step = len(cs) / k
        selected.extend(cs[int(i * step + step / 2)] for i in range(k))
    selected.sort(key=lambda c: c.metadata["chunk_index"])
    return selected


if __name__ == "__main__":
    documents = load_pdfs(CONFIG.paths.raw_dir)
    all_chunks = chunk_documents(documents)
    sample = select_chunks(all_chunks, CONFIG.run.max_chunks)
    per_source = Counter(c.metadata["source"] for c in sample)
    print(f"Selected for this run (run.max_chunks={CONFIG.run.max_chunks}): {len(sample)} -> {dict(per_source)}")
    lengths = [len(c.page_content) for c in all_chunks]
    print(f"Chunk length: min {min(lengths)}, mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}")
    first = sample[0]  # the first chunk this run will actually send to the model
    print("\n--- First chunk text ---")
    print(first.page_content)
    print("--- First chunk metadata ---")
    print(first.metadata)
