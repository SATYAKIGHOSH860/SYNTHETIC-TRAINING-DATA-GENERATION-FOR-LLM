"""
Step 1: Load source documents and split them into chunks for Q&A generation.

PDF / TXT / Markdown / DOCX -> LangChain Documents (one per page)
                            -> overlapping text chunks
                            -> boilerplate chunks removed.

Page numbers survive the split so every generated pair can cite its source.
Formats without real pages are cut into pseudo-pages so citations still mean
something and the rest of the pipeline never needs to know the difference.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

# NOTE: `langchain.text_splitter` was removed in LangChain 1.x (this project
# runs 1.3.18). The splitters moved to their own package. Importing the old
# path raises ModuleNotFoundError - verified on this machine.
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from config import RAW_DIR, use_utf8_console
except ImportError:  # running as `python src/ingest.py`
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import RAW_DIR, use_utf8_console

warnings.filterwarnings("ignore", category=DeprecationWarning)

# A page with fewer than this many characters is almost certainly a cover,
# a blank, or a full-page figure - nothing worth generating questions from.
MIN_PAGE_CHARS = 200

# Chunks shorter than this cannot support three specific questions.
MIN_CHUNK_CHARS = 100

# --- Boilerplate detection -------------------------------------------------
# Why this exists: a WHO guideline is ~20% front matter, bibliography and
# acknowledgements. Left in, the generator produces questions like "What is
# the ISBN of the print version?" - syntactically fine, worthless as clinical
# training data. Measured on this corpus: 20% of chunks. Filtering here also
# saves ~20% of the API budget.
#
# Each pattern was checked against real dropped/kept samples from these three
# PDFs to keep false positives near zero.

_FRONT_MATTER = re.compile(
    # Copyright / licensing / publishing block
    r"ISBN|CC BY-NC-SA|©|All rights reserved|WHO Press|book-orders"
    r"|rights and licensing|mediation rules|Suggested citation"
    r"|Cataloguing-in-Publication|Design and layout|Printed in"
    # WHO standard legal disclaimer. Every WHO PDF carries this verbatim, and
    # the first test run proved it survives: it produced questions like "Who
    # holds the ownership component in the work described in the WHO
    # disclaimer?" and "What do dotted and dashed lines on maps represent?" -
    # grammatical, well-formed, and completely useless as clinical data.
    r"|General disclaimers|designations employed|do not imply the expression"
    r"|legal status of any country|delimitation of its frontiers"
    r"|dotted and dashed lines|approximate border lines"
    r"|endorsed or recommended by WHO|preference to those of a similar nature"
    r"|preference to others of a similar nature"
    r"|distinguished by initial capital letters|third-party-owned component"
    r"|interpretation and use of the material|shall WHO be liable"
    r"|liable for damages|views expressed .{0,30}named authors",
    re.I,
)

# A bibliography entry = a link AND a journal-style citation. Requiring both
# keeps ordinary prose that merely cites a URL.
_HAS_LINK = re.compile(r"doi\.org|https?://", re.I)
_JOURNAL_CITE = re.compile(
    r"\b(19|20)\d{2}[;:]\s?\d+"
    r"|\bLancet\b|\bN Engl J Med\b|\bPLoS\b|\bBMJ\b|\bJ Acquir\b|\bClin Infect Dis\b",
    re.I,
)

# "Firstname Lastname (Affiliation" repeated - the signature of a contributor
# list. Deliberately strict: a looser comma/paren heuristic was tested and
# wrongly dropped real evidence text like "(RR 0.84; 95% CI 0.71-0.99) (106)".
_NAME_AFFILIATION = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-zA-Z.'\-]+)+\s*\(")


# Dot leaders ("Acknowledgements .......... iv") are the unambiguous signature
# of a contents page. Detecting the headings themselves would be wrong: an
# "Abbreviations" glossary and an "Executive summary" are real content, and a
# heading-based rule would throw both away. The leader dots only ever appear
# in a contents listing.
_TOC_DOT_LEADER = re.compile(r"\.{5,}")


# --- Supported formats -----------------------------------------------------
# PDF carries real page numbers. The others do not, so they are cut into
# pseudo-pages of this size, which keeps every citation in the dataset
# meaningful and lets the rest of the pipeline stay format-blind.
PAGE_CHARS = 3000

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx"}

FORMAT_NOTES = {
    ".pdf": "real page numbers",
    ".txt": "split into ~3000-character pages",
    ".md": "split into ~3000-character pages",
    ".markdown": "split into ~3000-character pages",
    ".docx": "split into ~3000-character pages",
}


def _paginate(text: str, source: str) -> list:
    """Cut unpaginated text into pseudo-pages on paragraph boundaries."""
    from langchain_core.documents import Document

    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    pages, current = [], ""
    for para in text.split("\n\n"):
        # Start a new page once this one is full, but never mid-paragraph -
        # a citation that points into the middle of a sentence is useless.
        if current and len(current) + len(para) > PAGE_CHARS:
            pages.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        pages.append(current.strip())

    return [
        Document(page_content=p, metadata={"source": source, "page": i})
        for i, p in enumerate(pages)
    ]


def pdf_page_count(path: Path) -> int:
    """Number of pages, without extracting any text (opening is ~0.3s)."""
    try:
        import pypdf

        return len(pypdf.PdfReader(str(path)).pages)
    except Exception:
        return 0


def load_one(
    path: Path, page_indices: list[int] | None = None, on_page=None
) -> list:
    """
    Load a single file into page Documents, dispatching on its extension.

    `page_indices` limits a PDF to specific pages. This matters enormously:
    pypdf's extract_text costs ~1.6 SECONDS per page, so a 592-page guideline
    takes 15.6 minutes to read in full - before a single API call - and the
    run then uses only the first few dozen chunks of it. Reading just the
    pages the run will actually use turns that into seconds.

    Page numbers in the metadata stay the true page numbers, so citations in
    the dataset still point at the right place in the original document.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        if page_indices is None:
            loaded = PyPDFLoader(str(path)).load()
            if on_page is not None:
                on_page(len(loaded))
            return loaded

        import pypdf
        from langchain_core.documents import Document

        reader = pypdf.PdfReader(str(path))
        total = len(reader.pages)
        out = []
        for i in page_indices:
            if not 0 <= i < total:
                continue
            try:
                text = reader.pages[i].extract_text() or ""
            except Exception:
                continue
            if text.strip():
                out.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "page": i},
                    )
                )
            if on_page is not None:
                on_page(1)
        return out

    if suffix == ".docx":
        import docx2txt

        result = _paginate(docx2txt.process(str(path)) or "", str(path))
        if on_page is not None:
            on_page(1)
        return result

    if suffix in {".txt", ".md", ".markdown"}:
        # errors="replace" rather than failing: a single bad byte in a large
        # text file should not cost the user the whole document.
        raw = path.read_text(encoding="utf-8", errors="replace")
        result = _paginate(raw, str(path))
        if on_page is not None:
            on_page(1)
        return result

    raise ValueError(f"Unsupported file type: {suffix}")


def classify_boilerplate(text: str) -> str | None:
    """Return a reason string if this chunk is boilerplate, else None."""
    if _TOC_DOT_LEADER.search(text):
        return "table_of_contents"
    if _FRONT_MATTER.search(text):
        return "front_matter"
    if _HAS_LINK.search(text) and _JOURNAL_CITE.search(text):
        return "reference_list"
    if len(_NAME_AFFILIATION.findall(text)) >= 3:
        return "contributor_list"
    # A chunk that is a quarter digits is a page-number index or a raw table.
    if sum(c.isdigit() for c in text) / max(len(text), 1) > 0.25:
        return "numeric_table"
    return None


def load_documents(
    data_dir: str | Path = RAW_DIR,
    page_budget: int | None = None,
    on_progress=None,
) -> list:
    """
    Load every supported document in the folder, one Document per page.

    Page numbers are preserved because a training set without citations is not
    defensible - a reviewer must be able to trace any generated answer back to
    the exact source page.

    Formats that are not paginated (plain text, Markdown, Word) are split into
    pseudo-pages of roughly PAGE_CHARS characters, so citations stay meaningful
    and the rest of the pipeline needs no special case.
    """
    data_dir = Path(data_dir)
    docs = []
    files = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ) if data_dir.exists() else []

    if not files:
        raise FileNotFoundError(
            f"No supported documents found in {data_dir}/\n"
            f"  Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}\n"
            f"  Upload files in the dashboard, or put them there and re-run."
        )

    # Work out which pages are worth reading at all. Without this a 592-page
    # PDF costs 15.6 minutes of text extraction to feed a run that will use
    # perhaps 25 chunks of it.
    page_plan: dict[Path, list[int] | None] = {p: None for p in files}
    if page_budget:
        pdf_sizes = {p: pdf_page_count(p) for p in files if p.suffix.lower() == ".pdf"}
        total_pdf_pages = sum(pdf_sizes.values())
        if total_pdf_pages > page_budget:
            for path, count in pdf_sizes.items():
                if count <= 0:
                    continue
                # Share the budget between documents in proportion to their
                # length, but always give each one at least a few pages so a
                # short document is never shut out entirely.
                share = max(3, round(page_budget * count / total_pdf_pages))
                if share >= count:
                    page_plan[path] = None
                    continue
                # Spread the sample across the whole document rather than
                # taking the front: the first pages of a guideline are title,
                # contents and copyright, which the boilerplate filter then
                # throws away - leaving the run with nothing.
                step = count / share
                page_plan[path] = [int(i * step) for i in range(share)]
            chosen = sum(
                len(v) if v is not None else pdf_sizes.get(k, 0)
                for k, v in page_plan.items()
                if k.suffix.lower() == ".pdf"
            )
            print(
                f"  Sampling {chosen} of {total_pdf_pages} PDF pages "
                f"(enough for ~{page_budget * 3} chunks; reading them all would "
                f"take about {total_pdf_pages * 1.6 / 60:.0f} minutes)"
            )

    # Total pages we intend to read, so the caller can show a real bar:
    # extracting text is ~1.6s per page, which is far too slow to leave the
    # dashboard sitting at 0%.
    planned_total = 0
    for path in files:
        chosen = page_plan.get(path)
        if chosen is not None:
            planned_total += len(chosen)
        elif path.suffix.lower() == ".pdf":
            planned_total += pdf_page_count(path) or 1
        else:
            planned_total += 1
    pages_done = 0

    for path in files:
        name = path.name
        try:
            def _page_tick(_n: int = 1) -> None:
                nonlocal pages_done
                pages_done += _n
                if on_progress is not None:
                    on_progress(pages_done, max(planned_total, 1), name)

            pages = load_one(path, page_plan.get(path), on_page=_page_tick)
        except Exception as exc:
            print(f"  SKIPPED {name}: could not read ({type(exc).__name__}: {exc})")
            continue
        if not pages:
            print(f"  SKIPPED {name}: no readable text")
            continue

        text_chars = sum(len(p.page_content.strip()) for p in pages)
        per_page = text_chars // max(len(pages), 1)

        # Store a clean source name rather than the full Windows path, so the
        # dashboard's per-source chart is readable.
        for p in pages:
            p.metadata["source_file"] = name

        docs.extend(pages)
        print(f"  Loaded {len(pages):>4} pages from {name}  ({per_page} chars/page)")

        # A scan with no text layer yields near-zero characters. pypdf cannot
        # fix that - the answer is a different PDF, not an OCR bolt-on.
        if per_page < 100:
            print(
                f"    WARNING: {name} averages only {per_page} chars/page. "
                + (
                    "This looks like a scanned image with no text layer; "
                    "pypdf cannot read it. Replace it with a text-based PDF."
                    if path.suffix.lower() == ".pdf"
                    else "There is very little text in this file."
                )
            )

    if not docs:
        raise RuntimeError("No readable pages loaded from any document.")

    print(f"\nTotal pages: {len(docs)}")
    return docs


# Kept so existing callers (grounding.py, the experiments, anything written
# before non-PDF support) keep working unchanged.
load_pdfs = load_documents


def chunk_documents(
    docs: list, chunk_size: int = 600, drop_boilerplate: bool = True
) -> list:
    """
    Split into 600-character chunks with 80-char overlap.

    Why 600: large enough to carry a complete clinical recommendation (the
    unit a useful question is asked about), small enough that the judge's
    groundedness check stays sharp - in a 2000-char chunk almost any answer
    looks supported, which makes the score meaningless.

    Why 80 overlap: a recommendation split mid-sentence across a boundary
    would otherwise produce two chunks that each answer half a question.
    """
    usable = [d for d in docs if len(d.page_content.strip()) >= MIN_PAGE_CHARS]
    print(f"Using {len(usable)} pages with real text (dropped {len(docs) - len(usable)})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=80,
        # Split on the most natural boundary available, degrading gracefully:
        # paragraph -> line -> sentence -> word.
        separators=["\n\n", "\n", ". ", " "],
        add_start_index=True,
    )
    chunks = splitter.split_documents(usable)

    valid = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]
    print(f"Created {len(valid)} chunks (dropped {len(chunks) - len(valid)} too short)")

    if not drop_boilerplate:
        return valid

    kept, reasons = [], Counter()
    for c in valid:
        reason = classify_boilerplate(c.page_content)
        if reason:
            reasons[reason] += 1
        else:
            kept.append(c)

    removed = len(valid) - len(kept)
    pct = removed / max(len(valid), 1) * 100
    print(f"Removed {removed} boilerplate chunks ({pct:.1f}%):")
    for reason, n in reasons.most_common():
        print(f"    {reason:<18} {n}")
    print(f"Content chunks ready: {len(kept)}")
    return kept


if __name__ == "__main__":
    use_utf8_console()

    print("=" * 70)
    print("STEP 1: INGEST - documents to chunks")
    print("=" * 70)

    docs = load_documents()
    chunks = chunk_documents(docs)

    print("\n--- Sample chunk (from the middle of the set) ---")
    mid = chunks[len(chunks) // 2]
    print(mid.page_content[:400])
    print(
        "\nMetadata:",
        {k: mid.metadata.get(k) for k in ("source_file", "page", "start_index")},
    )

    print("\n--- Per-source chunk counts ---")
    counts = Counter(c.metadata.get("source_file", "?") for c in chunks)
    for name, n in sorted(counts.items()):
        print(f"  {n:>5}  {name}")
