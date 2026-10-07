# Build prompt: Synthetic Training Data Generation Pipeline

Paste everything below the line into an AI coding assistant. It is written to
be followed top to bottom.

Every number in the "traps" section was measured on a real build, not guessed.
Those traps are the difference between this working and this wasting a day.

---

## WHAT TO BUILD

A pipeline that turns uploaded documents into a validated, deduplicated Q&A
training dataset, with a dashboard that shows what was removed and why.

**Documents in → quality-scored Q&A dataset out → dashboard proves it works.**

This is an M.Tech (AI & Data Science) industry project. It must be defensible
in a viva, so every threshold needs a stated reason and every claim needs a
measurement behind it.

---

## ENVIRONMENT

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.12.7 |
| GPU | **None.** CPU-only throughout |
| LLM | Groq API (remote) |
| Embeddings | `all-MiniLM-L6-v2`, local, CPU |

Never suggest CUDA, `device="cuda"`, bitsandbytes, or local LLM inference.
Project paths contain spaces and a `+`, so quote every path in shell commands.

---

## CRITICAL TRAPS — read before writing any code

These are real failures from a real build. Each one cost hours.

### 1. LangChain 1.x moved the text splitter
`from langchain.text_splitter import RecursiveCharacterTextSplitter` raises
`ModuleNotFoundError` on LangChain 1.3+. Use:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

### 2. The Llama model IDs are gone from Groq
`llama-3.3-70b-versatile` does not exist. A live `client.models.list()` on a
free account returned **no Llama chat models at all**. Working option at the
time of writing: `openai/gpt-oss-120b`.

**Verify before the first run** — Groq retires models regularly:
```python
from groq import Groq
for m in sorted(x.id for x in Groq(api_key=KEY).models.list().data):
    print(m)
```
Define the model ID as **one constant** so it changes in one place.

### 3. Groq free-tier limits — the daily token cap is invisible
| Limit | Value | In response headers? |
|---|---|---|
| Requests / day | 1,000 | yes |
| Tokens / minute | 8,000 | yes |
| **Tokens / day** | **200,000** | **NO — only inside the 429 body** |

The tokens-per-minute ceiling sets the **wall-clock floor**. A run using
189,645 tokens cannot finish faster than 189,645 ÷ 8,000 = **23.7 minutes**,
no matter how fast the code is. Optimise tokens, not code.

A first build ignored this, burned the whole daily allowance mid-validation,
and stalled for four hours leaving 141 of 263 pairs ungraded.

### 4. PDF text extraction costs ~1.6 SECONDS PER PAGE
Measured with `pypdf` on a 592-page, 10.8 MB WHO guideline: reading it in full
takes **939 seconds (15.6 minutes)** before a single API call — and the run
then uses ~25 chunks of it.

**You must budget pages.** Read only the pages the run will use, sampled
*evenly across the document* (the first pages are title, contents and
copyright, which the boilerplate filter then discards, leaving nothing).
This took ingest from 15.6 minutes to 40 seconds.

### 5. Windows console encoding kills long runs
WHO PDFs contain non-breaking hyphens (U+2011) and curly quotes. Windows
consoles default to cp1252 and `print()` dies with `UnicodeEncodeError` — at
the very end of a 20-minute run. Every entry point must call:
```python
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```

### 6. A total-only quality threshold lets hallucinations through
Tested with planted answers against a known passage:

| Planted defect | Score | Groundedness | Passes ≥4/6? |
|---|---|---|---|
| Correct answer | 6/6 | 2 | yes ✓ |
| **Invented dose "300 mg once daily"** | **4/6** | **0** | **YES — dangerous** |
| **Contradicts the passage threshold** | **4/6** | **0** | **YES — dangerous** |
| Truncated answer | 3/6 | 2 | no |

A total-only filter is not enough, and **a single veto on groundedness 0 is
not enough either**. Use a floor on *every* criterion:

```python
MIN_CRITERION_SCORES = {"groundedness": 2, "specificity": 1, "completeness": 1}
```

Groundedness sits at the strict **2 of 2**, because the 1 — "mostly supported,
adds a small unstated detail" — *is* the dangerous case: the unstated detail is
the invented dose. Measured on a real run: all 12 rejections failed on
groundedness, **11 of them scored 1 rather than 0**, and **11 of the 12 would
have passed a total-only filter at ≥ 4/6**. With a 0-only veto, eleven pairs
containing invented detail would have entered the dataset.

Write the rule **once**, in a `keep_decision()` function, and import it into
validation *and* every experiment. Three copies of a keep rule drift, and then
a sensitivity study measures something the pipeline does not do.

### 7. Streamlit never reloads your own modules
Streamlit re-runs `app.py` on every interaction but keeps imported modules in
`sys.modules` for the server's whole life. Editing `src/runner.py` has no
effect until restart, and the page fails confusingly — `AttributeError: no
attribute 'clear_inputs'`, then `TypeError: unexpected keyword argument
'on_progress'` — while the file on disk is perfectly correct.

Fix: compare file mtimes each rerun and `importlib.reload()` changed modules
in dependency order.

**Two details that cost real debugging time:**

1. **The reload must run BEFORE the `from config import ...` statements.**
   Those read straight out of the stale module object, so a check placed after
   them never gets the chance to run. The symptom is an `ImportError` for a
   name that is plainly present in the file.
2. **List every local module, including ones imported lazily inside a
   function.** A module left off the list keeps a stale copy for the life of
   the server. Renaming one constant in a lazily-imported helper broke every
   tab at once.

Stamp each module's file time onto the module object itself rather than using
a Streamlit cache, so the check works before `st.set_page_config` and needs no
Streamlit state.

### 8. Fixed `time.sleep()` between API calls is wrong in both directions
Too slow when the budget has headroom; far too fast once several calls land in
the same minute, earning 429s whose backoff costs minutes. Use a **token
bucket** that refills at the real tokens-per-minute rate.

Also: honour Groq's own retry hint. Its 429 says *"Please try again in
4m24.383s"* — parse and obey that instead of guessing.

### 9. The pipeline must mark itself finished
If the progress file is never set to "done", a **successful** run shows in the
dashboard as "presumed dead" once its heartbeat goes stale. Call the finish
function at the end of the run, and clear any previous run's timing plan at
the start (a stale plan made a fresh 8-chunk run report "8.4 minutes
remaining").

### 10. Deduplication over questions alone merges distinct facts
At cosine 0.85 over the **question only**, *"At what age is the first dose
given?"* and *"When is the second dose given?"* score **0.896** and one is
deleted — two different facts collapsed into one. The dedup rate looks good
while quietly destroying information.

Three fixes, all needed:

- threshold **0.90**, not 0.85
- embed the **question + answer**, because the distinguishing detail usually
  lives in the answer
- a **number guard**: never merge two answers whose stated numbers differ,
  however similar the wording ("25 g for adults" vs "15 g for children")

Also record the **lexical overlap** of the two questions beside the cosine
score on every removal. A pair at 0.95 cosine and 0.40 lexical is the entire
argument for using embeddings, stated as a measurement.

### 11. Never define the same tunable in two places
`main.py` declared its own `SIMILARITY_THRESHOLD = 0.85`, which silently
overrode the module's 0.90. A run used 0.85 while every comment, document and
log line said 0.90. The same duplicate default existed in the runner and in a
dashboard slider.

One definition, imported everywhere. This is the single cheapest bug class to
prevent and one of the most expensive to notice.

### 12. A long paid run must be resumable
Generation and judging are slow and metered. A run that dies at chunk 90 of
100 — quota exhausted, Ctrl-C, a crash — must not throw away the calls it
already paid for. Append each finished unit to a JSONL checkpoint as it
completes.

Guard it with a **fingerprint**: a hash of model, temperature, prompt version,
token caps and seed. Reuse a record only on an exact match, or you will blend
two different generators' output into one dataset and every number becomes
meaningless.

Deliberately **exclude the keep threshold and floors from the judge
fingerprint** — those are applied to the scores afterwards, so re-running at a
different threshold then costs nothing, which is what makes threshold
sensitivity experiments free.

Measured: a re-run after a completed run costs **1 API call and 578 tokens**,
against 104 calls and 62,500 cold.

### 13. Cap `max_tokens`, and set `reasoning_effort` where it exists
Groq admits a request only when `prompt + max_tokens` fits in what remains of
the per-minute budget, so an oversized cap makes every call wait longer for
admission. Set caps from measured usage (1024 generate, 512 judge here).

`reasoning_effort: "low"` on gpt-oss models cuts hidden reasoning tokens that
count against the same budget. Measured: generation tokens fell **32%** while
producing **more** pairs. Send it **only** to gpt-oss — it is a 400 on other
families, so branch on the model id.

### 14. Kappa is undefined when either rater is constant
If a human labels every pair identically, Cohen's kappa is not 0 — it is
undefined, because there is no chance-corrected scale to measure against.
`sklearn` will happily return `0.000`, which reads in a report as "the judge is
no better than chance" rather than "these labels cannot answer that".

Check **either** rater for zero variance, not both, and refuse to emit a
number. Print a warning naming the affected criteria. Exact agreement near 0%
on a 3-point scale is the tell — random labelling would land near 33%.

---

## ARCHITECTURE

```
Documents → chunks → boilerplate filter → LLM generates Q&A → LLM judge scores
                                                                    ↓
              Streamlit dashboard ← JSONL + stats ← semantic dedup
```

```
project/
├── data/
│   ├── raw/                    uploaded documents (emptied each session)
│   ├── cache/                  resume checkpoints (gitignored)
│   └── output/                 all generated artefacts (gitignored)
├── src/
│   ├── config.py               paths, model constant, key loading, UTF-8 fix
│   ├── ingest.py               documents → chunks + boilerplate filter
│   ├── generate.py             chunk → Q&A pairs
│   ├── validate.py             LLM-as-judge (batched)
│   ├── deduplicate.py          semantic dedup + diversity stats
│   ├── export.py               JSONL, ChatML, stats
│   ├── budget.py               token estimation, metering, rate limiting
│   ├── grounding.py            non-LLM hallucination check
│   ├── progress.py             progress file shared with the dashboard
│   ├── runner.py               uploads + background run control
│   └── checkpoint.py           fingerprinted resume for generate and judge
├── dashboard/
│   ├── app.py                  Streamlit dashboard
│   └── theme.py                design tokens and CSS
├── experiments/                threshold, dedup, judge-reliability studies
├── tests/test_pipeline.py      offline tests, no API calls
├── main.py                     runs the whole pipeline
├── pytest.ini
├── requirements.txt
└── .env                        GROQ_API_KEY (gitignored)
```

---

## BUILD ORDER

Build one file at a time. Run each and confirm sane output before continuing.

### Task 0 — `.gitignore` FIRST, before anything touches the API key
```
.env
venv/
__pycache__/
*.pyc
data/output/
.streamlit/
```

### Task 1 — `src/config.py`
- Paths resolved from `__file__`, not the cwd, so modules work from anywhere
- `GROQ_MODEL` as the single model constant
- `GEN_TEMPERATURE = 0.4` (variety without drift), `JUDGE_TEMPERATURE = 0.0`
  (a grade must be reproducible)
- `get_groq_key()` that fails loudly and distinguishes *missing* from
  *placeholder* (a real Groq key starts `gsk_` and is ~56 chars)
- `use_utf8_console()` — see trap 5
- `python src/config.py` prints the live model list and key status

### Task 2 — `src/ingest.py`
- Support **`.pdf`, `.docx`, `.txt`, `.md`**. PDFs keep real page numbers;
  the others are cut into ~3000-char pseudo-pages **on paragraph boundaries**
  so citations still point somewhere findable. Use `docx2txt` for Word.
- **Page budgeting (trap 4):** `load_documents(page_budget=N)` reads only N
  pages, split between documents in proportion to length, sampled evenly
  across each. Never take the first N pages.
- Chunk at **600 chars, 80 overlap**. Why 600: big enough to carry a complete
  recommendation, small enough that the judge's groundedness check stays sharp
  — in a 2000-char chunk almost anything looks supported.
- **Boilerplate filter.** Guideline PDFs are ~25% front matter, references and
  acknowledgements. Left in, you get *"What is the ISBN of the print
  version?"* and *"What do dotted and dashed lines on maps represent?"*
  Categories to detect:
  - `table_of_contents` — 5+ consecutive dots (dot leaders). This is the only
    safe signal; detecting headings like "Abbreviations" wrongly drops real
    glossaries and executive summaries.
  - `front_matter` — ISBN, copyright, licence, and the WHO legal disclaimer
    ("designations employed", "dotted and dashed lines", "legal status of any
    country", "shall WHO be liable")
  - `reference_list` — requires **both** a link **and** a journal-style
    citation, so ordinary prose citing a URL survives
  - `contributor_list` — 3+ matches of `Firstname Lastname (Affiliation`.
    A looser comma/paren heuristic was tried and wrongly dropped real evidence
    text like `(RR 0.84; 95% CI 0.71-0.99) (106)`.
  - `numeric_table` — >25% digits
- Report per-file page counts and warn when a PDF averages <100 chars/page
  (a scan with no text layer; pypdf cannot fix that — the answer is a
  different PDF, not OCR)

### Task 3 — `src/generate.py`
- One call per chunk → JSON array of pairs
- **Keep the prompt short.** Wall-clock is token-bound, so template size is
  wall-clock. A verbose 373-token template compressed to ~180 without losing
  a single rule.
- Prompt rules: answers only from the passage; questions self-contained (never
  "this passage" / "the text"); no questions about the document itself (ISBN,
  page numbers, contact details, figure numbers); 1–3 sentence answers; one
  each of **factual / definitional / procedural**
- Strip ```json fences before parsing — models add them regardless
- Fall back to the outermost `[...]` span if the whole string won't parse;
  handle `{"pairs": [...]}` too
- **Regex backstops** for rules the model breaks anyway: drop questions
  matching passage-references, and document-metadata (`which page`,
  `contents list`, `telephone`, `disclaimer`, …)
- Attach `source`, `page` (+1, pypdf is 0-indexed), `question_type`,
  `answerable`, and `chunk_text` (needed by the judge, stripped before export)
- Minimum lengths: question 15 chars, **answer 25** — a 15-character answer
  ("At six weeks") is not self-contained training data
- Send `max_tokens` and, for gpt-oss only, `reasoning_effort` (trap 13)
- **Abstention examples.** For ~10% of chunks (seeded, so runs reproduce), make
  a second call asking for ONE plausible, on-topic question the passage
  **cannot** answer. Pair it with a fixed refusal string defined in config —
  not generated — so the model learns one phrase rather than fifty paraphrases.
  A dataset of only answerable questions teaches a model that every question
  has an answer, so it invents one when it does not know.
- Checkpoint each finished chunk (trap 12)

### Task 4 — `src/validate.py`
- Three criteria, 0–2 each, total 0–6. Keep requires the total **and every
  per-criterion floor** (trap 6) — expose it as one shared `keep_decision()`.
- **Use a different model from the generator.** A model grading its own output
  shows self-preference bias; with one model doing both, the judge awarded 6/6
  to nearly everything and the pass rate meant nothing. This build used
  `openai/gpt-oss-120b` to generate and `qwen/qwen3.8-27b` to judge — different
  families, and each gets its own rate-limit bucket so judging does not eat the
  generator's quota.
- **Abstention pairs need their own check**, not the 0–6 rubric: a refusal has
  nothing to ground and answers nothing by design. Ask two booleans — is the
  passage genuinely unable to answer it, and is the question plausible — and
  give those pairs `quality_score = None` so they never enter the average.
  Verify the fixed answer string in code, not by paying a model to check your
  own literal.
- Checkpoint each judged pair, keyed by question (trap 12).
- **Batch by passage.** Pairs from one chunk share their passage, so judge all
  three in one call: sending the rubric and context once instead of three
  times cut judging tokens ~53%.
- Ask for a `reason` **only when a pair scores below 2** on something — a
  reason for a perfect pair explains nothing and costs output tokens.
- Map results back by an explicit `index` field, not list position, so a
  mis-ordered response can't attach a score to the wrong pair.
- **If a whole batch fails, fall back to judging its pairs one at a time.**
  A single malformed response otherwise leaves pairs silently ungraded.
- A failed call is **"not assessed"**, never a quality rejection — mixing them
  inflates the pass rate. Report the two separately.
- Write rejects to `rejected_pairs.jsonl` and *every* graded pair with
  sub-scores to `graded_pairs.jsonl` (the experiments re-filter it offline,
  so threshold sweeps cost no API calls).

### Task 5 — `src/deduplicate.py`
- `all-MiniLM-L6-v2`, `normalize_embeddings=True` so a dot product **is**
  cosine similarity
- Threshold **0.90** over **question + answer**, with a **number guard**
  (trap 10), greedy first-wins so re-running is deterministic
- Merge results in **generation order**. Grouping abstention pairs ahead of
  answerable ones lets a refusal win a collision against a good pair, which
  teaches the model to refuse a question the corpus answers.
- Record each removal with the question it duplicated, the similarity **and
  the lexical overlap of the two questions** (trap 10). Verified catches:
  - "At what CD4 count is a patient considered to have advanced HIV disease?"
    vs "What is the CD4 threshold for advanced HIV disease?" → **0.911**
  - "What is the recommended treatment for cryptococcal meningitis in
    HIV-positive adults?" vs "How should cryptococcal meningitis be treated in
    adults living with HIV?" → **0.927**

  These share few tokens — string matching keeps both. That difference is the
  entire point of this step.
- Diversity statistic: distribution of two-word question openers, to catch a
  dataset that is 80% "What is…"

### Task 6 — `src/budget.py`
- Constants for the three limits (trap 3) and measured per-chunk costs
- `estimate_run(chunks)` → tokens, requests, **and minimum minutes** (include
  the ~35s Python startup or a "2 minute" run feels like a broken promise)
- `preflight()` printed before spending anything
- `TokenMeter` — accumulates real usage from `response.usage`; **thread-safe**,
  since workers record concurrently
- `TokenBucket` — refills at the real rate, `acquire(tokens)` blocks until
  budget exists (trap 8)

### Task 7 — `src/grounding.py` — the independent check
The judge is itself a language model; an examiner will press on that. Check
the same thing **mechanically, with no model**:
- **Lexical overlap** — share of the answer's content words present in the
  source (stopwords removed)
- **Unsupported numbers** — numbers in the answer absent from the source.
  This is the sharp one: prose can be paraphrased, a dose cannot.

Both are needed. A planted answer with the **wrong threshold** ("above 7"
instead of "above 3") scored **1.00 overlap** — every word matched — and only
the number check caught it.

Report agreement with the LLM judge. Disagreements are where a human should
look. Be honest that low overlap often means paraphrase, not hallucination.

### Task 8 — `src/export.py`
- Strip `chunk_text` and the raw scores dict; export exactly `question`,
  `answer`, `source`, `page`, `question_type`, `quality_score`
- Assert on every run that internal fields did not leak
- ChatML variant for SFTTrainer, with the safety disclaimer as the system
  message
- `pipeline_stats.json` with every count, rate, distribution and the grounding
  summary — the dashboard reads only this
- A `run_health` block that says plainly when a run was incomplete, rather
  than letting "100% pass rate over 122 of 263 pairs" read as "100% pass rate"
- Disclaimer defined **once** and reused by README, dashboard and dataset card
- Make the disclaimer **adapt to the domain** — the same code runs over banking
  circulars as readily as clinical guidelines, and asserting "not medical
  advice" over a banking corpus is simply wrong

### Task 9 — `src/progress.py` and `src/runner.py`
- Progress via one small JSON file written atomically, with a **heartbeat** so
  a dead run is distinguishable from a slow one
- A **timing plan in seconds**, not fixed percentages: the 35s startup is 20%
  of a 3-minute run and 7% of a 9-minute one, so fixed weights cannot be right
  for both. Clear the plan when a run starts (trap 9).
- ETA that blends plan with measured rate, trusting measurement more as the
  run progresses
- Runs launch as a **detached subprocess** (`PYTHONUNBUFFERED=1`, else the log
  stays empty for twenty minutes). State in a file, not session state, so a
  browser refresh still sees the run.
- Uploads: max 8 files, 50 MB each. **Validate on arrival** — probe only the
  first ~3 pages to answer "does this have a text layer?"; the answer is the
  same and the full parse costs 1.6s/page.
- Report upload progress **weighted by bytes**, not file count — a 40 MB PDF
  and a 2 KB text file are one file each but nothing like the same work.

### Task 10 — `main.py`
Runs all five stages. Parameters at the top: `questions_per_chunk`,
`min_quality_score`, `similarity_threshold`, `max_chunks`, `push_to_hf`.
- Compute the page budget from `max_chunks` **before** ingest
- Sample chunks **evenly across the corpus**, not the first N — otherwise every
  chunk comes from one document and the per-source chart shows one bar
- One shared TokenBucket for both API stages
- Wrap in try/except and mark the run failed on any exception

### Task 11 — `dashboard/app.py`
Auto-reload src modules on mtime change (trap 7). Eight tabs, plus a download
bar above them (dataset JSONL + everything ZIP) so the files are one click
away from anywhere:

1. **Upload Files** — run-size presets with live time/token estimates, then
   the uploader. **Loading the files starts the pipeline immediately**; there
   is no separate start button, so the settings must sit *above* the upload
   box. Real byte-weighted progress bar during load.
2. **Run Pipeline** — status view: progress bar, stage, elapsed, ETA, live
   log, Stop button. Render the progress block on *both* tabs, because
   Streamlit cannot switch tabs programmatically.
3. **Pipeline Overview** — metric cards, survival funnel, per-source and
   question-type charts, opener diversity
4. **Quality Explorer** — score histogram over *every graded pair*, plus a
   minimum-score slider and text search, with the count updating live. Also a
   **Status filter** (answerable / abstention) and a toggle to show each
   pair's source passage.

   > The Status filter is not optional. Abstention pairs carry no score, so
   > without it, dragging the minimum-score slider off zero silently deletes
   > every one of them — which looks like a quality judgement but is a missing
   > number. Apply the score filter only to rows that have a score.
5. **Evidence** — rejected pairs with reasons beside kept ones; a chart of
   which criterion fails most often; each removed duplicate beside what it
   duplicated with **both** its cosine and lexical overlap; full sortable
   tables of everything; and an abstention section showing each unanswerable
   question with the judge's verdict
6. **Hallucination Check** — grounding bands, overlap histogram, grounding by
   question type and source, judge-vs-mechanical scatter, flagged numbers
7. **Judge Reliability** — does the judge discriminate (score spread), does it
   agree with the mechanical check, and a **blind hand-labelling tool**: a
   seeded sample, the judge's score hidden, three 0–2 radio columns, saved
   after every pair, then Cohen's kappa (trap 14)
8. **Downloads** — one-tap ZIP of everything (with a README inside), the
   dataset as JSONL *and* CSV (UTF-8 **with BOM**, or Excel mangles accents),
   a per-file preview, and each file individually with size, row count and
   timestamp

Use `@st.fragment(run_every=2)` for the progress block. A `time.sleep()` +
`st.rerun()` loop re-executes the whole script — measured at 1.33s per rerun,
that rebuilds all seven tabs to move a progress bar.

Empty `data/raw` once per server start (`@st.cache_resource`), so reopening the
project always asks for documents rather than silently reusing whatever was
left behind.

### Task 12 — experiments
- **Threshold sensitivity** — sweep min score **3, 4, 5 AND 6**. On a corpus
  where the judge is generous, 3/4/5 land within a pair of each other and the
  table says nothing; only 6 bites. A sensitivity experiment that shows no
  sensitivity is not an experiment.
  Report the **floors** separately from the total: on a real corpus the total
  threshold moved the dataset by 3 pairs while the floors moved it by 10–11 at
  every threshold. If you only sweep the total, you will credit the wrong
  mechanism.
- **Dedup sensitivity** — 0.75 / 0.85 / 0.90 / 0.95, reporting survivors and
  showing borderline removals
- **Judge reliability** — score a **fixed 40-pair** sample by hand (not a
  percentage: 20% of a 60-pair run is 12 pairs, and a kappa over 12 items
  supports no claim). Resumable, fixed seed, and the judge's score hidden while
  labelling. **Report it honestly** — if agreement is fair or worse (<0.4), say
  so and downweight the judge's numbers. That paragraph is worth more than
  another feature. And if the labels have no variance, refuse to emit a number
  at all (trap 14).

### Task 13 — tests
Offline, no API calls, fast enough to run constantly. Cover the places where a
silent change is most expensive: the keep rule at every floor boundary, the
dedup number guard, that a true reword is still removed, the seeded abstention
selection, JSON extraction with fences and stray prose, that export strips
internals, and checkpoint round-trip / fingerprint mismatch / truncated final
line. Stub the API boundary — testing Groq is not testing your code.

---

## SAFETY

Sources may be clinical. The README, dashboard and any dataset card must carry:
*educational and research use only, not professional advice, not for
decision-making, no patient data used.* Generate it from one constant so all
copies stay identical.

---

## WHAT "GOOD" LOOKS LIKE

From a real complete run (45 chunks, 1 document):

| Measure | Value |
|---|---|
| Raw pairs | 116 (2.58/chunk) — 111 answerable + 5 abstention |
| Graded by judge | 116 (all; 0 judge failures) |
| Passed the keep rule | 104 (89.7%) |
| Rejected | 12 — **all on the groundedness floor** |
| Duplicates removed | 1 (1.0%) at cosine 0.90 |
| **Final dataset** | **103 pairs** |
| Average quality | 5.94 / 6 (answerable only) |
| Sub-scores | groundedness 2.00, specificity 1.97, completeness 1.97 |
| Grounding (non-LLM) | 84.7% strongly grounded, **0** unsupported numbers |
| Judge vs mechanical agreement | 84.7% |
| Distinct two-word openers | 56 (most common 14.6%) |
| Runtime | 10 min 9 s — 104 calls, 62,500 tokens (31% of the daily budget) |
| Re-run from checkpoint | **1 call, 578 tokens, 16 s** |

After prompt compression, page budgeting and `reasoning_effort: low`, a
12-chunk run finishes in **~3 minutes**.

**A near-100% pass rate is a finding, not a success.** It means the total
threshold is doing little on that corpus. Say so, and show the sweep — then
check whether something *else* is doing the filtering. Here it was the
per-criterion floors: the total moved the dataset by 3 pairs, the floors by
10–11.

**The most valuable single number in this table is "0 unsupported numbers",**
because no language model produced it.

---

## VERIFICATION CHECKLIST

- [ ] `python src/config.py` lists live models and confirms the key
- [ ] `python src/ingest.py` gives readable chunks; boilerplate removed ~25%
- [ ] Generation smoke-tested on 5 chunks before spending hundreds of calls
- [ ] Judge tested against **planted bad pairs** (invented dose, contradiction,
      truncation) — it must reject them
- [ ] Dedup tested on **planted rewordings** — must catch ~0.9 similarity
- [ ] Export asserts no `chunk_text` or raw scores leaked
- [ ] A large PDF (500+ pages) ingests in under a minute, not 15
- [ ] A completed run leaves the dashboard saying "finished", not "presumed dead"
- [ ] Editing a `src/` file while the dashboard runs does not crash the page
- [ ] Dashboard renders with zero exceptions and no data present
- [ ] Killing a run mid-way and restarting it **resumes** rather than
      re-paying for the same calls
- [ ] Changing the model or prompt **invalidates** the checkpoint rather than
      blending two generators' output
- [ ] The same tunable is not defined in two files (grep for the number)
- [ ] `python -m pytest` passes, offline, with no API key set
- [ ] Every local module the dashboard imports — **including lazy, in-function
      imports** — is in the auto-reload list
- [ ] The kappa tool refuses to print a number when the labels have no variance
- [ ] `requirements.txt` pins every **directly imported** package and nothing
      it does not use; a separate deploy file pins **CPU-only torch**
      (`--extra-index-url https://download.pytorch.org/whl/cpu`) — the default
      Linux wheel pulls 2–3 GB of unused CUDA libraries

---

## DEPLOYMENT NOTE

Do **not** target Vercel. Measured footprint is **1,351 MB** (torch alone
495 MB) against Vercel's 250 MB serverless limit; runs last minutes against a
60–300s execution cap; and the design needs a persistent filesystem and a
long-running background process, neither of which serverless provides.

Use Hugging Face Spaces (built for this), Render, Railway or Fly.io.

And note: **the web framework is not the bottleneck.** The pipeline runs as a
separate process; the framework only polls a file. Runtime is set by the
8,000 tokens/minute ceiling, which follows you to any host.
