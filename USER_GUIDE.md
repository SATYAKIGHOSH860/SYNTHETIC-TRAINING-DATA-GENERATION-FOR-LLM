# Complete guide: every screen, every control, every likely question

Two parts:

- **Part 1 — Operating the app.** Every tab, every button, what it does and why
  it is there.
- **Part 2 — The viva.** Every question an examiner or interviewer is likely to
  ask, with an answer grounded in this project's actual measurements.

All numbers quoted here come from the run recorded in
`data/output/pipeline_stats.json` (45 chunks of `farmerbook.pdf`, 2026-10-06).
If you re-run the pipeline, re-read those numbers before your viva — they are
measurements, not decoration.

---

## What this project does, in one paragraph

You give it documents. It cuts them into passages, asks a large language model
to write question-and-answer pairs from each passage, asks a *different* model
to grade every pair against a published rubric, throws out the ones that fail,
removes near-duplicates using sentence embeddings, and writes a clean training
dataset in JSONL and ChatML. A dashboard then shows you exactly what was
removed and why — because the claim "this data is clean" is worth nothing
unless you can show the evidence.

**The one-line version:** PDFs in → quality-scored Q&A training dataset out →
dashboard proves it worked.

---

# PART 1 · OPERATING THE APP

Start it by double-clicking **`START_HERE.bat`**, or:

```bash
streamlit run dashboard/app.py
```

## Always on screen

### The sidebar — "This dataset"

Everything that produced the numbers you are looking at, so you can never
misread a chart because you forgot which run it came from.

| Row | Meaning |
|---|---|
| Generated | When the stats file was written |
| Generator | The model that wrote the pairs (`openai/gpt-oss-120b`) |
| Judge | The model that graded them (`qwen/qwen3.8-27b`) — deliberately different |
| Embeddings | `all-MiniLM-L6-v2`, used for deduplication |
| Chunks processed | How many passages went in |
| Keep threshold | The minimum total score, out of 6 |
| Criterion floors | The per-criterion minimums — `groun 2, speci 1, compl 1` |
| Dedup threshold | Cosine similarity at or above which two pairs count as duplicates |
| Raw pairs / Final pairs | Before and after all filtering |
| Pass rate | Share of graded pairs that survived |
| API calls / Tokens used / Wall time | What the run actually cost |
| Abstention pairs | How many "the document cannot answer this" examples are in the dataset |

**"Source documents"** expander — which file contributed how many pairs.

**"Reload results"** — the dashboard caches output files by modification time.
Press this if a run finished in another browser tab and the numbers look stale.

### The download bar, above the tabs

Two buttons visible from every tab, so the files are never more than one click
away:

- **⬇ Dataset (JSONL)** — the deliverable on its own.
- **⬇ All files (ZIP)** — every output file plus a `README.txt` inside the
  archive explaining what each one is. That README matters: it means the ZIP is
  still explicable after you have emailed it to someone who has never seen this
  dashboard.

---

## TAB 1 · Upload

The only tab you need to use. It is laid out as three numbered steps.

### Step 1 — "How long should this take?"

**Run size (radio).** Four presets:

| Preset | Chunks | Roughly |
|---|---|---|
| Quick (~3 min) | 12 | 3 min |
| Standard (~5 min) | 25 | 5 min |
| Large (~9 min) | 45 | 9 min |
| Custom | 5–150, you choose | — |

Picking **Custom** reveals a **"Chunks"** slider (5–150). Everything else shows
a **"Chunks to process"** metric instead.

> **Why a cap at all?** The binding constraint on this project is not compute,
> it is the Groq free tier's **200,000 tokens per day**. A 150-chunk run is
> roughly a sixth of your daily allowance. The cap stops you discovering that
> at chunk 140.

**"Minimum quality score" (slider, 0–6).** The total the judge must award for a
pair to be kept. Default **4**.

**"Duplicate similarity" (slider, 0.70–0.99).** Cosine similarity at or above
which two pairs are treated as duplicates. Default **0.90**.

**The cost cards.** Before you spend anything, the app shows: chunks to
process, estimated time, approximate Q&A pairs, estimated tokens, and what
percentage of the daily quota that is. Nothing is spent without being shown
first.

### Step 2 — "Your Groq API key"

Paste a key (`gsk_...`). If `.env` already holds one, the app says so and you
can leave this blank.

> **The key is never written to disk by the dashboard.** It is passed into the
> pipeline's child process through its environment only, and lives for that
> session. If you are running this on a shared or deployed machine, that is the
> property that matters.

### Step 3 — "Add your documents"

**The uploader.** 1–8 files, up to 50 MB each: **PDF, DOCX, TXT, MD**. Each
file is read as it lands, so an unsupported type, an empty file, or a scanned
PDF with no text layer is caught immediately — not after twenty minutes of
paid API calls.

**"Add to the current list instead of replacing it" (checkbox).** Off by
default, so loading new files replaces the old ones.

**The Load button** — labelled with the file count and the time estimate.

> **Loading the files starts the pipeline immediately.** There is no separate
> start button. That is why every setting sits *above* this step: by the time
> you load, you have already chosen them.

**"Currently loaded (n)"** lists what is queued.

> **The input folder is emptied when the app starts.** Every session begins by
> asking what to process, so a run can never quietly use documents you replaced
> last week. The trade-off: if you put files in `data/raw/` by hand for a
> command-line run, launching the dashboard will delete them. Keep a master
> copy elsewhere.

---

## TAB 2 · Run Status

A status view, not a control panel.

**While a run is in progress:** a progress bar with the current stage, four
cards (stage, elapsed, estimated remaining, pairs so far), a **"Stop this run"**
button, and a **"Live log"** expander streaming the pipeline's output.

**When idle:** how the last run ended, and **"Settings used by the last run"**,
plus a **"Log from the last run"** expander.

> The run happens in a **separate process**, tracked through a progress file on
> disk. Closing the browser does not kill it, and reopening the page picks it
> back up. This is also why the progress bar survives a page refresh.

---

## TAB 3 · Overview

**Four headline cards** — Final pairs, Avg quality, Pass rate, Duplicates
removed. Each has a tooltip stating precisely what it covers; *Avg quality*,
for instance, explains that abstention pairs are excluded because they carry no
score.

**"Funnel — what survived each stage"** — raw generated → passed judge → after
dedup, with percentages of the original.

**"Average sub-score"** — groundedness, specificity and completeness, each out
of 2. This tells you *how* the data is good or bad, which a single average
cannot.

**"Pairs per source document"** — useful when one document dominates the
dataset.

**"Question-type distribution"** — factual / definitional / procedural /
unanswerable. A dataset that is 90% factual is a monotonous dataset.

**"Question diversity"** — the most common two-word openers. This catches the
classic failure of synthetic generation: 200 questions that all begin "What
is…".

---

## TAB 4 · Explore

**"Score Distribution"** — a histogram over *every graded pair*, kept and
rejected. Showing only the survivors would hide the filter's effect entirely.

**"Filter The Dataset"**

- **"Minimum quality score" (slider)** — drag it up and watch weaker pairs
  disappear. The count updates live, so the size-versus-quality trade-off is
  visible rather than asserted.
- **"Search questions and answers"** — plain text search.
- **"Question type"** and **"Source document"** multi-selects.
- **"Status"** multi-select — *Answerable* and *Unanswerable (abstention)*.

> **Why Status exists.** Abstention pairs carry no 0–6 score. Without this
> control, dragging the minimum-score slider off zero would silently delete
> every one of them, which looks like a quality judgement but is really a
> missing number. The slider now applies only to pairs that have a score.

- **"Show the source passage with each pair" (toggle)** — displays the passage
  each pair was generated from, so you can check the answer against it
  yourself. The passage is looked up from `graded_pairs.jsonl`, because the
  exported dataset deliberately strips it.

**Three cards** — pairs shown, share of dataset, average score of what is
shown.

**The results list** — paginated, 25 at a time. Each pair expands to show the
answer, source, page and type. Abstention pairs are labelled `[abstention]`
instead of a score.

---

## TAB 5 · Evidence

The tab that proves the filtering did something.

**"Kept vs Rejected"** — two columns. On the left, the lowest-scoring rejected
pairs with the judge's reason and its three sub-scores. On the right, the
highest-scoring survivors.

**"Which criterion fails most often"** — a bar chart counting, across all
rejected pairs, how often each criterion scored 0 and how often it scored 1.
This answers a question the reject reasons alone cannot: *what is the generator
actually bad at?*

**"Most common rejection reasons"** — the judge's own words, grouped.

**"Semantic duplicates removed"** — each removed question beside the one it
duplicated, with **two** numbers: the cosine similarity *and* the lexical
overlap of the two question strings. A pair at 0.95 cosine and 0.40 lexical is
the entire argument for using embeddings instead of string matching, stated as
a measurement.

**"See everything"** — three expanders with full sortable tables: all rejected
pairs, all pairs the judge never graded (kept separate — those are not quality
rejections), and all kept pairs.

**"Unanswerable questions: teaching the model to abstain"** — four counters
(written, verified unanswerable, failed the check, in the dataset), the exact
abstention string, and every abstention pair with the judge's verdict on all
three checks.

---

## TAB 6 · Grounding

The only quality signal in the project that does **not** come from a language
model. It compares each answer against its source passage arithmetically.

- **"Grounding strength"** — share of answers strongly / moderately / weakly
  grounded by word overlap.
- **"How much of each answer appears in its source"** — the overlap histogram.
- **"Grounding by question type"** and **"by source document"** — where the
  weak answers are concentrated.
- **"Does the LLM judge agree with the mechanical check?"** — a scatter of
  judge score against word overlap. **The disagreements are the point**: pairs
  the judge called perfect but which share few words with the source are either
  good paraphrases or quiet hallucinations, and only a human can tell.
- **Answers containing unsupported numbers** — any figure in an answer that
  does not appear in its passage. This is the single highest-value check in the
  project: a fabricated dose or a fabricated threshold is the worst thing this
  pipeline could emit.
- **"Lowest word overlap — review these by hand"** — a ranked worklist.

---

## TAB 7 · Judge Reliability

"How much should you trust the scores?" — three checks, reported whatever they
say.

**1. Does the judge discriminate?** Pairs graded, share that scored a full 6/6,
how many distinct scores it used, judge failures. If almost everything scores
6/6 the dashboard says so plainly: a judge that rarely marks anything down is
not filtering much, and the honest reading is that the threshold is easy on
this corpus, not that the data is flawless.

**2. Agreement with the non-LLM check** — how often the judge and the
mechanical grounding check reach the same verdict.

**3. Agreement with a human (Cohen's kappa)** — and the blind labelling tool:

- **"The rubric you are scoring against"** expander — the same 0–2 definitions
  the judge was given.
- **"Jump to a pair"** selectbox — shows which are done.
- The passage, the question and the answer — **but never the judge's score**.
  That is what makes it blind, and it is the whole validity of the measurement.
- Three columns of radio buttons: **Groundedness, Specificity, Completeness**,
  each 0/1/2 with the rubric wording as captions.
- **"Save and go to next"** — saved to disk after every pair, so you can stop
  and come back.
- **"Compute agreement (Cohen's kappa)"** — enabled once at least 10 pairs are
  labelled.
- **"Start again: clear my labels"**.

> **Score each pair honestly and differently.** If you give every pair the same
> three numbers, kappa is mathematically undefined — a rater who never varies
> provides no chance-corrected scale to measure against. The tool detects this
> and refuses to print a number rather than reporting a misleading `0.000`.

---

## TAB 8 · Download

- **"⬇ Download everything"** — one ZIP, every file, plus an explanatory
  README inside.
- **"The dataset, in whichever format you need"** — JSONL for training
  pipelines, CSV for Excel (UTF-8 with BOM so accented characters survive).
- **"Preview a file"** — pick any output file and see its first 3 / 10 / 25
  records as a table, without downloading it.
- **"Individual files"** — each file with its size, row count, when it was
  written, and what it is for.

> Files are re-read from disk on every visit, so a download is always the
> latest run rather than a cached copy.

---

## Command line

Everything is available without the dashboard:

```bash
python src/budget.py                        # what will this run cost?
python main.py --max-chunks 45              # run the pipeline (resumes)
python main.py --max-chunks 45 --fresh      # ignore checkpoints, redo everything
streamlit run dashboard/app.py              # just the dashboard

python experiments/threshold_sensitivity.py # offline, free
python experiments/dedup_sensitivity.py     # offline, free
python experiments/judge_reliability.py           # hand-score the sample
python experiments/judge_reliability.py --report  # compute kappa

python -m pytest                            # 24 offline tests
python -m pytest -m "not slow"              # skip the two that load the embedder
```

---

# PART 2 · THE VIVA

Answers are grounded in this project's actual measurements. Where a number is
quoted, it comes from `data/output/pipeline_stats.json` or an
`experiment_*.json` file — re-read them after any re-run.

**One piece of advice before the specifics.** The strongest thing in this
project is not a feature, it is the honesty: the places where you measured
something, found it unflattering, and reported it anyway. Lead with those.
Examiners have seen a hundred "chat with your PDF" projects; they have seen
very few students who can say "my threshold is doing almost nothing, and here
is the experiment that shows it."

---

## A. The project in general

**Q1. Explain your project in two minutes.**

Any organisation that wants a domain-specific AI assistant needs labelled
question-and-answer training data. They have documents; they do not have
training pairs, and writing them by hand does not scale. This pipeline
automates it: documents are split into passages, a large language model writes
Q&A pairs from each passage, a *second, different* model grades every pair
against a published rubric, failures are rejected, near-duplicates are removed
with sentence embeddings, and a clean dataset is exported in JSONL and ChatML.

The part that makes it a project rather than a script is that it **measures
whether the cleaning worked** and shows the evidence: what was rejected, why,
what was removed as a duplicate, and what the judge itself can and cannot be
trusted to say.

**Q2. What problem does it solve that an existing tool does not?**

Generating synthetic data is easy; the hard part is knowing whether it is any
good. Most tutorial pipelines write the model's output straight to a file. This
one applies a published rubric, a second model, an independent non-LLM check,
and reports the pass rate honestly — including when the filter turns out to be
doing very little.

**Q3. Who would use it?**

Anyone with domain documents and no training data: a hospital with clinical
guidelines, a bank with regulatory circulars, an agriculture extension service
with field handbooks. All three have been run through it. The pipeline is
domain-agnostic — nothing in the code knows what the documents are about.

**Q4. Why is it called "synthetic" data?**

Because no human wrote the questions or the answers. They are generated by a
model from source text. That is the value — it scales — and also the risk,
which is what the validation layer exists to control.

**Q5. Walk me through the architecture.**

Five stages, each its own module:

1. **Ingest** (`src/ingest.py`) — load PDF/DOCX/TXT/MD, drop boilerplate, split
   into 600-character chunks with 80-character overlap.
2. **Generate** (`src/generate.py`) — one API call per chunk returns a JSON
   array of Q&A pairs; a share of chunks also get an abstention example.
3. **Validate** (`src/validate.py`) — a different model scores each pair 0–2 on
   three criteria; keep requires both the total and every per-criterion floor.
4. **Deduplicate** (`src/deduplicate.py`) — MiniLM embeddings, cosine ≥ 0.90,
   with a number guard.
5. **Export** (`src/export.py`) — JSONL, ChatML, statistics, optional
   HuggingFace push.

Supporting modules: `budget.py` (token accounting), `grounding.py`
(non-LLM check), `checkpoint.py` (resume), `progress.py` and `runner.py`
(dashboard integration), plus a 24-test suite.

---

## B. Ingestion and chunking

**Q6. Why 600-character chunks with 80-character overlap?**

600 characters is roughly a paragraph — enough context to support three
specific questions, small enough that the answer to any one of them is
locatable rather than buried. The 80-character overlap stops a sentence that
straddles a boundary from being lost to both chunks.

It is a tunable, not a law. The honest answer is that it was chosen as a
reasonable default and never swept; a threshold sweep over chunk size would be
a legitimate extension.

**Q7. How do you split — just every 600 characters?**

No. `RecursiveCharacterTextSplitter` tries separators in order: paragraph
breaks, then line breaks, then sentence ends, then spaces. It only cuts
mid-sentence if nothing better is available, so chunks tend to land on natural
boundaries.

**Q8. What is boilerplate filtering and why did you add it?**

The first real run produced questions like *"Who holds the ownership component
in the work described in the WHO disclaimer?"* and *"What do dotted and dashed
lines on maps represent?"* — grammatically perfect and completely worthless.
They came from title pages, contents listings, reference lists and map
legends.

So chunks are classified before generation and dropped if they look like
front matter, a table of contents (dot leaders), a reference list (citation
markers plus author initials), a contributor list, or a linearised numeric
table. On a WHO corpus this removed about 26% of chunks — which also saved
about 26% of the API budget.

**Q9. What went wrong with that filter?**

The first version of the contributor-list rule was too aggressive and dropped
genuine evidence text — a line containing `(RR 0.84; 95% CI 0.71–0.99) (106)`
looked like a citation list to it. I tightened the pattern to require an actual
`Surname AB` name-and-affiliation shape. **That is a good thing to volunteer**:
it shows the filter was validated against real output rather than assumed.

**Q10. What happens with a scanned PDF?**

`pypdf` extracts no text from an image-only PDF. Rather than generating garbage,
the loader checks each page has a minimum amount of real text and reports the
problem. The fix is a different PDF or an OCR step — OCR is deliberately out of
scope.

**Q11. What is "page budgeting"?**

PDF text extraction costs roughly 1.6 seconds per page. A 592-page document
took 15.6 minutes to ingest before any API call was made — longer than the rest
of the pipeline. Since a run only processes `max_chunks` chunks anyway, there
is no point extracting every page. The loader computes how many pages it
plausibly needs and stops. That cut the same ingest to about 40 seconds.

---

## C. Generation

**Q12. Why temperature 0.4?**

A trade-off. At 0.0 the model produces near-identical phrasing for similar
passages, which makes a monotonous dataset. Above about 0.7 it starts
embellishing — adding plausible detail that is not in the passage, which is
exactly the failure the rubric punishes. 0.4 gives variety in phrasing without
loosening its grip on the source.

**Q13. Why three question types?**

Left alone, the model writes "What is X?" over and over. Explicitly asking for
factual, definitional and procedural questions forces variety. The effect is
measurable: this run produced **56 distinct two-word openers** across 103
pairs, with the most common accounting for only **14.6%**.

**Q14. How do you get reliable JSON out of a language model?**

Three layers, because any one of them fails eventually:

1. The prompt demands a JSON array and nothing else.
2. Markdown code fences are stripped — instruction-tuned models add them anyway.
3. If `json.loads` still fails, a regular expression extracts the outermost
   `[...]` span, which handles replies wrapped in a sentence of commentary.

If all three fail the chunk is retried once, then skipped. One bad response
must never kill a twenty-minute run.

**Q15. What are "unanswerable" or abstention examples?**

A dataset made only of answerable questions teaches a model that every question
has an answer in the document — so when it does not know, it invents one. For
10% of chunks the pipeline makes a second call asking for one *plausible,
on-topic* question the passage genuinely **cannot** answer, and pairs it with a
fixed refusal: *"This information is not available in the provided document."*

Two details matter. The answer is a constant in code, not generated, so the
model learns one refusal phrase rather than fifty paraphrases. And the question
must stay on topic — otherwise you teach the model to refuse anything
unfamiliar rather than anything unsupported.

**Q16. How do you verify an abstention example is actually unanswerable?**

A separate judge prompt with two boolean checks: is the passage genuinely
unable to answer it, and is it a question a reader would plausibly ask. Both
must be true. The fixed-answer check is done in code — paying a model to verify
our own string literal would be absurd.

In the recorded run, **5 were written, 5 verified, 0 failed**, and all 5 are in
the dataset.

**Q17. Why does the selection of which chunks get an abstention example use a seed?**

So the dataset is reproducible. A dataset whose contents shift between runs
cannot be reported on, and the sensitivity experiments compare runs against
each other.

---

## D. Validation — the LLM-as-judge

**Q18. What is LLM-as-judge?**

Using one language model to evaluate another's output against a rubric. It
scales where human annotation does not. Its weakness is that it is a model
grading a model — it can be biased, inconsistent, or simply wrong, which is
why this project measures its reliability rather than assuming it.

**Q19. What is your rubric?**

Three criteria, 0–2 each, total 0–6:

- **Groundedness** — is every claim traceable to the passage? (2 = yes;
  1 = mostly, adds a small unstated detail; 0 = contradicts it or invents
  information)
- **Specificity** — is the question precise rather than vague? (0 = vague, or
  refers to "the passage")
- **Completeness** — is the answer self-contained?

**Q20. Why keep pairs at 4 out of 6?**

It keeps pairs that are solid on two criteria and imperfect on a third, which
is where most usable data sits. Dropping to 3 admits pairs weak everywhere;
raising to 5 costs data for little gain. **But see Q33** — the experiment shows
the total threshold does almost nothing on this corpus.

**Q21. Why per-criterion floors as well as a total? This is important.**

Because a sum hides a fatal defect. A pair scoring **groundedness 0,
specificity 2, completeness 2** totals 4/6 and passes a total-only filter — and
it is an answer that invents information. I tested this directly with planted
pairs: an answer asserting a drug dose never stated in the passage scored
exactly 4/6, and so did one that flatly contradicted the source.

So the keep rule has floors: `{groundedness: 2, specificity: 1, completeness: 1}`.
Groundedness sits at the strict 2 of 2, because *"mostly supported, adds a
small unstated detail"* is precisely the failure that matters — the unstated
detail is the invented dose.

**The measured justification is the strongest part of this answer.** In the
recorded run, all 12 rejections failed on groundedness, **11 of them scored
groundedness 1 rather than 0**, and **11 of the 12 would have passed a
total-only filter at ≥ 4/6**. The floors are not theoretical; they are the only
reason anything was rejected at all.

**Q22. Why is the judge a different model from the generator?**

Self-preference bias: a model rates its own phrasing more highly. With one
model doing both, the judge handed out 6/6 to almost everything and the pass
rate meant nothing. The generator is `openai/gpt-oss-120b`; the judge is
`qwen/qwen3.8-27b`. A secondary benefit is that each model has its own
rate-limit bucket, so judging does not eat the generator's quota.

**Q23. Why temperature 0.0 for the judge?**

Judging must be reproducible. The same pair must get the same score today and
tomorrow, or no threshold experiment means anything.

**Q24. Why only 800 characters of passage in the judge prompt?**

The judge only needs enough context to verify the answer. Sending the full
chunk costs tokens without improving the grade — and a longer context makes the
model *more* forgiving on groundedness, because in a wall of text almost
anything looks supported.

**Q25. What is batched judging and why?**

Pairs from the same chunk share a passage, so they are graded in one call
rather than three: the rubric and the passage are sent once instead of three
times. This roughly halved judging cost. Each pair carries an index so a
mis-ordered reply cannot attach a score to the wrong pair, and if a batch comes
back unparseable the pairs are re-judged individually rather than silently left
ungraded.

**Q26. Your pass rate is 89.7%. Is that good?**

It is a *finding*, not a grade. A high pass rate means one of three things: the
generator is good, the judge is lenient, or the corpus is easy. The dashboard
says so explicitly, and the threshold experiment (Q33) shows that on this
corpus the total threshold barely filters at all — what filters is the
groundedness floor.

The honest framing: **89.7% is the pass rate; 12 rejections all on
groundedness, 11 of which a total-only filter would have kept, is the result
that matters.**

---

## E. Deduplication

**Q27. Why not just remove exact duplicate strings?**

Because the duplicates are not exact. *"What is Article 21?"* and *"Explain
Article 21 of the Constitution"* share few tokens — string or fuzzy matching
keeps both — but they are the same question. Embeddings put them at roughly
0.88 cosine similarity, which catches them.

**Q28. How does the embedding comparison work?**

Each pair is encoded with `all-MiniLM-L6-v2` into a 384-dimensional vector,
normalised to unit length. Because the vectors are unit length, their **dot
product is exactly the cosine similarity**, so the whole comparison is one
matrix multiplication rather than a Python loop.

**Q29. Explain cosine similarity.**

The cosine of the angle between two vectors. 1 means identical direction, 0
means unrelated, −1 means opposite. It measures *orientation* rather than
magnitude, which is what you want for text: a long answer and a short one about
the same thing should count as similar.

**Q30. Why 0.90 and not 0.85?**

This is a measured correction, and a good story to tell. At 0.85 over the
question alone, the pipeline merged *"At what age is the first dose given?"*
with *"When is the second dose given?"* at cosine **0.896** — two different
facts collapsed into one. The dedup rate looked good and was partly counting
real information loss as a success.

Three changes fixed it:

1. threshold raised to **0.90**
2. the **answer is embedded alongside the question**, because the
   distinguishing detail usually lives in the answer
3. a **number guard** — two answers stating different numbers are never merged
   however similar their wording, so "25 g for adults" and "15 g for children"
   stay apart

**Q31. What is "first-wins" and why does it matter?**

The algorithm walks the list in order and keeps a pair only if it is below the
threshold against everything kept so far — so the survivor is the earliest
occurrence. That makes the output stable: re-running does not reshuffle which
version of a question survives.

It also had a subtle bug worth mentioning. Abstention pairs were initially
merged into the kept list ahead of answerable ones, which meant a refusal could
win a collision against a good answerable pair — teaching the model to refuse a
question the corpus actually answers. Fixing the ordering so results come out
in generation order fixed it.

**Q32. Your dedup removed only 1 pair. Isn't that a weak result?**

It is an honest one. At 0.85 over questions alone it removed more, but the
sensitivity experiment shows what those extra removals were: at 0.75 it removes
4, and the borderline ones are questions that are arguably distinct. On a
45-chunk corpus there simply are not many true duplicates. On a larger run,
neighbouring passages overlap and real duplicates appear.

---

## F. The experiments

**Q33. What did the threshold sensitivity experiment show?**

That the total threshold is almost irrelevant on this corpus, and the floors do
all the work:

| min score | floors | survivors | avg quality |
|---|---|---|---|
| 3 | yes | 98 | 5.94 |
| 3 | no | 109 | 5.81 |
| 4 (default) | yes | 98 | 5.94 |
| 4 | no | 108 | 5.83 |
| 5 | yes | 95 | 6.00 |
| 6 | yes | 95 | 6.00 |

Moving the threshold from 3 to 6 changes the dataset by three pairs. Comparing
any row with floors against the same row without shows a consistent 10–11 pair
difference at every threshold.

**The finding to state out loud:** the obvious reading of "a ≥ 4/6 quality
filter" is that the 4 is doing the work. On this corpus it is not. A report
that credited the threshold would be wrong.

**Q34. And the dedup sensitivity experiment?**

| cosine | survivors | removed |
|---|---|---|
| 0.75 | 95 | 4 (4.0%) |
| 0.85 | 97 | 2 (2.0%) |
| 0.90 (default) | 98 | 1 (1.0%) |
| 0.95 | 98 | 1 (1.0%) |

The single removal at 0.90 is a genuine duplicate — *"What does the Kisan
Credit Card Scheme refer to?"* versus *"What is the purpose of the Kisan Credit
Card Scheme?"* at cosine 0.951.

**Q35. What is Cohen's kappa and why use it instead of plain agreement?**

Plain agreement is inflated by chance. If two raters both label 90% of items
"good", they will agree most of the time by luck alone. Kappa corrects for
that: it measures agreement *beyond* what chance predicts. 0 means
chance-level, 1 means perfect. The usual Landis & Koch bands: <0.20 slight,
0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, >0.81 almost
perfect.

I also compute a **quadratic-weighted** kappa, because the scale is ordinal —
disagreeing 2-versus-0 is worse than 2-versus-1, and unweighted kappa treats
both as simply "different".

**Q36. What is your kappa?**

**Not measured.** That is the honest answer, and you should give it directly.

A 40-pair sample was labelled, but every label carried the same three scores.
Kappa against a constant rater is mathematically undefined — a rater who never
varies provides no chance-corrected scale. Exact agreement of 0–2.5% is itself
the giveaway: random labelling on a 3-point scale would land near 33%.

The experiment now detects this and refuses to emit a number, rather than
printing a confident-looking `0.000`. Previously it only did that when *both*
raters were constant, which is how a meaningless "κ = 0.000, slight agreement"
reached an earlier draft of the README.

**So the current position is: the judge's scores are not verified by any human,
and the report says so.** The closest thing to external validation is the
non-LLM grounding check, which agrees with the judge on 84.7% of pairs.

**Q37. If the kappa came back low, what would you do?**

Report it and downweight the judge's numbers in the conclusions — that is
explicitly better than hiding it. A low kappa would mean the quality scores are
not trustworthy as an absolute measure, though they might still be useful for
*ranking* pairs. I would then lean harder on the non-LLM grounding check, and
consider a better rubric or a stronger judge model.

---

## G. The independent grounding check

**Q38. What is it and why does it exist?**

Everything else in the quality story comes from a language model. If the judge
is wrong, every number moves together and nothing catches it. So there is one
check with no model in it at all: for each pair it computes the word overlap
between the answer and its source passage, and flags any number in the answer
that does not appear in the passage.

**Q39. What did it measure?**

98 answerable pairs analysed: **84.7% strongly grounded** (overlap ≥ 0.7),
11.2% moderate, 4.1% weak, and **0 answers containing an unsupported number**.
It agrees with the LLM judge on 84.7% of pairs.

**Q40. Why is the disagreement interesting rather than embarrassing?**

Because it is a worklist. Four pairs the judge called fully grounded scored low
on word overlap. Those are either good paraphrases or quiet hallucinations, and
only a human can tell which — so that is exactly where a reviewer should start.
The check is a triage tool, not an oracle.

**Q41. Isn't word overlap a crude measure?**

Yes, and deliberately so. Its value is independence, not sophistication. A good
paraphrase scores low and is fine; that is a known false positive. The number
check complements it: in a planted test, an answer where *every word* matched
the passage still contained a fabricated threshold, and only the number check
caught it. Neither signal is sufficient alone.

**Q42. Did raising the groundedness floor actually improve the data, or just the judge's own numbers?**

A fair challenge, and the answer is measured. When the floor went from 1 to 2,
the **independent** grounding check rose from 81.0% to 86.8% strongly grounded
— measured on the earlier WHO corpus, which is where that before/after
comparison was run. That is confirmation from outside the model
that the stricter rule improved the data, rather than the judge agreeing with
itself.

---

## H. Engineering and infrastructure

**Q43. What was the hardest engineering constraint?**

The Groq free tier's **daily token cap of 200,000**, and the fact that it is
**invisible**. Per-minute limits come back in response headers; the daily token
count does not. It only appears inside the body of a 429 error:

```
Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199706
```

The first full run burned the entire day's allowance and stalled for four
hours, leaving **141 of 263 pairs ungraded**. The dataset was unusable, and
worse, its "100% pass rate" described only the 122 pairs that got graded.

**Q44. How did you fix it?**

`src/budget.py`:

- **Preflight** — estimate tokens and requests before spending anything, and
  print what fraction of the daily budget the run will consume.
- **A meter** — Groq returns exact usage on every response, so after the first
  call the estimate is replaced by ground truth.
- **Stop cleanly** — the run halts before the wall rather than thrashing
  against it, and generation reserves roughly half the budget for judging,
  because pairs that can never be graded are worse than pairs never generated.
- **Honest reporting** — if anything *was* left ungraded, `run_health` records
  it and the dashboard shows a banner, so the pass rate is never quoted as if
  it covered everything.

**Q45. What is the TokenBucket, and why not `time.sleep(1)` between calls?**

A fixed sleep is wrong in both directions: too slow when the budget has
headroom, too fast when it does not. The real ceiling is **8,000 tokens per
minute**, so a shared token bucket meters spending and workers wait their turn
on it.

**Q46. Why concurrency if the limit is tokens per minute?**

Concurrency does not raise the ceiling — it stops request *latency* from
leaving the ceiling unused. Each call takes about 3 seconds to return; with one
worker the token budget sits idle during that wait. Four workers keep it
occupied. Results are reassembled in input order so deduplication's first-wins
rule stays deterministic.

**Q47. What is checkpointing and why did you add it?**

Generation and judging are the two slow, paid stages. Before checkpointing, a
run that hit the daily wall at chunk 90 of 100 — or that you stopped with
Ctrl-C — threw away every call it had already paid for.

Now each finished unit is appended to a JSONL file as it completes, and the
next run skips anything already there. **Measured: a re-run after a completed
run costs 1 API call and 578 tokens, against 104 calls and 62,500 tokens cold.**

**Q48. How do you stop a checkpoint being reused when it shouldn't be?**

A **fingerprint** — a hash of everything that would change the output: model,
temperature, prompt version, token caps, seed, questions per chunk. A record is
reused only when the fingerprint matches exactly. Change the prompt or the
model and every old record is ignored, because blending two generators' output
into one dataset would make every number meaningless.

One deliberate subtlety: the judge checkpoint **excludes** the keep threshold
and the floors. Those are applied to the scores afterwards, so re-running at a
different threshold costs nothing — which is what makes the sensitivity
experiments free.

**Q49. What if the process dies mid-write?**

Records are appended, never rewritten, and each write is flushed and fsynced.
A crash can therefore only damage the final line, and the loader discards a
line it cannot parse. There is a test for exactly this.

**Q50. Tell me about a bug you found and fixed.**

Several are worth having ready:

- **Streamlit module staleness.** Streamlit re-executes the script on every
  interaction but keeps imported modules in memory for the life of the server.
  Editing a file under `src/` therefore had no effect and produced confusing
  errors — an `AttributeError` for a function that plainly exists, then an
  `ImportError` for a name sitting right there in the file. The fix reloads
  modules whose file timestamp has changed; critically, that check must run
  **before** the `from config import ...` statements, because those read
  straight out of the stale module object.
- **A shadowed constant.** `main.py` defined its own `SIMILARITY_THRESHOLD = 0.85`,
  which silently overrode the module default of 0.90 — so a run used 0.85 while
  every comment and document said 0.90. The same duplicate default existed in
  the runner and the dashboard slider. All three now import from one place.
- **The kappa degeneracy.** Described in Q36.
- **Abstention ordering in dedup.** Described in Q31.

**Q51. How is the dashboard decoupled from the pipeline?**

The pipeline runs as a **separate process**. The two communicate through a
progress file on disk containing stage, counts, a heartbeat and status. So
closing the browser does not kill a run, the page can reconnect to a run in
flight, and a crashed pipeline is detectable by a stale heartbeat rather than
hanging the UI.

**Q52. What do your tests cover?**

24 offline tests, no API calls, running in about 80 seconds (2 seconds if you
skip the two that load the embedder). They cover the parts where a silent
change is most expensive:

- the keep rule, including every floor boundary case
- deduplication: the number guard, that a true reword is still removed, and
  that the lexical overlap is recorded
- generation: short answers dropped, context-dependent questions dropped,
  the seeded abstention selection
- JSON extraction tolerating fences and surrounding prose
- export shape: internals stripped, `answerable` carried through
- checkpoint round-trip, fingerprint mismatch, and the truncated final line

**Q53. Why not test the API calls?**

They cost money, they are non-deterministic, and they test Groq rather than my
code. The API boundary is stubbed where a test needs it. What is worth testing
is the decision logic — a wrong keep rule quietly admits fabricated answers,
and a wrong dedup rule quietly deletes distinct facts. Both look like a working
pipeline from the outside.

---

## I. Fundamentals an examiner may probe

**Q54. What is an embedding?**

A dense vector representation of text where semantically similar text ends up
close together in the vector space. `all-MiniLM-L6-v2` produces 384 dimensions.
Unlike one-hot or bag-of-words representations, it captures meaning rather than
surface form, which is exactly why it catches rewordings that string matching
misses.

**Q55. Why that embedding model specifically?**

It is small (~90 MB), fast on CPU, and strong for its size on sentence
similarity. This project has **no GPU** — an AMD integrated chip — so
everything local must be CPU-viable. A larger model would be marginally better
and considerably slower.

**Q56. Why is the LLM remote rather than local?**

No GPU. Running a 120-billion-parameter model locally is not possible on this
hardware; even a 7B model would be painfully slow on CPU. Groq serves them over
an API at high speed. The trade-off is the rate limits, which is why
`budget.py` exists.

**Q57. What is ChatML and why export it?**

A conversational message format — a list of `{role, content}` objects with
roles `system`, `user` and `assistant`. It is what supervised fine-tuning
frameworks such as `SFTTrainer` expect. Exporting it means the dataset is
directly trainable without a conversion step.

The system message carries the safety disclaimer deliberately: if this data is
ever used to fine-tune a model, the safety framing should be part of the
training signal rather than a note in a README nobody reads.

**Q58. What is supervised fine-tuning?**

Continuing training of a pre-trained model on labelled input-output pairs so it
learns a task or domain. This project produces the *data* for that step. The
fine-tuning itself is explicitly future work.

**Q59. What is a token?**

The unit a language model actually reads — roughly a word piece; about 4
characters of English on average. It matters here because every limit in the
project is denominated in tokens: 200,000 per day, 8,000 per minute.

**Q60. What does `reasoning_effort: low` do?**

gpt-oss models produce hidden reasoning tokens before answering. Those count
against the same per-minute budget. For an extraction task like this, low
effort keeps the output quality while cutting the hidden spend. **Measured
effect on the earlier WHO corpus, before and after the change: generation
tokens fell 32% (31,399 → 21,278) while producing *more* pairs (63 → 76).**
(That comparison is from a different corpus to the headline numbers above, so
do not try to reconcile the two.) It is sent only to gpt-oss — it is a 400 error on qwen,
which is why the code checks the model family.

**Q61. Why cap `max_tokens`?**

Groq admits a request only when `prompt + max_tokens` fits inside what remains
of the per-minute budget. An oversized cap does not just risk a long reply — it
makes every call wait longer for admission. The caps (1024 generate, 512 judge)
are set from measured usage.

**Q62. Why not use RAG instead?**

Different problem. RAG answers questions at inference time by retrieving
passages; it does not produce training data, and it needs the documents present
at query time. This pipeline produces a dataset you can train a model on, after
which the model answers without the documents. They are complementary — you
might well use this data to fine-tune a model that is then used inside a RAG
system.

**Q63. Why not just fine-tune on the raw document text?**

Continued pre-training on raw text teaches style and vocabulary, not
instruction-following. If you want a model that answers questions, you need
question-answer pairs. That is the gap this fills.

---

## J. Safety and ethics

**Q64. What are the risks of this project?**

Three, and they compound:

1. **Hallucination.** A model writes the answers; it can invent things. In a
   clinical corpus a fabricated dose is the worst possible output. Mitigation:
   the groundedness floor, and the independent number check.
2. **Misuse.** Someone could treat the dataset as authoritative. Mitigation:
   the disclaimer travels with the data — in the README, the dashboard, the
   ChatML system prompt and the HuggingFace card, all from one constant so they
   cannot drift apart.
3. **Amplification.** Errors in the source documents propagate into the
   dataset and then into any model trained on it. The pipeline is faithful to
   its source; it does not fact-check the source.

**Q65. How is the disclaimer handled?**

Generated from a single constant, `SAFETY_DISCLAIMER` in `src/export.py`, and
it **adapts to the corpus**: `looks_clinical()` detects clinical sources and
strengthens the wording to "not medical advice, not clinical guidance". The
recorded run is agricultural, so the generic wording applies — and I verified
the ChatML system prompt says "not professional advice" rather than wrongly
claiming a medical context.

**Q66. What about data privacy?**

Only public documents. No patient data, no personal data, ever. The API key is
read from `.env`, which is gitignored and was gitignored *before* the key was
ever put there. The dashboard passes a user-supplied key through the child
process environment only — never to disk, never to a log.

**Q67. Could the generated data be biased?**

Yes, in two ways. The source documents carry their own bias and the pipeline
faithfully reproduces it. And the generator may systematically favour certain
phrasings or topics within a passage. The diversity statistics partly surface
the second; the first is not addressed and should be acknowledged as a
limitation.

---

## K. Limitations — be ready to volunteer these

**Q68. What are the weaknesses of this project?**

Lead with these rather than being caught by them:

1. **The judge is unverified by a human.** The kappa is not measured (Q36).
2. **The judge is lenient on this corpus.** 95 of 98 scored pairs got full
   marks. The threshold is doing almost nothing (Q33).
3. **Deduplication removed one pair.** Honest, but the stage demonstrates
   little at this corpus size (Q32).
4. **Chunk size and overlap were never swept.** 600/80 is a reasonable default,
   not a measured optimum.
5. **No OCR**, so scanned PDFs are rejected rather than handled.
6. **Single corpus per run.** Cross-document duplicates are only caught within
   one run.
7. **The grounding check is lexical**, so good paraphrases register as false
   positives.
8. **No model has actually been fine-tuned on the data yet**, so the ultimate
   claim — that cleaning improves a trained model — is untested.

**Q69. What would you do with more time or budget?**

In priority order: a proper blind kappa on 40 varied labels; fine-tune a small
model on cleaned versus raw data and compare, which turns the whole project
from "the data looks better" into "the data *is* better"; sweep chunk size; try
a stronger judge or an ensemble of judges; and add a cross-run deduplication
index.

**Q70. If you started again, what would you do differently?**

Put every tunable in one configuration file from the start. Several constants
ended up defined in two places, and one of them — the dedup threshold — silently
overrode the other for a whole run. I would also write the tests earlier; they
were added late, and two of the bugs they now guard against were found by
accident rather than by a test.

---

## L. Likely rapid-fire questions

| Question | Answer |
|---|---|
| Generator model? | `openai/gpt-oss-120b` via Groq |
| Judge model? | `qwen/qwen3.8-27b` — deliberately different |
| Embedding model? | `all-MiniLM-L6-v2`, 384 dimensions, CPU |
| Chunk size / overlap? | 600 characters / 80 |
| Generation temperature? | 0.4 |
| Judge temperature? | 0.0 |
| Keep rule? | total ≥ 4/6 **and** groundedness 2, specificity 1, completeness 1 |
| Dedup threshold? | cosine ≥ 0.90, over question + answer, with a number guard |
| Abstention ratio? | 10% of chunks |
| Daily token budget? | 200,000 — the binding constraint |
| Per-minute tokens? | 8,000 |
| Final dataset size? | 103 pairs from 45 chunks |
| Average quality? | 5.94 / 6 |
| Pass rate? | 89.7% |
| Strongly grounded? | 84.7% |
| Unsupported numbers? | 0 |
| Cohen's kappa? | Not measured — labels had no variance |
| Run cost? | 104 calls, 62,500 tokens, 10 min 9 s |
| Tests? | 24, offline, no API calls |
| Output formats? | JSONL, ChatML JSONL, CSV, plus stats and audit files |

---

## M. If you are asked to demonstrate it live

1. Open the dashboard, show the **sidebar** — this is what produced the
   numbers.
2. **Overview** — the funnel. 116 raw → 104 passed → 103 final.
3. **Explore** — drag the quality slider and let them watch the count fall.
   Then toggle **Status** off and on to show the abstention pairs are governed
   separately, and explain why.
4. **Evidence** — this is the tab to spend time on. Show a rejected pair, read
   the judge's reason out loud, point out that it scored 5/6 and a total-only
   filter would have kept it. Then the duplicates, with cosine beside lexical
   overlap.
5. **Grounding** — "this is the only number here that no language model
   produced."
6. **Judge Reliability** — say plainly that the kappa is not yet measured and
   why, and that the tool refuses to print a misleading number.
7. **Download** — preview the JSONL, show the `answerable` field.

If you have time, run `python -m pytest -m "not slow"` — 22 tests in 2 seconds
is a good closing note.
