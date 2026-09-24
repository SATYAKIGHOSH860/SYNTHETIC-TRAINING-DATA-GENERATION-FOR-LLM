### 1. Quality threshold sensitivity

| Rule | Min score | Kept | % of judged | After dedup | Avg quality | Kept with groundedness < 2 |
|---|---|---|---|---|---|---|
| total only (spec) | ≥ 3 | 232 | 100% | 226 | 5.922 | 10 |
| total only (spec) | ≥ 4 | 230 | 99% | 224 | 5.948 | 8 |
| total only (spec) | ≥ 5 | 227 | 98% | 221 | 5.974 | 5 |
| total only (spec) | ≥ 6 | 221 | 95% | 215 | 6.0 | 0 |
| total + floors (used) | ≥ 3 | 222 | 96% | 216 | 5.995 | 0 |
| total + floors (used) | ≥ 4 | 222 | 96% | 216 | 5.995 | 0 |
| total + floors (used) | ≥ 5 | 222 | 96% | 216 | 5.995 | 0 |
| total + floors (used) | ≥ 6 | 221 | 95% | 215 | 6.0 | 0 |

### 2. Deduplication sensitivity

| Setting | Cosine threshold | Input | Removed | Survivors | Removed % |
|---|---|---|---|---|---|
| question only (spec) | 0.75 | 232 | 73 | 159 | 31.5% |
| question only (spec) | 0.8 | 232 | 42 | 190 | 18.1% |
| question only (spec) | 0.85 | 232 | 27 | 205 | 11.6% |
| question only (spec) | 0.9 | 232 | 18 | 214 | 7.8% |
| question only (spec) | 0.95 | 232 | 6 | 226 | 2.6% |
| question + answer, number guard (used) | 0.75 | 232 | 48 | 184 | 20.7% |
| question + answer, number guard (used) | 0.8 | 232 | 27 | 205 | 11.6% |
| question + answer, number guard (used) | 0.85 | 232 | 11 | 221 | 4.7% |
| question + answer, number guard (used) | 0.9 | 232 | 6 | 226 | 2.6% |
| question + answer, number guard (used) | 0.95 | 232 | 2 | 230 | 0.9% |

### 3. Judge reliability (Cohen's kappa, judge vs human)

_Not yet computed: label data/output/human_labels.json (dashboard tab 4)._

### 4. Question-type and opening-word diversity

| | Before dedup | After dedup |
|---|---|---|
| definitional | 33 | 31 |
| factual | 169 | 166 |
| procedural | 20 | 19 |
| unanswerable | 10 | 10 |
| distinct opening words | 19 | 19 |
| most common opening | “what” 46% | “what” 46% |