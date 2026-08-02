# Naija-Petro evaluation harness

A benchmark and scoring harness for the Naija-Petro models and the deployed RAG app. It answers
one question: **does the fine-tuning, and does the retrieval layer, actually make the answers
better, and by how much on which axis?**

Everything here is offline until you point it at a target. Nothing runs automatically.

## Contents

| File | Purpose |
|---|---|
| `questions.jsonl` | 73 benchmark questions with a key-facts checklist for each |
| `run_eval.py` | The runner: generation, judging, deterministic grading, sanity checks, reports |
| `README.md` | This file |

## Why this exists

The project's previous evaluation was degenerate. An LLM judge scored the base model **3.00 / 5.0**
and the fine-tuned model **3.00 / 5.0** across 30 questions: a delta of exactly zero. That is not a
result, it is a broken instrument, and it is why the Hugging Face model cards carry no scores.

A judge collapses to a constant for predictable reasons: a single blended "overall quality" score
with no anchors, no ground truth to check against, no penalty for hedging, and a scoring loop that
quietly writes a default value whenever parsing fails. This harness is built against each of those.

| Failure mode | What this harness does about it |
|---|---|
| One blended score, nothing to disagree about | Six independent dimensions, each scored separately |
| "Rate this 1 to 5" with no definition of 3 | Every dimension has explicit anchors for 1, 3 and 5 |
| Judge has no ground truth, so it rates vibes | Each question ships a `key_facts` checklist, passed to the judge as the standard |
| Judge asserts a verdict with no evidence | The judge must quote a verbatim span for `factual_accuracy`; the quote is checked against the answer in code |
| Only the judge measures coverage | Key-fact coverage is also computed lexically in code; both numbers are reported side by side |
| Judge does arithmetic badly and inconsistently | Numeric questions are graded in code against `numeric_answer` with a relative tolerance |
| A judge that cannot tell good from filler | `--control-probe` judges a canned filler answer alongside real ones and checks the gap |
| Parse failure silently becomes a 3 | A malformed judgement is recorded as a failure and excluded, never defaulted |
| Ties everywhere in a head to head | Pairwise mode reports the tie rate separately and randomises answer order to expose position bias |
| Nobody notices the numbers never move | The run scores itself: per-dimension variance, distinct values, and a loud verdict |

**Scores from this harness are only meaningful when the sanity check passes.** A run that fails the
sanity check prints `SANITY CHECK: FAIL. DO NOT PUBLISH THESE SCORES.` and exits with status 2.
Treat that as a hard gate, not a suggestion.

## The benchmark set

73 questions, one JSON object per line in `questions.jsonl`.

| Category | Count |
|---|---:|
| `nigerian_regulation` | 10 |
| `fiscal_terms` | 8 |
| `reservoir_engineering` | 8 |
| `nigeria_field_knowledge` | 8 |
| `completions` | 7 |
| `drilling` | 7 |
| `petroleum_geoscience` | 7 |
| `eor` | 6 |
| `production_operations` | 6 |
| `well_testing` | 6 |

Difficulty: 9 easy, 40 medium, 24 hard. 35 questions are marked `requires_retrieval`, 38 are not.
10 are numeric, 4 are traps.

### Fields

| Field | Meaning |
|---|---|
| `id` | Stable identifier, for example `NR-04`. Used as the resume key, so do not renumber |
| `category` | One of the ten categories above |
| `difficulty` | `easy`, `medium` or `hard` |
| `question` | The user turn, sent verbatim to every target |
| `key_facts` | Specific checkable claims a correct answer should contain. This is the ground truth given to the judge and to the lexical grader |
| `requires_retrieval` | `true` when the answer depends on Nigeria-specific facts a base model would not reliably hold, `false` for universal petroleum engineering |
| `trap` | Present and `true` on questions with a deliberately false premise |
| `numeric_answer`, `unit`, `rel_tolerance` | Present on calculation questions. Graded in code, never by the judge |

### The `requires_retrieval` split

This is the most useful column in the report. Universal petroleum engineering is what fine-tuning
should improve; Nigeria-specific fact recall is what retrieval should improve. A fine-tuned model
without RAG that gains on `requires_retrieval: false` and not on `true` is behaving exactly as
expected. A RAG system that does not beat the bare model on `requires_retrieval: true` is not
earning its latency.

### Trap questions

Four questions contain a false or confused premise, and their `key_facts` say that a correct
response declines or corrects it:

- `NR-06` asks about the "Petroleum Industry Act 2023". There is no such Act; the statute is the
  PIA 2021.
- `NR-07` asks which schedule empowers NMDPRA to approve upstream field development plans. NMDPRA
  is the midstream and downstream regulator; upstream FDPs go to NUPRC.
- `RE-06` asks how the Vogel correlation estimates bubble point pressure. Vogel is an IPR, not a
  PVT correlation.
- `PG-03` asks at what depth the Benin Formation becomes the main oil reservoir in the Niger Delta.
  The Benin Formation is a continental freshwater aquifer; the reservoirs are in the Agbada.

These are the hallucination measurement. A model that invents a schedule number to satisfy `NR-07`
is doing the single most damaging thing an assistant can do in a regulated industry.

## The judging design

### Six dimensions, each with anchors

| Dimension | What it measures |
|---|---|
| `factual_accuracy` | Are the claims actually made correct? The judge must quote the decisive span |
| `key_fact_coverage` | How much of the `key_facts` checklist is present |
| `nigerian_specificity` | Concrete, load-bearing Nigerian detail against generic or decorative mentions |
| `engineering_rigour` | Assumptions stated, governing equations given, units carried, limits flagged |
| `citation_quality` | Named, appropriate, checkable sources; official over news; a fabricated citation scores 1 |
| `hallucination` | **Inverted: higher is better.** Invented statutes, sections, projects, figures, or playing along with a false premise |

The full anchor text for 1, 3 and 5 lives in `DIMENSIONS` at the top of `run_eval.py`, which is the
single source of truth: the same text is sent to the judge and printed by `--dry-run`. The judge
must return strict JSON with a short justification per dimension, the indices of the checklist items
it found, and any specifics it believes were fabricated.

### Checks on the judge itself

- **Quote verification.** The `factual_accuracy` quote is matched against the answer after
  whitespace normalisation. The report gives the verification rate; below 50 percent the judge is
  inventing its evidence and its justifications cannot be trusted.
- **Two coverage measures.** The judge's coverage score sits next to a lexical coverage figure
  computed in code from the checklist tokens. The lexical measure is a crude proxy and the absolute
  values will differ; that is fine. What is not fine is the lexical measure moving across questions
  while the judge's does not, which fires a warning.
- **No default scores.** If the judge returns unparseable JSON after its retries, the question is
  recorded as a judge failure and excluded from the aggregate. Failures are counted and reported,
  and above 10 percent they fire a warning, because a silent default of 3 is precisely how the
  previous evaluation produced its flat result.
- **Control probe.** `--control-probe N` judges a canned, fluent, content-free answer on the first N
  questions. Real answers must beat it by at least 0.75 on the overall mean. If they do not, the
  judge cannot distinguish substance from filler and the run is void.

### Deterministic grading

Numeric questions are never graded by the judge. `grade_numeric` extracts every number from the
answer, including scientific notation such as `2.23 x 10^8`, and compares against `numeric_answer`
within `rel_tolerance`. It accepts the target expressed at a shifted magnitude (thousands, millions,
billions) or as a fraction rather than a percentage, since `223 MMSTB` and `0.218` are correct
answers, and records which scale matched so a reviewer can see what was accepted.

### Pairwise mode

`pairwise` puts two systems on the same question and asks the judge to pick a winner. The two
answers are presented in an order randomised per question from `--seed`, and the mapping is written
to every record, so position bias is measurable rather than assumed away. The report gives win,
loss and tie rates with the **tie rate broken out**, plus the rate at which position 1 won. An
all-ties result is the degenerate signature, and a position-1 win rate outside 35 to 65 percent means
the judge is scoring layout rather than content.

## Running it

Requires `httpx` and `python-dotenv`, both already in `requirements.txt`. Judge credentials are read
from `.env` by variable name: `NVIDIA_API_KEY` and, if set, `NVIDIA_BASE_URL`. The default judge is
`nvidia/llama-3.3-nemotron-super-49b-v1` on the NVIDIA NIM endpoint. Override with `--judge-model`,
`--judge-url` and `--judge-key-env`.

### Always start with a dry run

```bash
python eval/run_eval.py validate

python eval/run_eval.py single \
  --kind openai --url http://localhost:11434/v1 --model qwen3:8b --name base-8b \
  --out eval/results/base.json --dry-run --limit 2
```

`--dry-run` prints the exact system prompt, the exact user turn, and the exact judge prompt, then
exits. It makes no network call at all.

### Comparison 1: base model against fine-tuned model

Score each separately, then diff them offline. Use the **same** system prompt and sampling for both,
or you are measuring prompting rather than the model.

```bash
# base Qwen3-8B, served locally by Ollama or vLLM
python eval/run_eval.py single \
  --kind openai --url http://localhost:11434/v1 --model qwen3:8b \
  --name base-8b --out eval/results/base-8b.json \
  --control-probe 5 --concurrency 2

# the fine-tune, same endpoint shape
python eval/run_eval.py single \
  --kind openai --url http://localhost:11434/v1 --model naija-petro-8b \
  --name naija-petro-8b --out eval/results/ft-8b.json \
  --control-probe 5 --concurrency 2

# per-dimension paired delta, offline, no network call
python eval/run_eval.py compare \
  --run-a eval/results/base-8b.json \
  --run-b eval/results/ft-8b.json \
  --out eval/results/delta-base-vs-ft.json
```

`compare` is where the old failure would have surfaced immediately: if every dimension moves by less
than 0.05, it says so in capital letters and refuses to pass.

### Comparison 2: fine-tuned model against fine-tuned model plus RAG

The deployed app is the fine-tuned model plus retrieval, calculators and the citation prompt, so
scoring the app against the bare model measures the whole retrieval layer.

```bash
python eval/run_eval.py single \
  --kind app --url https://naija-petro.shinzii.tech --name app-rag \
  --token-env ACCESS_TOKEN --out eval/results/app-rag.json --concurrency 2

python eval/run_eval.py compare \
  --run-a eval/results/ft-8b.json --run-b eval/results/app-rag.json \
  --out eval/results/delta-ft-vs-rag.json
```

Expect the gain to concentrate in `nigerian_specificity`, `citation_quality` and the
`requires_retrieval: true` rows. If it does not, the retrieval layer is not doing what it claims.

### Comparison 3: head to head

```bash
python eval/run_eval.py pairwise \
  --a-kind openai --a-url http://localhost:11434/v1 --a-model qwen3:8b --a-name base-8b \
  --b-kind app --b-url https://naija-petro.shinzii.tech --b-name app-rag \
  --out eval/results/base-vs-app.json --seed 1337
```

Rerun with a different `--seed` and confirm the winner does not change. If it does, the result is
position bias, not quality.

### Scoring the deployed app

`--kind app` POSTs to `<url>/chat` and reassembles the Server-Sent Events stream, keeping the
`meta` source list and any `tool` results alongside the answer, so the report shows how many sources
each answer actually carried. The app applies its own system prompt and retrieval, so
`--system-prompt-file` and `--model` are ignored for app targets. Set `--token-env` to the name of
an environment variable holding an access token if the daily free limit would otherwise stop the
run at 10 questions. Keep `--concurrency` at 1 or 2: the app is capped to a single GPU container and
the first request cold-starts for a minute or two.

## Flags

### Common to `single` and `pairwise`

| Flag | Effect |
|---|---|
| `--questions PATH` | Question set, defaults to `eval/questions.jsonl` |
| `--limit N` | First N questions only. Use for a smoke test |
| `--category NAME` | Repeatable filter, for example `--category nigerian_regulation` |
| `--difficulty NAME` | Repeatable filter |
| `--id ID` | Repeatable single-question filter |
| `--concurrency N` | Questions in flight at once, default 4. Lower it against one GPU |
| `--answer-retries N` | Retries with exponential backoff and jitter on transport errors, 429 and 5xx |
| `--answer-timeout S` | Per request, default 300 s to survive a cold start |
| `--out PATH.json` | Results JSON. A `.jsonl` and a `.md` are written alongside it |
| `--resume` | Skip questions already in the sibling `.jsonl`. Records are appended as they finish, so a crashed run loses at most the questions in flight |
| `--dry-run` | Print the exact prompts and exit, no network call |
| `--system-prompt-file PATH` | Override the system prompt for `openai` targets |
| `--no-system-prompt` | Send no system prompt, for a bare base-model comparison |
| `--judge-url`, `--judge-model`, `--judge-key-env` | Judge endpoint, model, and the **name** of the env var holding its key |
| `--judge-temperature` | Default 0.0. Raise only if you want to measure judge variance itself |
| `--judge-retries`, `--judge-timeout` | Judge retry budget and per call timeout |
| `--judge-thinking on/off` | Nemotron models take a `detailed thinking on/off` system line. Default `off` |

### Target flags

`single` takes them bare (`--kind`, `--url`, `--model`, `--name`, `--key-env`, `--token-env`,
`--reasoning`, `--units`, `--temperature`, `--max-tokens`). `pairwise` takes the same set twice with
`--a-` and `--b-` prefixes.

### `single` only

| Flag | Effect |
|---|---|
| `--control-probe N` | Also judge a canned filler answer on the first N questions and check the gap. Recommended: 5 |
| `--fact-threshold F` | Lexical overlap fraction at which a key fact counts as covered, default 0.6 |

### `pairwise` only

`--seed N` controls per-question answer order. Change it and rerun to test position bias.

## Cost and runtime

Per full 73-question `single` run:

- **Judging** is roughly 200,000 input tokens and 30,000 output tokens, since each judge prompt
  carries the rubric, the checklist and the answer. At typical hosted prices for a mid-sized model
  that is cents rather than dollars, and on the NVIDIA build tier it is credits. `--control-probe 5`
  adds five more judge calls.
- **Generation** dominates. 73 answers of up to 1,400 tokens each is roughly 30 to 60 minutes of GPU
  time at `--concurrency 2`, plus one cold start. Against the deployed app on a capped L4 that is
  the whole bill. Locally with Ollama it is free but slower.
- A `pairwise` run generates from both systems, so roughly double the generation and about half the
  judge tokens of two `single` runs.

Use `--limit 10 --control-probe 3` for a shakedown before committing to a full run, and `--resume`
so nothing is paid for twice.

## Reading the output

Three files per run, from `--out eval/results/app-rag.json`:

- `app-rag.jsonl` is the append-only record stream, one line per question, written as work
  completes. This is what `--resume` reads.
- `app-rag.json` is the full result: metadata, every per-question per-dimension score with the
  judge's justifications and quote, the aggregate, and the sanity verdict.
- `app-rag.md` is the readable summary.

Read the markdown in this order:

1. **The sanity verdict, first, every time.** If it says FAIL, stop. Every number below it is
   suspect and none of it goes on a model card.
2. **The per-dimension table.** Look at the `SD`, `Distinct` and `Histogram` columns before the
   `Mean`. A dimension with SD below 0.35, or fewer than 3 distinct values across the run, is not
   measuring anything, whatever its mean says.
3. **The retrieval-dependence split.** This is where a claim about the RAG layer lives or dies.
4. **The numeric table.** Deterministic, so it is the one part of the report that cannot be
   flattered by a lenient judge. A model that reasons well but computes badly shows up here and
   nowhere else.
5. **The trap table.** Low `hallucination` scores here mean the model invents statutes under
   pressure, which matters more than any average.
6. **The ten weakest answers.** Read the actual text in the JSON before drawing conclusions. The
   judge is an instrument, not an authority.

The `compare` report adds a per-dimension paired delta with a `t-like` column, which is the mean
delta over the standard error of the paired differences. Above roughly 2 in absolute value the
difference is unlikely to be sampling noise at this sample size. It is a rough guide, not a
significance test, and 73 questions is a small sample: report the delta with its spread, never as a
bare headline number.

## Caveats

- An LLM judge is a noisy instrument even when it is working. The sanity checks catch a dead
  instrument, not a biased one. Spot check the judge's justifications against the answers.
- `key_facts` is hand written and is the ground truth for scoring. If a key fact is wrong, every
  system is penalised for being right. Domain review of the checklist matters more than any tuning
  of the runner.
- The lexical coverage measure rewards vocabulary overlap and can be fooled by an answer that uses
  the right words wrongly. It exists as a cross-check on the judge, not as a score.
- 73 questions is enough to see a real effect and not enough to resolve a small one.
- Results files contain full model answers. Consider keeping `eval/results/` out of version control.
