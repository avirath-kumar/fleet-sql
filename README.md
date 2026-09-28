# fleet-sql

A text-to-SQL deep agent over fleet, configuration and deferral data, and a
LangSmith loop that finds a real architectural problem in it and shows the fix
working.

The problem is the one a prototype hits the moment it leaves demo data: some
pre-approved queries return about 500 rows, which is too much to put in a
model's context. **The interesting part is what happens instead of a crash.**

## What actually goes wrong

Ask the naive build *"which station carries the most open deferrals on the
737-800 fleet?"* and it answers confidently:

> ORD carries the most open deferrals, with **118**.

The database says **105**. Ask again and it says 103. It is not broken in any
way you would notice: the number is plausible, the station is right, and the
per-station figures even sum to the correct total.

What happened is visible in the trace. The query returned 511 rows and ~140 KB.
The harness spilled that to `/large_tool_results/call_…` rather than inline it,
the agent then delegated with an untyped `task("read the file and count…")`
string, and the subagent counted by reading. It approximated.

So the failure is not "context overflow". It is **a model doing arithmetic on
data it had to read**, which is what putting rows in a conversation asks it to
do.

## The fix

Two changes, both in `src/agent/`.

**1. Rows never leave the database as rows.** `run_query` returns a *receipt* —
exact row count, columns, a five-row preview, and counted breakdowns of the
low-cardinality columns. The rows go to a file.

```
511-row result, inline : 139,515 bytes  (~34,900 tokens)
511-row result, receipt:   2,079 bytes  (~   520 tokens)     67x
```

The receipt keeps the *number* exact, which is the thing usually being asked.
Truncating to "the first 20 of something" loses it.

Then targeted tools work on the stored result, and each returns something
bounded: `describe_result`, `filter_result` (returns a new receipt, chainable),
`aggregate_result` (same cost at 20 rows or 5,000), `page_result` (hard cap of
50, whatever `limit` says), `export_report` (writes a CSV and cites the path).

The counts are computed in Python, so the model never does the arithmetic.

**2. Delegation has a schema.** `analyze_result(result_id, question, group_by)`
replaces `task("go look at this")`. The analyst subagent runs with the result
tools only — it cannot start new queries — and returns a typed `AnalysisFinding`
carrying `evidence` and `rows_inspected`. That is what lets an evaluator check
that a number was counted rather than guessed.

## What the evals show

Both builds, same dataset, same harness — only the tools differ.

Three arms, same dataset, same harness — only the tools and the receipt differ.

```
question                                  v1       v2-regressed   v2
---------------------------------------------------------------------
distinct 737-800 aircraft w/ open cat C   0/3      3/3            3/3
which station carries the most deferrals  0/3      3/3            3/3
(six other questions)                     3/3      3/3            3/3
---------------------------------------------------------------------
answer_is_correct                         18/24    24/24          24/24
no_silent_truncation                      0.67     1.00           1.00
context_efficiency                        0.84     0.97           0.99
rows_stayed_out_of_context                0.89     1.00           1.00
```

**`answer_is_correct` is code, not a judge.** The failure is a plausible wrong
number, and a judge reading "ORD carries the most, with 118" scores it well.
Every example carries a `truth_sql`; the evaluator runs it and compares.

**`rows_stayed_out_of_context` is close to 1.00 on every arm** — and that is
the honest finding. The naive build rarely blows the context budget, because
the harness evicts the oversized result first. It stays cheap and becomes
wrong. Measuring only context size would have shown almost nothing.

### What is reliable, and what is not

Measured across seven runs of this unchanged suite:

| metric | v1 | v2 |
|---|---|---|
| `no_silent_truncation` | 0.67 – 0.74 | **1.00 every time** |
| `rows_stayed_out_of_context` | 0.85 – 0.96 | **1.00 every time** |
| `context_efficiency` | 0.81 – 0.95 | 0.96 – 0.99 |
| `answer_is_correct` | 18–22 / 24 | 23–24 / 24 |

**The structural metrics separate perfectly and never move.** v1 puts 140 KB in
the context and needs the harness to rescue it; v2 never does. That is a fact
about the architecture, and it holds on every run.

**The correctness metric is noisy.** v1 has scored anywhere from 18 to 22 out
of 24, and the station question — which failed 0/3 in one run — passed 3/3 in
another. Wrong numbers are a *consequence* of putting rows in the context, and
consequences are probabilistic. Do not promise "v1 gets this wrong" on stage;
it will get it right roughly a third of the time.

Lead with the structural gap, which always holds, and use a correctness failure
as the illustration of what it costs when it bites.

The regressed arm is subtler still and often does not reproduce at all: it has
scored 2/3, 0/1, 2/3, 3/3 and 3/3 on the distinct-count question. Removing
`distinct` from the receipt does not remove the capability, because
`aggregate_result` with `count_distinct` still reaches the answer. It supports
"this cleanup made the agent less reliable", not "broke it".

When the catalogue has a pre-aggregated query, every build is fine. The
architecture earns its keep exactly where the catalogue runs out.

## The loop

```bash
scripts/setup_langsmith.py                      # project, dataset, online evaluators, queue
scripts/generate_traces.py --variant v1 --tag before-fix   # traffic, no dataset yet
scripts/setup_langsmith.py --promote 5          # failing traces -> dataset examples
scripts/experiment.sh                           # score v1 and v2 on it
scripts/report.py                               # the table above
```

Online evaluators score live traffic; offline ones score the dataset. They are
deliberately different: an online evaluator runs sandboxed with no network and
no filesystem, so the check that settles a number against SQLite cannot run
there. Live traffic gets the structural checks, which are free; the dataset gets
the one that needs the database.

`--promote` is the step the loop turns on. A trace that scored badly is a
question the suite did not have. Promoted examples are tagged `source:
promoted`, so re-running `dataset.py` replaces the curated ones and leaves them.

## Layout

```
data/        schema.sql, generate.py -> fleet.db   (deterministic, regenerate freely)
src/agent/   catalog.py   the pre-approved queries
             results.py   the receipt and the result store
             tools/       run_query + the targeted result tools, both variants
             subagents/   the typed analyst
             agent.py     v1 and v2, same harness
evals/       dataset.py, evaluators.py, run_experiment.py
scripts/     setup_langsmith.py, generate_traces.py, experiment.sh, report.py, compare.py
tests/       11 hermetic checks, no model calls
```

## Setup

```bash
cp .env.example .env          # fill in the two LangSmith keys
uv sync
python data/generate.py       # builds fleet.db
.venv/bin/python -m pytest -q
scripts/setup_langsmith.py
```

Two credentials, not interchangeable: a workspace key (`lsv2_pt_…`) for tracing
and datasets, and a gateway key (`lsv2_sk_…`) that pays for model calls.

`AGENT_VARIANT=v1|v2` picks the build; `scripts/compare.py "question"` runs one
question through both and prints what each cost.

## The data

`data/generate.py` is deterministic and asserts the two numbers the demo turns
on: the 747-8 has exactly **43** top-level configuration slots (the figure the
original prototype returned), and the 737-800 fleet carries **511** open
deferrals (the ~500-row result). 349 aircraft, 8 models, 796 configuration
slots, 1,764 deferrals.
