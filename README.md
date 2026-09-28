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
distinct 737-800 aircraft w/ open cat C   1/3      2/3            3/3
which station carries the most deferrals  1/3      3/3            3/3
(six other questions)                     3/3      3/3            3/3
---------------------------------------------------------------------
answer_is_correct                         20/24    23/24          24/24
no_silent_truncation                      0.70     1.00           1.00
context_efficiency                        0.88     0.97           0.99
rows_stayed_out_of_context                0.93     1.00           1.00
```

**`v2-regressed` is a real regression, not a contrived one.** Cleaning up the
receipt collapsed two column summaries into one and dropped high-cardinality
columns entirely — and with them the cardinality. "How many distinct aircraft
are affected" is exactly that number. The station question got *better* in the
same change; the distinct question degraded to 2/3, intermittently. Code review
did not catch it. Re-running the evals did.

That middle column is the argument for this whole loop: it is the failure that
passes a smoke test.

Two things worth reading carefully.

**`answer_is_correct` is code, not a judge.** The failure is a plausible wrong
number, and a judge reading "ORD carries the most, with 118" scores it well.
Every example carries a `truth_sql`; the evaluator runs it and compares.

**`rows_stayed_out_of_context` is 1.000 on both** — and that is the honest
finding. The naive build never blows the context budget, because the harness
spills the oversized result to a file first. It stays cheap and becomes wrong.
Measuring only context size would have shown no problem at all.

### Is it consistent?

Three repetitions, 24 scored runs per build. Both derived questions fail **0/3**
on v1 and pass 3/3 on v2:

| question | database | v1 | v2 |
|---|---|---|---|
| which station carries the most open deferrals | 105 | 0/3 | 3/3 |
| distinct 737-800 aircraft with an open category C deferral | 107 | 0/3 | 3/3 |

The other six pass on both builds every time. When the catalogue has a
pre-aggregated query, the naive build calls it and is fine — the architecture
earns its keep exactly where the catalogue runs out.

**v1's score is itself unstable**: 18, 20, 20 and 21 out of 24 across four runs
of the same unchanged suite, with both derived questions scoring anywhere from
0/3 to 1/3. The instability is the finding, not noise around it — a smoke test
run once would have passed. That is why `scripts/experiment.sh` defaults to
three repetitions.

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
