# Architecture

## The rule

**Rows are never a tool's return value.** Everything else follows.

A pre-approved query runs, its rows go to a file, and the tool returns a
receipt. Later tools work on the file. Nothing that scales with row count ever
becomes a message.

## Where results are written

```
.results/
  index.json              every result this session, with lineage
  res_55734893f9.jsonl    one JSON object per line, 511 of them
  res_55734893f9.csv      written only if export_report was called
```

JSONL because it appends without parsing, streams without loading, and one
corrupt line costs one row. The directory is gitignored — results are derived
and rebuildable from `data/fleet.db`.

`index.json` is the part that makes a receipt durable:

```json
{"res_55734893f9": {"query": "open_deferrals_by_fleet",
                    "params": {"model": "737-800"},
                    "row_count": 511, "parent_id": null, "derived_by": null},
 "res_a1b2c3d4e5": {"query": "filter(res_55734893f9)",
                    "params": {"column": "station_code", "op": "eq", "value": "ORD"},
                    "row_count": 105, "parent_id": "res_55734893f9",
                    "derived_by": "station_code eq ORD"}}
```

A file, not memory. A `result_id` lives in the message that carried it, and
messages get compacted — at which point the rows are still on disk with nothing
pointing at them. The index is what `list_results` reads, and why a handle from
three turns ago still works.

## How results are queried

`run_query(query_id, params)` → `results.query()` → `results.save()` → receipt:

```json
{"result_id": "res_55734893f9", "query": "open_deferrals_by_fleet",
 "params": {"model": "737-800"},
 "row_count": 511,
 "columns": ["deferral_id", "tail_number", "mel_ref", "category", ...],
 "preview": [ ... 5 rows ... ],
 "breakdown": {"category":     {"C": 230, "D": 179, "B": 65, "A": 37},
               "station_code": {"ORD": 105, "ANC": 102, "SIN": 91, ...},
               "status":       {"open": 438, "extended": 73}},
 "path": ".results/res_55734893f9.jsonl",
 "parent_id": null, "derived_by": null}
```

2,079 bytes for a result whose rows are 139,515. Four parts, each earning its
place:

- **`row_count` is exact.** Truncating to "the first 20 of something" loses the
  number, and the number is usually the question.
- **`preview` shows shape, not content.** Five rows, so the agent knows what a
  row looks like without reading any.
- **`breakdown` is counted in Python** at write time. The most common follow-up
  — "how does that split?" — is answered before it is asked, with no model
  arithmetic.
- **`path`** names the file. The agent's backend is rooted at `.results/`,
  so `grep`, `ls` and `read_file` genuinely reach it — see below.

Results of 25 rows or fewer come back inline. Forcing a second call to read six
rows is friction with no benefit.

### The breakdown doubles as the menu

`breakdown`'s keys are the columns you can filter on; their keys are the values
present. That is the whole answer to "a receipt says 511 rows and then what?" —
the agent can say *"511 open, across six stations and four categories, and I
can narrow to any of them"* without reading a row.

It is derived from the data, bounded at 8 distinct values. A column with more
is omitted rather than truncated: a half-listed set invites the agent to assume
the values it cannot see do not exist.

## How results are modified

Every modification returns a **new receipt**, never rows.

| tool | returns | cost at 500 rows |
|---|---|---|
| `aggregate_result(result_id, group_by)` | a small table | same as at 20 |
| `filter_result(result_id, column, op, value)` | a new receipt | same as at 20 |
| `describe_result(result_id)` | column shape | same as at 20 |
| `page_result(result_id, offset, limit)` | rows, **capped at 50** | bounded |
| `list_results()` | the index | bounded |
| `export_report(result_id, title)` | a path | bounded |

### Reading the actual rows

Three ways, in increasing cost:

| need | tool |
|---|---|
| "how many / how does it split" | the receipt — `row_count`, `breakdown`, `distinct` |
| "find the ones mentioning X" | **`grep`** over `.results/<id>.jsonl` |
| "show me some records" | `page_result`, capped at 50, with column projection |
| "I want all of them" | `export_report` → a CSV path for the user |

The backend is rooted at `.results/` rather than the project root, so the agent
gets `ls`, `glob`, `grep` and `read_file` over the offloaded rows **and nothing
else**. grep is the right tool for "find the row that says X" — better than
anything hand-written, and it returns matching lines rather than the file.

Measured: asked to find 737-800 deferrals mentioning a shimmy damper, the agent
ran the query, grepped the JSONL, and answered "27 descriptions, on 26 tails,
N155FL being the only tail with two". The database agrees exactly.

`read_file` on a 140 KB result is the one path that puts rows back in context.
It is available because occasionally it is the right answer, and because a tool
set that forbids the expensive option tends to produce elaborate ways of
reaching it anyway.

### Filtering

`filter_result` supports `eq ne in lt lte gt gte contains`, one column at a
time, and is **chainable** — multi-condition narrowing is a sequence of
filters, each recording its parent. There is no `OR` and no cross-column
predicate; that is a real limit, and the workaround is a broader filter plus an
aggregate.

### Lineage, and how reverting works

```
run_query ─────────────> res_A  (511)   parent: none
                            │
   filter station_code=ORD  │
                            ▼
                         res_B  (105)   parent: res_A
                                        derived_by: "station_code eq ORD"
```

"Go back to all of them" is not a special tool. `list_results` shows `res_B`'s
parent is `res_A`, so the agent filters from `res_A` instead. There was a
`widen_result` for this; it was deleted, because two tools for one capability
is a choice the model should not have to make, and it always reached for
`list_results` anyway.

### Counts are computed in Python, never by the model

This is the line that fixes the wrong answers. `aggregate_result` groups in
code. The naive build said ORD had 118 open deferrals because a model read
140 KB of JSON and counted; this one says 105 because Python did.

## Typed delegation

`analyze_result(result_id, question, group_by)` replaces `task("go look at
this")`.

```python
class AnalysisFinding(BaseModel):
    answer: str
    row_count: int
    evidence: list[Evidence]      # {column, value, count}
    rows_inspected: int
    narrowed_result_id: str | None
```

Two properties matter. The analyst has the result tools but **not**
`run_query`, so it cannot open a second route to bulk rows. And `evidence` is
typed, so an evaluator can check that a number was counted rather than guessed
— which a prose answer makes impossible.

## Why there is no "offload large tool results" middleware

deepagents ships one, in `middleware/_overflow_clip.py`. We do not use it, and
the reason is the whole argument.

**That middleware is a recovery path.** From its own docstring: it runs when
`SummarizationMiddleware`'s `wrap_model_call` catches a `ContextOverflowError`.
By the time it fires, the rows have already been a message — it evicts them to
`/large_tool_results/{tool_call_id}` and hands the agent a stub to read back.

Three consequences, all of which we measured:

1. **It fires late, or not at all.** One v1 run carried 135,761 characters
   straight into context and was never evicted, because nothing had overflowed
   yet. A backstop keyed on overflow cannot prevent the cost that precedes
   overflow.
2. **What it hands back is a file of JSON rows.** The agent still has to read
   and count them, which is where the wrong numbers came from.
3. **Middleware is necessarily reactive.** It sees a tool result after the tool
   returned it. Anything it does is damage control.

Offloading at the **tool** instead of in middleware means there is nothing to
intercept: `run_query` never constructs a 500-row message, so no layer has to
rescue one. And because the tool knows what it just wrote, it can return counts
and a breakdown — which middleware cannot, having only an opaque blob.

The general form: **middleware is the right place for a policy that must apply
to tools you do not control.** For tools you own, the tool is the right place,
and the result is smaller, earlier and more informative.

## What a trace shows

`response_format=Answer` makes the structured output the final message, so
without intervention the Messages view of every trace ends in this:

    {"answer":"The 747-8 fleet has 12 aircraft...","counts":[{"label":...

The answer is in there, wrapped in the fields around it. `readable_answer`
rewrites that message's content with the prose, by returning an AIMessage with
**the same id** -- `add_messages` treats a repeated id as an overwrite rather
than an append. Appending a second message does not work: the blob remains,
still above the readable one.

The result is the split a trace wants:

| where | what |
|---|---|
| `messages[-1]` | the prose answer, and nothing else |
| `structured_response` | `counts`, `result_ids`, `report_path` |
| run state | `tool_payload_chars`, `total_tokens`, `harness_evicted_results` |

Nothing is lost, which matters here specifically: `answer_is_correct` reads
`counts` off `structured_response` to check a figure against the database, and
never looks at `messages`. Verified after the change -- all four evaluators
score the new shape unchanged.

Three guards keep it from damaging a transcript. It fires only when there is
prose to show, only on an AI message, and only when that message's text starts
with `{` -- so a real turn is never overwritten.

## What the run cost, and where it lives

`middleware/run_cost.py` extends the agent's state:

```python
class RunCost(AgentState):
    tool_payload_chars: int
    total_tokens: int
    harness_evicted_results: int
    tool_calls: int

@after_agent(state_schema=RunCost)
def run_cost(state, runtime) -> dict: ...
```

`create_deep_agent` takes both `state_schema` and `middleware`, so these are
part of what `agent.invoke()` returns. Every caller gets them -- the
experiment, live traffic, a notebook -- without doing anything.

This replaced a `summarize()` helper both callers had to remember to apply, an
arrangement that had already failed: the traffic generator did not call it, so
its root run carried agent state and nothing else, and every online evaluator
returned None on 12 of 12 traces while looking correctly configured. **A field
that is part of state cannot be forgotten.**

`generate_traces.py` still wraps each question in `@traceable`, but for a
smaller reason: an online evaluator attaches to a run, so something has to be
the root. It no longer decides what the fields are.

### What is not measured here

The metrics above are facts about the OUTCOME and the cost. They say nothing
about the PATH -- and v1 and v2 differ precisely in which tools they call. A
trajectory evaluator (`agentevals`, trajectory match or LLM judge) would catch
"delegated with task() instead of analyze_result" directly, rather than
inferring it from `harness_evicted_results`. LangSmith also exposes a
`trajectory` variable to online thread evaluators, though only in the GCP US
region. Neither is here yet; both would add a dimension rather than replace one.

## Both builds

| | v1 | v2 |
|---|---|---|
| tools | `run_query_inline` | the eight above |
| what a query returns | every row | a receipt |
| who counts | the model, reading | Python |
| delegation | `task(prose)` | `analyze_result(typed)` |

The harness, model, prompt scaffold and catalogue are identical. Only the tools
differ, so an experiment comparing them measures the result-handling
architecture and nothing else.
