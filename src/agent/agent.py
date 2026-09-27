"""The fleet agent, in two variants.

The harness is identical in both; only the tools differ. That is deliberate —
it means an experiment comparing them is measuring the result-handling
architecture and not a prompt, a model or a framework difference.
"""
from __future__ import annotations

import os

from deepagents import create_deep_agent
from pydantic import BaseModel, Field

import _env  # noqa: F401  (side effect: loads .env)
from catalog import describe_catalog
from model import build_model
from subagents.analyst import analyze_result
from tools.query import V1_TOOLS, V2_TOOLS

VARIANT = os.environ.get("AGENT_VARIANT", "v2").lower()


class Count(BaseModel):
    """A number the answer states, pulled out so it can be checked."""
    label: str = Field(description="What was counted, e.g. 'open deferrals on the 737-800 fleet'")
    value: int = Field(description="The figure stated in the answer")


class Answer(BaseModel):
    answer: str = Field(description="The reply, in prose. Never a dump of rows.")
    counts: list[Count] = Field(default_factory=list,
                                description="Every figure the answer asserts")
    result_ids: list[str] = Field(default_factory=list,
                                  description="Result ids consulted")
    report_path: str | None = Field(
        None, description="Path to an exported file, when one was written")


_SHARED = f"""You answer questions about an aircraft fleet: airframes, their
configuration, and open maintenance deferrals (MEL items).

You do not write SQL. You call one of these pre-approved queries by id and
supply its parameters:

{describe_catalog()}

Answer from the data, state the figures you relied on, and put every number you
assert into `counts`."""

V1_PROMPT = _SHARED + """

Use run_query_inline to run a query and read its rows."""

V2_PROMPT = _SHARED + """

HOW RESULTS WORK. run_query does not return rows. It returns a receipt: the
exact row count, the columns, a five-row preview, and breakdowns of the
low-cardinality columns. The rows go to a file. A small result comes back
inline; a large one does not.

Answer from the receipt whenever the receipt is enough — it usually is. "How
many open deferrals does the 737-800 fleet have" is answered by `row_count`,
and "how do they split by category" by the `category` facet, without reading a
single row.

When you need more:
  aggregate_result  group and count. Costs the same at 20 rows or 5,000.
  filter_result     narrow to a new result; chain it to narrow again.
  describe_result   column types, distinct counts, ranges.
  page_result       specific records, bounded. Ask for the columns you need.
  list_results      what has already been run, with sizes and lineage.
  analyze_result    hand a large result to an analyst subagent with a question.
  export_report     write the full set to a file and cite its path.

RESULTS PERSIST. A result id stays valid after the turn that made it. When a
follow-up refers to something already run -- "narrow that to ORD", "what about
category A", "go back to all of them" -- call list_results to find it. A
filtered result lists its `parent_id`, so reverting a filter means filtering
from the parent, not re-running the query. Re-running the query is wrong: it costs a round
trip and, if you guess different parameters, silently answers a different
question.

TELL THE USER WHAT THEY CAN ADJUST. Every receipt carries `breakdown`: its keys
are the columns worth narrowing on, and their keys are the values present.
`distinct` names the rest, with how many values each has. When you hand back
a count or a report, say what the options are -- "511 open, across six stations
and four categories; I can break it down or narrow to any of them" -- rather
than stopping at the number. A user who cannot see the axes cannot ask for the
next thing.

NEVER try to list hundreds of rows in an answer. If someone asks for something
that has hundreds of matches, give them the count and the shape of it, then
either narrow the question with them or export a report and cite the path.
That is the correct answer, not a consolation prize."""


def build_agent(variant: str | None = None):
    v = (variant or VARIANT).lower()
    tools = list(V1_TOOLS) if v == "v1" else [*V2_TOOLS, analyze_result]
    return create_deep_agent(
        model=build_model(),
        tools=tools,
        system_prompt=V1_PROMPT if v == "v1" else V2_PROMPT,
        response_format=Answer,
    )


agent = build_agent()
