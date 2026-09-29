"""The fleet agent, in two variants.

The harness is identical in both; only the tools differ. That is deliberate —
it means an experiment comparing them is measuring the result-handling
architecture and not a prompt, a model or a framework difference.
"""
from __future__ import annotations

import os

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from pydantic import BaseModel, Field

import _env  # noqa: F401  (side effect: loads .env)
import results
from catalog import describe_catalog
from model import build_model
from middleware.run_cost import RunCost, run_cost
from tools.query import V1_TOOLS

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

For "which X has the most" or "rank by X" questions, call the matching `*_counts_by_*` query; never enumerate groups one at a time or pull an unbounded row query.

Answer from the data, state the figures you relied on, and put every number you
assert into `counts`."""

V1_PROMPT = _SHARED + """

Use run_query_inline to run a query and read its rows."""

V2_PROMPT = V1_PROMPT

def build_agent(variant: str | None = None):
    """v1 naive | v2 offload | v2-regressed, which is v2 with a real regression.

    v2-regressed has v2's tools and prompt and differs only in what the receipt
    carries: no `distinct`. It exists so the suite can demonstrate a cleanup
    breaking something, which is the failure evals catch and review does not.
    """
    v = (variant or VARIANT).lower()
    tools = list(V1_TOOLS)
    return create_deep_agent(
        model=build_model(),
        tools=tools,
        # Rooted at .results/, NOT the project root: the agent gets ls, glob,
        # grep and read_file over the offloaded rows and nothing else. This is
        # the escape hatch for "find me the row that says X" -- grep is better
        # at that than any tool I would write, and it returns matching lines
        # rather than the file.
        backend=FilesystemBackend(root_dir=str(results.DIR)),
        # What the run cost rides in state, so agent.invoke() returns it and
        # every caller -- experiment, live traffic, a notebook -- gets the
        # fields an evaluator scores without doing anything.
        # readable_answer first: it rewrites the final message, and run_cost
        # counts tool payload, so neither depends on the other's result.
        middleware=[run_cost],
        state_schema=RunCost,
        system_prompt=V1_PROMPT if v == "v1" else V2_PROMPT,
        response_format=Answer,
    )


agent = build_agent()
