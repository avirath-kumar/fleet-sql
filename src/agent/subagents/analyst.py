"""Delegation with a schema, instead of `task("go look at this")`.

Deep Agents ships a `task` tool whose entire interface is a prose string. That
is fine for open-ended work and wrong for this: the orchestrator already knows
exactly what it wants -- a named result, a specific question, a column to group
on -- and a string throws that structure away and asks the subagent to recover
it by reading. Two costs follow. The subagent re-derives context the caller
already had, and the answer comes back as prose nobody can assert on.

So the analyst is exposed as an ordinary tool with a Pydantic request and a
Pydantic response. The orchestrator fills in fields; the finding comes back
with `rows_inspected` and `evidence` attached, which is what makes an evaluator
able to check that a number was counted rather than guessed.

It runs with the result tools ONLY -- no run_query. A subagent that can start
new queries is a second way to pull 500 rows into a context.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field

import results
from model import build_model
from tools.query import (aggregate_result, describe_result, filter_result,
                         page_result)


class Evidence(BaseModel):
    """One counted fact, so a claim can be checked rather than believed."""
    column: str = Field(description="Column the breakdown was taken over")
    value: str = Field(description="Value within that column")
    count: int = Field(description="Rows matching it")


class AnalysisFinding(BaseModel):
    """What the analyst returns. Typed, so evaluators can assert on it."""
    answer: str = Field(description="Two or three sentences, no row dumps")
    row_count: int = Field(description="Rows in the result examined")
    evidence: list[Evidence] = Field(default_factory=list,
                                     description="Counts backing the answer")
    rows_inspected: int = Field(0, description="Rows actually read, via page_result")
    narrowed_result_id: str | None = Field(
        None, description="Set when the analyst produced a narrower result worth keeping")


ANALYST_PROMPT = """You analyse ONE already-materialised query result.

You cannot run queries. You have describe_result, aggregate_result,
filter_result and page_result, and they all work on the result id you are given.

Work from counts. aggregate_result answers most questions about a large result
without reading a single row, and it costs the same whether the result has 20
rows or 5,000. Reach for page_result only when the question genuinely needs
specific records, and then ask for the columns you need rather than all of them.

Put the numbers you relied on in `evidence`, and set `rows_inspected` to the
number of rows you actually read."""


@tool(parse_docstring=True)
def analyze_result(result_id: str, question: str,
                   group_by: str | None = None) -> dict:
    """Delegate analysis of a stored result to a focused analyst subagent.

    Use this when a result is too large to reason about directly and the
    question needs more than one breakdown. The analyst sees only this result,
    and returns a typed finding with the counts it relied on.

    Args:
        result_id: The result to analyse.
        question: The specific question to answer about it.
        group_by: Column to break down by, when you already know which one.
    """
    rows = results.load(result_id)          # fails loudly on a bad id, before the model call
    agent = create_agent(
        model=build_model(),
        tools=[describe_result, aggregate_result, filter_result, page_result],
        system_prompt=ANALYST_PROMPT,
        response_format=AnalysisFinding,
    )
    ask = f"Result id: {result_id} ({len(rows)} rows).\nQuestion: {question}"
    if group_by:
        ask += f"\nStart by grouping on: {group_by}"
    out = agent.invoke({"messages": [{"role": "user", "content": ask}]})
    finding = out.get("structured_response")
    if finding is None:
        return {"answer": "the analyst returned no structured finding",
                "row_count": len(rows), "evidence": [], "rows_inspected": 0}
    return finding.model_dump()
