"""The run summary both the eval harness and production traffic emit.

ONE definition, imported by both, because an online evaluator reads a run's
outputs and can only score fields that are actually there. The first version of
generate_traces.py invoked the agent directly, so its root run carried the
agent's state and nothing else -- and every online evaluator returned None on
12 of 12 traces while looking perfectly configured.

If a metric matters in an experiment it has to be emitted in production too,
or the dataset and the live project are measuring different things.
"""
from __future__ import annotations

#: Where the deepagents harness puts a tool result too big to inline.
SPILL_MARKER = "/large_tool_results/"


def summarize(out: dict) -> dict:
    """Agent output -> the fields the evaluators score."""
    msgs = out.get("messages") or []
    payload, tokens, spilled, calls = 0, 0, 0, 0
    for m in msgs:
        if getattr(m, "type", None) == "tool":
            content = str(m.content)
            payload += len(content)
            spilled += SPILL_MARKER in content
            calls += 1
        usage = getattr(m, "usage_metadata", None) or {}
        tokens += usage.get("total_tokens", 0)
    ans = out.get("structured_response")
    return {
        "structured_response": ans.model_dump() if ans is not None else {},
        "tool_payload_chars": payload,
        "total_tokens": tokens,
        "spilled_tool_results": spilled,
        "tool_calls": calls,
    }
