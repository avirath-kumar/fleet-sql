"""The run summary both the eval harness and live traffic emit.

ONE definition, imported by both, because an online evaluator scores a run's
outputs and can only read fields that are there. The traffic generator first
invoked the agent directly, so its root run carried agent state and nothing
else -- every online evaluator returned None on 12 of 12 traces while looking
correctly configured.

If a metric matters in an experiment it has to be emitted in production too, or
the dataset and the live project are measuring different things.
"""
from __future__ import annotations

#: Where deepagents parks a tool result it had to evict. NOTE this is a
#: RECOVERY path, not prevention: `_overflow_clip` runs only after
#: SummarizationMiddleware catches a ContextOverflowError, so a result can be
#: enormous and still never land here if the context has not blown yet. Our own
#: offload happens before the rows exist as a message at all.
HARNESS_OFFLOAD_MARKER = "/large_tool_results/"


def summarize(out: dict) -> dict:
    """Agent output -> the fields the evaluators score."""
    msgs = out.get("messages") or []
    payload = tokens = evicted = calls = 0
    for m in msgs:
        if getattr(m, "type", None) == "tool":
            content = str(m.content)
            payload += len(content)
            evicted += HARNESS_OFFLOAD_MARKER in content
            calls += 1
        tokens += (getattr(m, "usage_metadata", None) or {}).get("total_tokens", 0)
    answer = out.get("structured_response")
    return {
        "structured_response": answer.model_dump() if answer is not None else {},
        "tool_payload_chars": payload,
        "total_tokens": tokens,
        #: Tool results the HARNESS had to evict after an overflow. Nonzero
        #: means rows reached the context and were rescued, not prevented.
        "harness_evicted_results": evicted,
        "tool_calls": calls,
    }
