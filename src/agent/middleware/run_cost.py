"""What the run cost, carried in agent state.

`agent.invoke()` should return everything an evaluator needs, and these are
facts about the run, so they belong in its output rather than in a helper the
caller has to remember to apply.

This replaces a `summarize()` function that both the eval harness and the
traffic generator had to call. That arrangement had already failed once: the
traffic generator did not call it, so its root run carried agent state and
nothing else, and every online evaluator returned None on 12 of 12 traces while
looking correctly configured. A field that is part of state cannot be forgotten.
"""
from __future__ import annotations

from langchain.agents.middleware import AgentState, after_agent

#: Where deepagents parks a tool result it had to evict. NOTE this is a
#: RECOVERY path, not prevention: `_overflow_clip` runs only after
#: SummarizationMiddleware catches a ContextOverflowError, so a result can be
#: enormous and never land here if the context has not blown yet. Our own
#: offload happens before rows are ever a message.
HARNESS_OFFLOAD_MARKER = "/large_tool_results/"


class RunCost(AgentState):
    """Agent state plus what the run cost to produce."""
    #: Characters of tool output that entered the conversation. The number this
    #: whole architecture exists to hold down.
    tool_payload_chars: int
    total_tokens: int
    #: Tool results the HARNESS evicted after an overflow. Nonzero means rows
    #: reached the context and were rescued, not prevented.
    harness_evicted_results: int
    tool_calls: int


@after_agent(state_schema=RunCost)
def run_cost(state: RunCost, runtime) -> dict:
    payload = tokens = evicted = calls = 0
    for m in state["messages"]:
        if getattr(m, "type", None) == "tool":
            content = str(m.content)
            payload += len(content)
            evicted += HARNESS_OFFLOAD_MARKER in content
            calls += 1
        tokens += (getattr(m, "usage_metadata", None) or {}).get("total_tokens", 0)
    return {"tool_payload_chars": payload, "total_tokens": tokens,
            "harness_evicted_results": evicted, "tool_calls": calls}
