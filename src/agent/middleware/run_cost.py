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

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langgraph.types import Command

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


@dataclass
class _Counters:
    tool_payload_chars: int = 0
    total_tokens: int = 0
    harness_evicted_results: int = 0
    tool_calls: int = 0
    depth: int = 0


_counters: ContextVar[_Counters | None] = ContextVar("run_cost_counters", default=None)


class RunCostMiddleware(AgentMiddleware[RunCost]):
    """Accumulate model and tool costs while the agent executes."""

    state_schema = RunCost

    def before_agent(self, state: RunCost, runtime) -> None:
        counters = _counters.get()
        if counters is None:
            counters = _Counters(depth=1)
            _counters.set(counters)
        else:
            counters.depth += 1

    def after_agent(self, state: RunCost, runtime) -> dict[str, int]:
        counters = _counters.get() or _Counters()
        result = {
            "tool_payload_chars": counters.tool_payload_chars,
            "total_tokens": counters.total_tokens,
            "harness_evicted_results": counters.harness_evicted_results,
            "tool_calls": counters.tool_calls,
        }
        counters.depth -= 1
        if counters.depth <= 0:
            _counters.set(None)
        return result

    def wrap_tool_call(self, request, handler):
        counters = _counters.get()
        if counters is None:
            counters = _Counters(depth=1)
            _counters.set(counters)
        result = handler(request)
        counters.tool_calls += 1
        for content in _tool_result_contents(result):
            text = str(content)
            counters.tool_payload_chars += len(text)
            if HARNESS_OFFLOAD_MARKER in text:
                counters.harness_evicted_results += 1
        return result

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        counters = _counters.get()
        if counters is None:
            counters = _Counters(depth=1)
            _counters.set(counters)
        response = handler(request)
        for message in response.result:
            counters.total_tokens += (getattr(message, "usage_metadata", None) or {}).get(
                "total_tokens", 0
            )
        return response


def _tool_result_contents(result: Any) -> list[Any]:
    if isinstance(result, ToolMessage):
        return [result.content]
    if isinstance(result, Command):
        return [
            message.content
            for message in result.update.get("messages", [])
            if isinstance(message, ToolMessage)
        ]
    return []


run_cost = RunCostMiddleware()
