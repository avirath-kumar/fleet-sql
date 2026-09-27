"""What we measure.

The headline metric is `answer_is_correct`, and it is deliberately code rather
than a judge: the failure this demo is about is an agent that states a wrong
number confidently, and a judge reading "ORD carries the most, with 103" scores
it well because it reads exactly like the right answer. Only the database can
settle it.

The rest measure the cost of getting there, which is the other half of the
argument: an architecture that is correct but ruinous has not solved anything.
"""
from __future__ import annotations

import pathlib
import re
import sqlite3
from typing import Any

DB = pathlib.Path(__file__).resolve().parents[1] / "data" / "fleet.db"

#: Tool payload over this many characters means rows reached the conversation.
#: A receipt is ~2 KB; a 500-row result inline is ~140 KB. Anything above this
#: is the naive shape, whatever the agent then did with it.
PAYLOAD_BUDGET = 20_000


def _truth(sql: str) -> int:
    con = sqlite3.connect(DB)
    try:
        return int(con.execute(sql).fetchone()[0])
    finally:
        con.close()


def _response(run: Any) -> dict:
    out = getattr(run, "outputs", None) or {}
    return out.get("structured_response") or out.get("response") or out or {}


def _stated_numbers(run: Any) -> list[int]:
    """Every figure the answer asserts: the typed `counts`, plus any integer in
    the prose. The prose is included because an answer that states the number
    only in text is still stating it."""
    r = _response(run)
    nums = [int(c["value"]) for c in (r.get("counts") or [])
            if isinstance(c, dict) and str(c.get("value", "")).lstrip("-").isdigit()]
    for m in re.finditer(r"\b(\d{1,6})\b", str(r.get("answer") or "")):
        nums.append(int(m.group(1)))
    return nums


def answer_is_correct(run: Any, example: Any) -> dict:
    """Does the answer state the number the database gives?

    Scores on presence of the true figure among the numbers asserted. Being
    generous about WHERE the number appears is on purpose -- the question is
    whether the agent arrived at the right figure, not whether it filled in a
    field correctly.
    """
    sql = ((getattr(example, "outputs", None) or {}).get("truth_sql"))
    if not sql:
        return {"key": "answer_is_correct", "score": None, "comment": "no ground truth"}
    truth = _truth(sql)
    stated = _stated_numbers(run)
    hit = truth in stated
    label = (getattr(example, "outputs", None) or {}).get("label", "")
    return {"key": "answer_is_correct", "score": 1.0 if hit else 0.0,
            "comment": (f"{label}: database says {truth}; answer stated "
                        f"{stated[:8] if stated else 'no figures'}")}


def rows_stayed_out_of_context(run: Any, example: Any = None) -> dict:
    """Did query rows reach the conversation?

    Measured on tool-output size rather than on which tool was called, because
    the thing that hurts is bytes in the context window, and an agent can reach
    that total by any route.
    """
    out = getattr(run, "outputs", None) or {}
    chars = out.get("tool_payload_chars")
    if chars is None:
        return {"key": "rows_stayed_out_of_context", "score": None,
                "comment": "target did not report tool payload size"}
    ok = chars <= PAYLOAD_BUDGET
    return {"key": "rows_stayed_out_of_context", "score": 1.0 if ok else 0.0,
            "comment": f"{chars:,} chars of tool output "
                       f"({'within' if ok else 'over'} the {PAYLOAD_BUDGET:,} budget)"}


def context_efficiency(run: Any, example: Any = None) -> dict:
    """Tokens spent, normalised so lower is better and the score is readable.

    1.0 at or under 10k tokens, 0.0 at 60k and above, linear between. A score
    rather than the raw count because an experiment table of raw token counts
    invites comparing two arms that answered different numbers of questions.
    """
    out = getattr(run, "outputs", None) or {}
    tokens = out.get("total_tokens")
    if not tokens:
        return {"key": "context_efficiency", "score": None, "comment": "no token usage recorded"}
    lo, hi = 10_000, 60_000
    score = 1.0 if tokens <= lo else 0.0 if tokens >= hi else (hi - tokens) / (hi - lo)
    return {"key": "context_efficiency", "score": round(score, 3),
            "comment": f"{tokens:,} tokens"}


def no_silent_truncation(run: Any, example: Any = None) -> dict:
    """Did the agent hand off an oversized payload to something untyped?

    The harness spills a large tool result to /large_tool_results/ and the agent
    then reads it back, usually by delegating with a prose `task` string. That
    path is where the wrong numbers came from, so it is worth scoring on its
    own rather than inferring it from the payload size.
    """
    out = getattr(run, "outputs", None) or {}
    spilled = out.get("spilled_tool_results")
    if spilled is None:
        return {"key": "no_silent_truncation", "score": None,
                "comment": "target did not report spills"}
    return {"key": "no_silent_truncation", "score": 0.0 if spilled else 1.0,
            "comment": (f"{spilled} tool result(s) spilled to a file and re-read"
                        if spilled else "no oversized tool result")}


ALL = [answer_is_correct, rows_stayed_out_of_context, context_efficiency, no_silent_truncation]
