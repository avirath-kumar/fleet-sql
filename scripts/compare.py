"""Ask one question of both variants and report what it cost each."""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "src/agent")
import _env  # noqa: F401
from agent import build_agent

QUESTION = sys.argv[1] if len(sys.argv) > 1 else \
    "How many open deferrals does the 737-800 fleet have, and how do they break down by MEL category?"


def tool_payload_chars(messages) -> int:
    """Bytes of tool output that entered the conversation. The number this
    architecture exists to control."""
    total = 0
    for m in messages:
        if getattr(m, "type", None) == "tool":
            total += len(str(m.content))
    return total


def run(variant: str) -> dict:
    agent = build_agent(variant)
    t0 = time.monotonic()
    out = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    secs = time.monotonic() - t0
    msgs = out["messages"]
    usage = 0
    for m in msgs:
        u = getattr(m, "usage_metadata", None) or {}
        usage += u.get("total_tokens", 0)
    ans = out.get("structured_response")
    return {"variant": variant, "seconds": round(secs, 1),
            "tool_chars": tool_payload_chars(msgs), "tokens": usage,
            "tool_calls": sum(1 for m in msgs if getattr(m, "type", None) == "tool"),
            "answer": (ans.answer if ans else "(no structured answer)"),
            "counts": ([c.model_dump() for c in ans.counts] if ans else [])}


print(f"Q: {QUESTION}\n")
for v in ("v1", "v2"):
    try:
        r = run(v)
    except Exception as exc:
        print(f"  {v}: FAILED — {str(exc)[:300]}\n")
        continue
    print(f"  {r['variant']}  {r['seconds']:>5.1f}s  {r['tool_calls']} tool call(s)  "
          f"tool payload {r['tool_chars']:>8,} chars  ~{r['tokens']:>7,} tokens")
    print(f"      counts: {json.dumps(r['counts'])[:150]}")
    print(f"      {r['answer'][:230]}\n")
