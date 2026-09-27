#!/usr/bin/env python3
"""Put realistic traffic through the agent so there is something to inspect.

    scripts/generate_traces.py --variant v1 --tag before-fix

This is the first step of the loop: a prototype answering real questions in a
project, before anyone has written a dataset. The questions are the ones a
configuration team actually asks -- a mix the catalogue covers directly and a
mix it does not, because the second kind is where the trouble is.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/agent"))
sys.path.insert(0, str(ROOT))

import _env  # noqa: E402,F401
import os  # noqa: E402

QUESTIONS = [
    "How many top-level configuration slots does the 747-8 have?",
    "What open deferrals are on N101FL?",
    "How many aircraft are in the 747-8 fleet?",
    "How many open deferrals does the 737-800 fleet have?",
    "Which station carries the most open deferrals on the 737-800 fleet, and how many?",
    "On the 737-800 fleet, how many open deferrals are in ATA chapter 25?",
    "How many distinct 737-800 aircraft have at least one open category C deferral?",
    "Across the whole fleet, how many open deferrals fall due within 7 days?",
    "How many open deferrals does the A320-200 fleet have in ATA chapter 33?",
    "Which ATA chapter has the most open deferrals across the A320-200 fleet?",
    "How many 737-800 aircraft are based at MEM?",
    "Give me the open category A deferrals on the 737-800 fleet.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="v1", choices=["v1", "v2"])
    ap.add_argument("--tag", default=None, help="Run tag, e.g. before-fix")
    ap.add_argument("--limit", type=int, default=len(QUESTIONS))
    args = ap.parse_args()

    os.environ.setdefault("LANGSMITH_PROJECT", "fleet-sql")
    from agent import build_agent
    from langchain_core.tracers.context import tracing_v2_enabled

    agent = build_agent(args.variant)
    tag = args.tag or f"{args.variant}-traffic"
    wrong = 0
    for i, q in enumerate(QUESTIONS[: args.limit], start=1):
        t0 = time.monotonic()
        try:
            with tracing_v2_enabled(project_name=os.environ["LANGSMITH_PROJECT"],
                                    tags=[tag, args.variant]):
                out = agent.invoke({"messages": [{"role": "user", "content": q}]})
            msgs = out["messages"]
            payload = sum(len(str(m.content)) for m in msgs
                          if getattr(m, "type", None) == "tool")
            spill = sum("/large_tool_results/" in str(m.content) for m in msgs
                        if getattr(m, "type", None) == "tool")
            wrong += bool(spill)
            print(f"  [{i:2}/{args.limit}] {time.monotonic()-t0:5.1f}s  "
                  f"{payload:>6,} chars  {'SPILLED' if spill else '       '}  {q[:56]}")
        except Exception as exc:  # noqa: BLE001 - one bad run must not stop traffic
            print(f"  [{i:2}/{args.limit}] FAILED {str(exc)[:80]}")
    print(f"\n  {args.limit} runs into project "
          f"{os.environ['LANGSMITH_PROJECT']!r}, tagged {tag!r}")
    print(f"  {wrong} spilled an oversized tool result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
