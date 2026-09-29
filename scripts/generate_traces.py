#!/usr/bin/env python3
"""Put realistic traffic through the agent so there is something to inspect.

    scripts/generate_traces.py --variant v1 --tag before-fix
    scripts/generate_traces.py --variant v1 --tag before-fix \
        --deployment-url https://your-deployment.us.langgraph.app

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
from urllib.parse import urlsplit

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
    ap.add_argument("--variant", default="v1", choices=["v1", "v2"],
                    help="Local agent variant; in hosted mode this is only a trace tag")
    ap.add_argument("--tag", default=None, help="Run tag, e.g. before-fix")
    ap.add_argument("--limit", type=int, default=len(QUESTIONS))
    ap.add_argument("--deployment-url", help="Send runs to a LangSmith Cloud deployment")
    args = ap.parse_args()

    if args.limit < 1:
        ap.error("--limit must be at least 1")
    hosted = args.deployment_url is not None
    if hosted:
        url = urlsplit(args.deployment_url)
        if (url.scheme != "https" or not (url.hostname or "").endswith(".langgraph.app")
                or url.username or url.password or url.port not in (None, 443)):
            ap.error("--deployment-url must be an HTTPS LangSmith Cloud URL")
        from langgraph_sdk import get_sync_client

        client = get_sync_client(
            url=args.deployment_url, api_key=os.environ["LANGSMITH_API_KEY"]
        )
    else:
        os.environ.setdefault("LANGSMITH_PROJECT", "fleet-sql")
        from agent import build_agent
        from langchain_core.tracers.context import tracing_v2_enabled

        agent = build_agent(args.variant)

    #: No wrapper. The agent IS the root run, so its outputs are its state:
    #: `messages` (prose, thanks to readable_answer) plus structured_response
    #: and the run-cost fields. LangSmith renders a Messages view when outputs
    #: carry `messages`, and a wrapper returning a bare dict rendered as
    #: generic Fields instead -- which is what this used to do.
    tag = args.tag or f"{args.variant}-traffic"
    evicted = 0
    succeeded = 0
    selected = QUESTIONS[: args.limit]
    for i, q in enumerate(selected, start=1):
        t0 = time.monotonic()
        try:
            inputs = {"messages": [{"role": "user", "content": q}]}
            if hosted:
                out = client.runs.wait(
                    None, "agent", input=inputs,
                    config={"tags": [tag, args.variant, "hosted"]},
                )
            else:
                with tracing_v2_enabled(project_name=os.environ["LANGSMITH_PROJECT"],
                                        tags=[tag, args.variant]):
                    out = agent.invoke(inputs)
            if not isinstance(out, dict):
                raise TypeError("Agent did not return a state object")
            chars = out.get("tool_payload_chars")
            chars = chars if isinstance(chars, int) else 0
            evicted += bool(out.get("harness_evicted_results"))
            succeeded += 1
            print(f"  [{i:2}/{len(selected)}] {time.monotonic()-t0:5.1f}s  "
                  f"{chars:>7,} chars  "
                  f"{'EVICTED' if out.get('harness_evicted_results') else '       '}  {q[:52]}")
        except Exception as exc:  # noqa: BLE001 - one bad run must not stop traffic
            print(f"  [{i:2}/{len(selected)}] FAILED {type(exc).__name__}")
    target = ("the deployment's linked LangSmith project" if hosted else
              f"project {os.environ['LANGSMITH_PROJECT']!r}")
    print(f"\n  {succeeded}/{len(selected)} successful runs into {target}, tagged {tag!r}")
    print(f"  {evicted} needed the harness to evict an oversized result")
    return 0 if succeeded == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
