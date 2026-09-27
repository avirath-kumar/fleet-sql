"""Score one variant against the dataset.

    python evals/run_experiment.py --variant v2

The target reports what the run COST alongside what it said -- tool payload
size, token usage, and whether the harness spilled an oversized tool result to
a file. Those three are not visible from the answer, and they are half of what
the experiment is comparing.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/agent"))
sys.path.insert(0, str(ROOT))

import _env  # noqa: E402,F401
from langsmith import Client  # noqa: E402

from evals.dataset import DATASET_NAME, upsert  # noqa: E402
from evals.evaluators import ALL  # noqa: E402

#: Where the harness puts a tool result too big to inline. Counting these is
#: how `no_silent_truncation` knows a spill happened.
SPILL_MARKER = "/large_tool_results/"


def make_target(variant: str):
    from agent import build_agent

    agent = build_agent(variant)

    def target(inputs: dict) -> dict:
        out = agent.invoke({"messages": [{"role": "user", "content": inputs["question"]}]})
        msgs = out["messages"]
        payload, tokens, spilled = 0, 0, 0
        for m in msgs:
            if getattr(m, "type", None) == "tool":
                content = str(m.content)
                payload += len(content)
                spilled += SPILL_MARKER in content
            usage = getattr(m, "usage_metadata", None) or {}
            tokens += usage.get("total_tokens", 0)
        ans = out.get("structured_response")
        return {
            "structured_response": ans.model_dump() if ans is not None else {},
            "tool_payload_chars": payload,
            "total_tokens": tokens,
            "spilled_tool_results": spilled,
            "tool_calls": sum(1 for m in msgs if getattr(m, "type", None) == "tool"),
        }

    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default=os.environ.get("AGENT_VARIANT", "v2"),
                    choices=["v1", "v2"])
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--skip-upsert", action="store_true")
    args = ap.parse_args()

    if not args.skip_upsert:
        upsert()
    client = Client()
    result = client.evaluate(
        make_target(args.variant),
        data=DATASET_NAME,
        evaluators=ALL,
        experiment_prefix=f"fleet-{args.variant}",
        max_concurrency=args.concurrency,
        metadata={"variant": args.variant},
    )
    name = getattr(result, "experiment_name", "")
    print(f"\nEXPERIMENT_NAME={name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
