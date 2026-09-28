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

def make_target(variant: str):
    """The function under test. A pass-through, because the agent's own state
    carries what the evaluators score -- see middleware/run_cost.py."""
    from agent import build_agent

    agent = build_agent(variant)

    def target(inputs: dict) -> dict:
        out = agent.invoke({"messages": [{"role": "user", "content": inputs["question"]}]})
        answer = out.get("structured_response")
        # `messages` first: LangSmith renders a Messages view when outputs carry
        # it, and a dict without it renders as generic Fields. readable_answer
        # has already made the last message prose rather than the JSON blob.
        return {
            "messages": out.get("messages", []),
            "structured_response": answer.model_dump() if answer is not None else {},
            **{k: out.get(k, 0) for k in
               ("tool_payload_chars", "total_tokens", "harness_evicted_results", "tool_calls")},
        }

    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default=os.environ.get("AGENT_VARIANT", "v2"),
                    choices=["v1", "v2", "v2-regressed"])
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--skip-upsert", action="store_true")
    #: Repetitions matter here: the naive build is non-deterministically wrong
    #: (118 on one run, 103 on another), so a single pass cannot tell a real
    #: failure from a coin flip.
    ap.add_argument("--repetitions", type=int, default=1)
    args = ap.parse_args()

    # results.py reads AGENT_VARIANT at import time, and make_target imports
    # the agent, so this has to be set before either happens.
    os.environ["AGENT_VARIANT"] = args.variant
    if not args.skip_upsert:
        upsert()
    client = Client()
    result = client.evaluate(
        make_target(args.variant),
        data=DATASET_NAME,
        evaluators=ALL,
        experiment_prefix=f"fleet-{args.variant}",
        max_concurrency=args.concurrency,
        num_repetitions=args.repetitions,
        metadata={"variant": args.variant, "repetitions": args.repetitions},
    )
    name = getattr(result, "experiment_name", "")
    print(f"\nEXPERIMENT_NAME={name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
