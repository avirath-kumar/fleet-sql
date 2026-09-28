#!/usr/bin/env python3
"""Build every LangSmith asset this demo needs, idempotently.

    scripts/setup_langsmith.py                 everything
    scripts/setup_langsmith.py --only dataset
    scripts/setup_langsmith.py --promote 5     add failing traces to the dataset

Sections: project, dataset, evaluators, queue, experiments.

ONLINE EVALUATORS ARE NOT THE OFFLINE ONES. A LangSmith online evaluator runs
sandboxed against a single run's inputs and outputs, with no network and no
filesystem -- so `answer_is_correct`, which settles a number by querying the
database, cannot run there. What CAN run online is the structural half: did an
oversized tool result spill, and how big was the tool payload. Those two are
also the cheap ones, so live traffic gets the check that costs nothing and the
dataset gets the check that needs the database.
"""
from __future__ import annotations

import argparse
import os
import time
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/agent"))
sys.path.insert(0, str(ROOT))

import _env  # noqa: E402,F401
import requests  # noqa: E402
from langsmith import Client  # noqa: E402

from evals.dataset import DATASET_NAME, EXAMPLES, upsert  # noqa: E402

PROJECT = os.environ.get("LANGSMITH_PROJECT", "fleet-sql")
QUEUE = f"fleet-sql-review-{os.environ.get('DEMO_PRESENTER', 'robert')}"
API = "https://api.smith.langchain.com"
OWNED_PREFIX = "fleet-sql-"

#: Structural checks, cheap and decidable from one run. Written as source
#: because LangSmith runs them in its own sandbox.
#:
#: `perform_eval(run)` takes EXACTLY ONE positional argument. The offline
#: signature is (run, example) and copying it here fails the upload with
#: "Function should take exactly 1 positional arguments." -- an online
#: evaluator has no example to compare against.
ONLINE_EVALUATORS = {
    "no_silent_truncation": '''
def perform_eval(run):
    """1 when no tool result was too big to inline."""
    out = run.get("outputs") or {}
    evicted = out.get("harness_evicted_results")
    if evicted is None:
        return {"no_silent_truncation": None}
    return {"no_silent_truncation": 0.0 if evicted else 1.0}
''',
    "rows_stayed_out_of_context": '''
def perform_eval(run):
    """1 when tool output stayed under the context budget."""
    out = run.get("outputs") or {}
    chars = out.get("tool_payload_chars")
    if chars is None:
        return {"rows_stayed_out_of_context": None}
    return {"rows_stayed_out_of_context": 1.0 if chars <= 20000 else 0.0}
''',
}


def _headers(json_body: bool = False) -> dict:
    h = {"x-api-key": os.environ["LANGSMITH_API_KEY"]}
    ws = os.environ.get("LANGSMITH_WORKSPACE_ID", "").strip()
    if ws:
        h["X-Tenant-Id"] = ws
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def setup_project(client: Client) -> str:
    try:
        p = client.read_project(project_name=PROJECT)
    except Exception:
        p = client.create_project(project_name=PROJECT,
                                  description="Fleet SQL agent traffic.")
    print(f"  project '{PROJECT}' ({p.id})")
    return str(p.id)


def setup_dataset() -> None:
    print(f"  dataset '{DATASET_NAME}' ({upsert()})  {len(EXAMPLES)} curated examples")


def setup_queue(client: Client, project_id: str) -> None:
    """An annotation queue, so a human can label what the code cannot."""
    existing = {q.name: q for q in client.list_annotation_queues()}
    q = existing.get(QUEUE) or client.create_annotation_queue(
        name=QUEUE, description="Fleet SQL answers to eyeball before promoting.")
    print(f"  queue '{QUEUE}' ({q.id})")


def setup_evaluators(client: Client, project_id: str) -> None:
    """Attach the structural checks to live traffic, replacing our own first."""
    listed = requests.get(f"{API}/api/v1/runs/rules", headers=_headers(), timeout=60)
    removed = 0
    if listed.status_code == 200:
        for rule in listed.json():
            if str(rule.get("session_id")) == project_id and \
               rule.get("display_name") in ONLINE_EVALUATORS:
                r = requests.delete(f"{API}/api/v1/runs/rules/{rule['id']}",
                                    headers=_headers(), timeout=60)
                removed += r.status_code in (200, 204)
    if removed:
        print(f"  removed {removed} existing rule(s)")
    made = 0
    for name, source in ONLINE_EVALUATORS.items():
        body = {
            "display_name": name,
            "session_id": project_id,
            "is_enabled": True,
            "sampling_rate": 1.0,
            "filter": "eq(is_root, true)",
            "code_evaluators": [{"name": name, "code": source.strip()}],
        }
        r = requests.post(f"{API}/api/v1/runs/rules", headers=_headers(True),
                          json=body, timeout=60)
        if r.status_code in (200, 201):
            made += 1
        else:
            print(f"  ! {name}: {r.status_code} {r.text[:140]}")
    print(f"  {made} online evaluator(s) scoring new traces")


def promote_traces(client: Client, limit: int) -> None:
    """Add failing production traces to the dataset.

    THE STEP THE LOOP TURNS ON. A trace that scored badly is a question the
    suite did not have; promoting it is how the dataset grows out of real usage
    rather than out of someone's imagination. Curated examples carry a `source`
    of dataset.py and are replaced on every upsert -- these carry `promoted`,
    so upsert leaves them alone.
    """
    ds = client.read_dataset(dataset_name=DATASET_NAME)
    have = {(ex.inputs or {}).get("question") for ex in client.list_examples(dataset_id=ds.id)}
    added = 0
    # limit=100 is the API maximum -- 200 fails with
    # "Limit exceeds maximum allowed value of 100", which is a 400, not a
    # truncated list, so it takes the whole call down.
    for run in client.list_runs(project_name=PROJECT, is_root=True, limit=100):
        if added >= limit:
            break
        q = (run.inputs or {}).get("question")
        if not q or q in have:
            continue
        stats = run.feedback_stats or {}
        failing = any((s or {}).get("avg") == 0.0 for s in stats.values())
        if not failing:
            continue
        # A promoted example arrives WITHOUT ground truth, and that is the
        # honest state: production knows the answer was suspect, not what the
        # right answer was. `answer_is_correct` abstains until a human writes
        # the truth_sql, which is exactly the review step the annotation queue
        # exists for. Scoring it 0 instead would invent a failure.
        client.create_examples(
            dataset_id=ds.id, inputs=[{"question": q}],
            outputs=[{"label": f"promoted from run {str(run.id)[:8]}",
                      "truth_sql": ""}],
            metadata=[{"source": "promoted", "run_id": str(run.id),
                       "needs_truth_sql": True,
                       "why": "scored 0 on a structural check in production"}])
        have.add(q)
        added += 1
        print(f"    + {q[:70]}")
    print(f"  promoted {added} failing trace(s) into '{DATASET_NAME}'")


def prune_experiments(client: Client, keep: int) -> None:
    """Delete old experiments on our dataset, keeping the newest `keep` per arm.

    Iterating on a demo leaves a lot of these -- 21 after a day, most of them
    a half-finished idea -- and the comparison view is unreadable with every
    attempt in it. Kept per ARM rather than overall, so pruning to one cannot
    leave two runs of the same variant and none of another.
    """
    from collections import defaultdict

    ds = client.read_dataset(dataset_name=DATASET_NAME)
    by_arm: dict[str, list] = defaultdict(list)
    for proj in sorted(client.list_projects(reference_dataset_id=ds.id),
                       key=lambda p: p.start_time or "", reverse=True):
        by_arm[(proj.metadata or {}).get("variant") or "?"].append(proj)

    removed = kept = 0
    for arm, projects in sorted(by_arm.items()):
        for i, proj in enumerate(projects):
            if i < keep:
                kept += 1
                continue
            # Deletes are rate limited per session: clearing 21 experiments
            # got 429s after the first ten. Back off rather than losing the
            # rest, which left the view half-pruned and needing another pass.
            for attempt in range(5):
                try:
                    client.delete_project(project_id=proj.id)
                    removed += 1
                    break
                except Exception as exc:  # noqa: BLE001
                    if "429" not in str(exc) and "Rate limit" not in str(exc):
                        print(f"  ! {proj.name}: {str(exc)[:90]}")
                        break
                    time.sleep(2 ** attempt)
            else:
                print(f"  ! {proj.name}: still rate limited after 5 tries")
    print(f"  {removed} removed, {kept} kept ({len(by_arm)} arm(s), keeping {keep} each)")


SECTIONS = ("project", "dataset", "evaluators", "queue")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=SECTIONS, default=None)
    ap.add_argument("--promote", nargs="?", const=5, type=int, default=None,
                    metavar="N", help="add up to N failing traces to the dataset")
    ap.add_argument("--prune-experiments", nargs="?", const=1, type=int, default=None,
                    metavar="KEEP",
                    help="delete old experiments, keeping the newest KEEP per arm "
                         "(default 1; 0 clears them all)")
    args = ap.parse_args()
    client = Client()

    if args.prune_experiments is not None:
        print("\n[prune experiments]")
        prune_experiments(client, args.prune_experiments)
        return 0

    if args.promote is not None:
        print("\n[promote]")
        promote_traces(client, args.promote)
        return 0

    wanted = args.only or list(SECTIONS)
    project_id = None
    if "project" in wanted or "evaluators" in wanted or "queue" in wanted:
        print("\n[project]")
        project_id = setup_project(client)
    if "dataset" in wanted:
        print("\n[dataset]")
        setup_dataset()
    if "evaluators" in wanted:
        print("\n[online evaluators]")
        setup_evaluators(client, project_id)
    if "queue" in wanted:
        print("\n[annotation queue]")
        setup_queue(client, project_id)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
