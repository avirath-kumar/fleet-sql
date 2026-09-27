#!/usr/bin/env python3
"""Compare the experiments on our dataset, one column per variant."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/agent"))
sys.path.insert(0, str(ROOT))

import _env  # noqa: E402,F401
from langsmith import Client  # noqa: E402

from evals.dataset import DATASET_NAME  # noqa: E402


def main() -> int:
    client = Client()
    ds = client.read_dataset(dataset_name=DATASET_NAME)
    projects = sorted(client.list_projects(reference_dataset_id=ds.id),
                      key=lambda p: p.start_time or "", reverse=True)
    if not projects:
        print("no experiments yet — run scripts/experiment.sh")
        return 1

    # Newest experiment per variant, so a re-run replaces rather than clutters.
    latest: dict[str, object] = {}
    for p in projects:
        v = (p.metadata or {}).get("variant") or p.name.split("-")[1]
        latest.setdefault(v, p)

    # Aggregate from the RUNS, not from project.feedback_stats -- that field is
    # empty on an experiment project and reading it printed a table of zeros.
    stats: dict[str, dict[str, list[float]]] = {}
    examples: dict[str, int] = {}
    for v, proj in latest.items():
        per_key: dict[str, list[float]] = {}
        runs = list(client.list_runs(project_name=proj.name, is_root=True))
        examples[v] = len(runs)
        for r in runs:
            for key, fb in (r.feedback_stats or {}).items():
                if fb.get("avg") is not None:
                    per_key.setdefault(key, []).append(fb["avg"])
        stats[v] = per_key

    cols = sorted(latest)
    keys = sorted({k for s in stats.values() for k in s})
    width = 26

    print()
    print("  " + "metric".ljust(30) + "".join(f"{v:<{width}}" for v in cols))
    print("  " + "-" * (30 + width * len(cols)))
    for k in keys:
        row = "  " + k.ljust(30)
        for v in cols:
            vals = stats[v].get(k) or []
            row += (f"{sum(vals)/len(vals):.3f}  (n={len(vals)})".ljust(width)
                    if vals else "—".ljust(width))
        print(row)
    print("  " + "-" * (30 + width * len(cols)))
    overall = {}
    for v in cols:
        means = [sum(x)/len(x) for x in stats[v].values() if x]
        overall[v] = sum(means) / len(means) if means else 0.0
    print("  " + "OVERALL".ljust(30) + "".join(f"{overall[v]:.3f}".ljust(width) for v in cols))
    print()
    for v in cols:
        print(f"  {v}: {latest[v].name}  ({examples[v]} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
