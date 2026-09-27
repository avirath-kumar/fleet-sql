#!/usr/bin/env python3
"""Per-example pass rate across repetitions, for both variants.

The aggregate table says v1 is worse. This says whether it is RELIABLY worse,
which is a different claim and the one worth making: the naive build is
non-deterministically wrong, so a single pass cannot separate a real failure
from a coin flip.
"""
from __future__ import annotations

import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/agent"))
sys.path.insert(0, str(ROOT))

import _env  # noqa: E402,F401
from langsmith import Client  # noqa: E402

from evals.dataset import DATASET_NAME  # noqa: E402

KEY = "answer_is_correct"


def main() -> int:
    client = Client()
    ds = client.read_dataset(dataset_name=DATASET_NAME)
    projects = sorted(client.list_projects(reference_dataset_id=ds.id),
                      key=lambda p: p.start_time or "", reverse=True)
    latest: dict[str, object] = {}
    for p in projects:
        meta = p.metadata or {}
        if int(meta.get("repetitions") or 1) < 2:
            continue                      # single-pass runs cannot answer this
        v = meta.get("variant") or p.name.split("-")[1]
        latest.setdefault(v, p)
    if not latest:
        print("no multi-repetition experiments yet — "
              "run evals/run_experiment.py --repetitions 3")
        return 1

    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for v, proj in latest.items():
        for r in client.list_runs(project_name=proj.name, is_root=True):
            q = str((r.inputs or {}).get("question") or "")[:52]
            fb = (r.feedback_stats or {}).get(KEY) or {}
            if fb.get("avg") is not None:
                scores[v][q].append(fb["avg"])

    cols = sorted(latest)
    questions = sorted({q for v in cols for q in scores[v]})
    print(f"\n  {KEY}, per example, across repetitions\n")
    print("  " + "question".ljust(54) + "".join(f"{v:<12}" for v in cols))
    print("  " + "-" * (54 + 12 * len(cols)))
    for q in questions:
        row = "  " + q.ljust(54)
        for v in cols:
            vals = scores[v].get(q) or []
            row += (f"{int(sum(vals))}/{len(vals)}".ljust(12) if vals else "—".ljust(12))
        print(row)
    print("  " + "-" * (54 + 12 * len(cols)))
    row = "  " + "TOTAL".ljust(54)
    for v in cols:
        allv = [x for vals in scores[v].values() for x in vals]
        row += f"{int(sum(allv))}/{len(allv)}".ljust(12)
    print(row + "\n")
    for v in cols:
        print(f"  {v}: {latest[v].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
