"""The regression suite: questions where the answer is a number we can check.

Every example carries `truth_sql`. The evaluators run it and compare, so
correctness is decided by the database rather than by a judge's opinion -- which
matters here because the failure being measured is a model that produces
plausible wrong numbers, and a judge reading a plausible answer scores it well.

`needs_derivation` marks the questions the query catalogue cannot answer
directly. Those are the ones that separate the two builds: when a pre-aggregated
query exists, a naive agent calls it and is fine.
"""
from __future__ import annotations

import os

from langsmith import Client

DEMO_PRESENTER = os.getenv("DEMO_PRESENTER", "robert").strip() or "robert"
DATASET_NAME = f"fleet-sql-{DEMO_PRESENTER}"
CURATED = "dataset.py"

EXAMPLES: list[dict] = [
    # --- the catalogue answers these directly; both builds should pass -------
    {
        "question": "How many top-level configuration slots does the 747-8 have?",
        "truth_sql": "SELECT COUNT(*) FROM config_slots WHERE model='747-8' AND parent_slot_id IS NULL",
        "label": "747-8 top-level configuration slots",
        "category": "catalogued", "needs_derivation": False,
        "why": "the figure the prototype returned; a reviewer checks this one first",
    },
    {
        "question": "How many open deferrals does the 737-800 fleet have?",
        "truth_sql": "SELECT COUNT(*) FROM deferrals d JOIN aircraft a USING(tail_number) "
                     "WHERE a.model='737-800' AND d.status IN ('open','extended')",
        "label": "737-800 open deferrals",
        "category": "oversized", "needs_derivation": False,
        "why": "the oversized result, but the count alone is on the receipt",
    },
    {
        "question": "How many aircraft are in the 747-8 fleet?",
        "truth_sql": "SELECT COUNT(*) FROM aircraft WHERE model='747-8'",
        "label": "747-8 aircraft",
        "category": "catalogued", "needs_derivation": False,
        "why": "a small result; neither build should struggle",
    },
    # --- the catalogue has no pre-aggregated answer; these separate the builds
    {
        "question": "Which station carries the most open deferrals on the 737-800 fleet, "
                    "and how many does it have?",
        "truth_sql": "SELECT COUNT(*) FROM deferrals d JOIN aircraft a USING(tail_number) "
                     "WHERE a.model='737-800' AND d.status IN ('open','extended') "
                     "AND d.station_code='ORD'",
        "label": "open deferrals at the top station (ORD) on the 737-800 fleet",
        "category": "derived", "needs_derivation": True,
        "why": "no catalogue query groups by station; the naive build counts by hand and misses",
    },
    {
        "question": "On the 737-800 fleet, how many open deferrals are in ATA chapter 25?",
        "truth_sql": "SELECT COUNT(*) FROM deferrals d JOIN aircraft a USING(tail_number) "
                     "WHERE a.model='737-800' AND d.status IN ('open','extended') AND d.ata_chapter=25",
        "label": "737-800 open deferrals in ATA 25",
        "category": "derived", "needs_derivation": True,
        "why": "a slice of the oversized result",
    },
    {
        "question": "How many distinct 737-800 aircraft have at least one open category C deferral?",
        "truth_sql": "SELECT COUNT(DISTINCT d.tail_number) FROM deferrals d JOIN aircraft a USING(tail_number) "
                     "WHERE a.model='737-800' AND d.status IN ('open','extended') AND d.category='C'",
        "label": "737-800 aircraft with an open category C deferral",
        "category": "derived", "needs_derivation": True,
        "why": "a distinct count over the oversized result; counting by hand is hopeless",
    },
    {
        "question": "Across the whole fleet, how many open deferrals fall due within 7 days?",
        "truth_sql": "SELECT COUNT(*) FROM deferrals WHERE status IN ('open','extended') "
                     "AND julianday(due_date) - julianday('2026-09-27') BETWEEN 0 AND 7",
        "label": "fleet-wide open deferrals due within 7 days",
        "category": "oversized", "needs_derivation": False,
        "why": "a second oversized query, so the finding is not one query's quirk",
    },
    {
        "question": "How many open deferrals does the A320-200 fleet have in ATA chapter 33?",
        "truth_sql": "SELECT COUNT(*) FROM deferrals d JOIN aircraft a USING(tail_number) "
                     "WHERE a.model='A320-200' AND d.status IN ('open','extended') AND d.ata_chapter=33",
        "label": "A320-200 open deferrals in ATA 33",
        "category": "derived", "needs_derivation": True,
        "why": "a different fleet, so the result is not tuned to one model",
    },
]


def upsert(client: Client | None = None) -> str:
    """Create or refresh the examples. Replaces its own, leaves others alone."""
    from langsmith.schemas import DataType

    client = client or Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        ds = client.read_dataset(dataset_name=DATASET_NAME)
        for ex in client.list_examples(dataset_id=ds.id):
            if (ex.metadata or {}).get("source") == CURATED:
                client.delete_example(example_id=ex.id)
    else:
        ds = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Fleet SQL agent: questions whose answer is a number the "
                        "database can settle. Marks which ones the query catalogue "
                        "cannot answer without deriving.",
            data_type=DataType.kv,
        )
    client.create_examples(
        dataset_id=ds.id,
        inputs=[{"question": e["question"]} for e in EXAMPLES],
        outputs=[{"label": e["label"], "truth_sql": e["truth_sql"]} for e in EXAMPLES],
        metadata=[{"category": e["category"], "needs_derivation": e["needs_derivation"],
                   "why": e["why"], "source": CURATED} for e in EXAMPLES],
    )
    return str(ds.id)


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src/agent"))
    import _env  # noqa: F401
    print(f"{DATASET_NAME}: {upsert()}  ({len(EXAMPLES)} examples)")
