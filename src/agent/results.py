"""Running the approved SQL.

One rule: **rows are never a tool's return value.** A query writes its rows to
a file and hands back a `Result` -- counts, columns, a short preview and a
breakdown of the narrow columns -- which is a couple of kilobytes whatever the
query returned. Every later tool works on the file.

A receipt rather than a truncated list, because truncation answers "here are
the first 20 of something" and loses the number, which is usually the thing
being asked.
"""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "fleet.db"
DIR = ROOT / ".results"
INDEX = DIR / "index.json"

#: The regression arm, kept because it is a true story rather than a contrived
#: one. Cleaning up the receipt once collapsed two column summaries into one
#: and dropped high-cardinality columns entirely -- and with them the
#: cardinality. "How many distinct aircraft are affected" is exactly that
#: number, so v2 quietly lost an answer it used to get. Nothing in review
#: caught it; re-running the evals did.
OMIT_DISTINCT = os.environ.get("AGENT_VARIANT", "").lower() == "v2-regressed"

PREVIEW_ROWS = 5        # enough to show shape, never enough to read
MAX_PAGE = 50           # the hardest limit here; page_result will not exceed it
MAX_INLINE = 25         # below this, withholding rows is friction with no gain
MAX_VALUES = 8          # a column with more distinct values is not a facet


def query(sql: str, params: dict) -> tuple[list[str], list[dict]]:
    """Run approved SQL. The only place this module touches the database."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(sql, params)
        return [c[0] for c in cur.description], [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


