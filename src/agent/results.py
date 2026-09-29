"""Persist query rows and return bounded result receipts."""
from __future__ import annotations

import json
import pathlib
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "fleet.db"
DIR = ROOT / ".results"
INDEX = DIR / "index.json"

PREVIEW_ROWS = 5
MAX_PAGE = 50
MAX_INLINE = 25


@dataclass
class Result:
    result_id: str
    query_id: str
    params: dict
    row_count: int
    columns: list[str]
    preview: list[dict]
    full_rows: list[dict] = field(default_factory=list, repr=False)

    def receipt(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "full_rows"}


def query(sql: str, params: dict) -> tuple[list[str], list[dict]]:
    """Run approved SQL and return its columns and rows."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(sql, params)
        return [column[0] for column in cur.description], [dict(row) for row in cur.fetchall()]
    finally:
        con.close()


def write(result: Result) -> pathlib.Path:
    """Persist all rows and update the result index."""
    DIR.mkdir(parents=True, exist_ok=True)
    path = DIR / f"{result.result_id}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in result.full_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    index = {}
    if INDEX.exists():
        index = json.loads(INDEX.read_text(encoding="utf-8"))
    index[result.result_id] = result.receipt()
    INDEX.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    return path


def read(result_id: str) -> list[dict]:
    """Read all rows for a stored result."""
    path = DIR / f"{result_id}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def receipt_from_rows(query_id: str, params: dict, columns: list[str], rows: list[dict]) -> Result:
    """Build a result receipt from query rows."""
    return Result(
        result_id=uuid.uuid4().hex,
        query_id=query_id,
        params=params,
        row_count=len(rows),
        columns=columns,
        preview=rows[:PREVIEW_ROWS],
        full_rows=rows,
    )
