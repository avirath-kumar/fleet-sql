"""Where query rows live instead of in the conversation.

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
import pathlib
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "fleet.db"
DIR = ROOT / ".results"
INDEX = DIR / "index.json"

PREVIEW_ROWS = 5        # enough to show shape, never enough to read
MAX_PAGE = 50           # the hardest limit here; page_result will not exceed it
MAX_INLINE = 25         # below this, withholding rows is friction with no gain
MAX_VALUES = 8          # a column with more distinct values is not a facet


@dataclass
class Result:
    """What a query returns instead of its rows."""
    result_id: str
    query: str
    params: dict
    row_count: int
    columns: list[str]
    preview: list[dict]
    path: str
    #: column -> {value: count}, for columns narrow enough to be worth it.
    #: Doubles as the menu of available refinements: the keys are what you can
    #: filter on, the inner keys are the values actually present. A receipt
    #: without this reads as "job done" and leaves the agent nothing to offer.
    breakdown: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Present only for a small result.
    rows: list[dict] | None = None
    #: Lineage, so a filter is reversible and "go back" does not mean re-query.
    parent_id: str | None = None
    derived_by: str | None = None

    def payload(self) -> dict:
        d = asdict(self)
        if self.rows is None:
            d.pop("rows")
        return d


def _breakdown(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Count every column narrow enough to be a facet.

    Derived from the DATA, not from a per-query declaration. The catalogue used
    to name its own facet columns, which meant a column nobody thought to list
    was invisible as a refinement even when it had four distinct values.
    """
    out: dict[str, dict[str, int]] = {}
    for col in (rows[0] if rows else {}):
        counts: dict[str, int] = {}
        for r in rows:
            key = str(r.get(col))
            counts[key] = counts.get(key, 0) + 1
            if len(counts) > MAX_VALUES:
                break
        if 1 < len(counts) <= MAX_VALUES:
            out[col] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    return out


def query(sql: str, params: dict) -> tuple[list[str], list[dict]]:
    """Run approved SQL. The only place this module touches the database."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(sql, params)
        return [c[0] for c in cur.description], [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def save(rows: list[dict], query_name: str, params: dict, columns: list[str] | None = None,
         parent_id: str | None = None, derived_by: str | None = None,
         inline_under: int = MAX_INLINE) -> Result:
    """Write rows to a file and return the receipt.

    One function for both a fresh query and a derived set, because they differ
    only in lineage -- and two near-identical writers is how the two drift.
    """
    DIR.mkdir(exist_ok=True)
    result_id = f"res_{uuid.uuid4().hex[:10]}"
    path = DIR / f"{result_id}.jsonl"
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")

    result = Result(
        result_id=result_id, query=query_name, params=params,
        row_count=len(rows),
        columns=columns or (list(rows[0]) if rows else []),
        preview=rows[:PREVIEW_ROWS], path=str(path.relative_to(ROOT)),
        breakdown=_breakdown(rows),
        rows=rows if len(rows) <= inline_under else None,
        parent_id=parent_id, derived_by=derived_by,
    )
    _index(result)
    return result


def load(result_id: str) -> list[dict]:
    """Every row of a stored result. In-process only -- never a tool's return."""
    path = DIR / f"{result_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no such result: {result_id}")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _index(result: Result) -> None:
    """Record a result so its id outlives the message that carried it.

    A file, not memory: compact the conversation and the id is gone from the
    history while the rows are still on disk, orphaned.
    """
    known_results = known()
    known_results[result.result_id] = {
        "query": result.query, "params": result.params,
        "row_count": result.row_count, "parent_id": result.parent_id,
        "derived_by": result.derived_by,
    }
    INDEX.write_text(json.dumps(known_results, indent=1, default=str))


def known() -> dict:
    if not INDEX.exists():
        return {}
    try:
        return json.loads(INDEX.read_text())
    except json.JSONDecodeError:
        return {}
