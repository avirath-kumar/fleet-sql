"""Where query rows go instead of into the conversation.

The rule this module exists to enforce: **rows are never a tool's return
value.** `run_query` writes them to a file and hands back a receipt — counts,
columns, a short preview and facet breakdowns — which is a few hundred bytes
whatever the query returned. Everything afterwards works on the file.

Why a receipt rather than a truncated list: truncation answers "here are the
first 20 of something" and loses the number, which is usually the thing that
was actually asked. A receipt keeps the number exact and defers the rows.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
DB_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "fleet.db"
RESULTS_DIR = ROOT / ".results"

#: Rows in a receipt's preview. Small on purpose: the preview shows the SHAPE
#: of a result, not its content. Anything that needs content pages or filters.
PREVIEW_ROWS = 5
#: Hardest limit in the system. `page_result` will not return more than this in
#: one call however politely it is asked, because a tool that can be talked into
#: returning 500 rows is the bug this demo is about.
MAX_PAGE = 50
#: Facet values reported per column. A column with more distinct values than
#: this is summarised by its cardinality instead, so a receipt cannot grow with
#: the data.
MAX_FACET_VALUES = 8


#: A registry of every result this session produced, so a handle outlives the
#: message that carried it. Without this a receipt is only as durable as the
#: conversation: compact the history and the id is gone while the file is still
#: on disk, orphaned.
INDEX_PATH = RESULTS_DIR / "index.json"


@dataclass
class Receipt:
    """What a query returns instead of its rows."""
    result_id: str
    query_id: str
    params: dict[str, Any]
    row_count: int
    columns: list[str]
    #: First few rows, for shape only.
    preview: list[dict]
    #: column -> {value: count} for low-cardinality columns; the useful half of
    #: a 500-row result is almost always one of these breakdowns.
    facets: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Where the rows actually are, so the agent's own file tools can reach them.
    path: str = ""
    #: Set when the result is small enough that withholding it would be silly.
    rows: list[dict] | None = None
    #: Lineage. A filtered result records what it came from and how, so the
    #: agent can widen again instead of re-running the query -- and so "undo
    #: that last filter" is a thing it can actually do.
    parent_id: str | None = None
    derived_by: str | None = None
    #: What you can still do to this result: column -> the values present.
    #: Named as an affordance rather than left implicit in `facets`, because a
    #: receipt that only says "511 rows" tells the agent the job is done. This
    #: says which adjustments are available without reading a row.
    refinable: dict[str, list[str]] = field(default_factory=dict)

    def to_tool_result(self) -> dict:
        d = asdict(self)
        if self.rows is None:
            d.pop("rows")
        return d


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _refinable(rows: list[dict], wanted: tuple[str, ...]) -> dict[str, list[str]]:
    """Columns worth narrowing on, and the values actually present.

    Bounded by MAX_FACET_VALUES: a column with more distinct values than that
    is omitted rather than truncated, because a half-listed set of options
    invites the agent to assume the ones it cannot see do not exist.
    """
    out: dict[str, list[str]] = {}
    for col in wanted:
        vals = sorted({str(r.get(col)) for r in rows if r.get(col) is not None})
        if 1 < len(vals) <= MAX_FACET_VALUES:
            out[col] = vals
    return out


def _record(receipt: "Receipt") -> None:
    """Append a result to the session registry."""
    import json as _json
    index = {}
    if INDEX_PATH.exists():
        try:
            index = _json.loads(INDEX_PATH.read_text())
        except Exception:
            index = {}
    index[receipt.result_id] = {
        "query_id": receipt.query_id, "params": receipt.params,
        "row_count": receipt.row_count, "columns": receipt.columns,
        "parent_id": receipt.parent_id, "derived_by": receipt.derived_by,
    }
    INDEX_PATH.write_text(_json.dumps(index, indent=1, default=str))


def known_results() -> dict:
    """Everything produced this session, newest last."""
    import json as _json
    if not INDEX_PATH.exists():
        return {}
    try:
        return _json.loads(INDEX_PATH.read_text())
    except Exception:
        return {}


def _facets(rows: list[dict], columns: list[str], wanted: tuple[str, ...]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for col in wanted:
        if col not in columns:
            continue
        counts: dict[str, int] = {}
        for r in rows:
            key = str(r.get(col))
            counts[key] = counts.get(key, 0) + 1
        if len(counts) <= MAX_FACET_VALUES:
            out[col] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
        else:
            # Cardinality plus the top few, so the receipt size is bounded by
            # MAX_FACET_VALUES rather than by how varied the data happens to be.
            top = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:MAX_FACET_VALUES])
            out[col] = {"__distinct__": len(counts), **top}
    return out


def materialize(query_id: str, params: dict, sql: str, facets: tuple[str, ...],
                inline_under: int = 25) -> Receipt:
    """Run the SQL, write the rows to disk, return the receipt.

    `inline_under` is the one concession to ergonomics: a result of a handful of
    rows is returned inline, because forcing a second tool call to read six rows
    is friction with no benefit. The threshold is small and fixed.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    con = _connect()
    try:
        cur = con.execute(sql, params)
        columns = [c[0] for c in cur.description]
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

    result_id = f"res_{uuid.uuid4().hex[:10]}"
    path = RESULTS_DIR / f"{result_id}.jsonl"
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")

    receipt = Receipt(
        result_id=result_id,
        query_id=query_id,
        params=params,
        row_count=len(rows),
        columns=columns,
        preview=rows[:PREVIEW_ROWS],
        facets=_facets(rows, columns, facets),
        path=str(path.relative_to(ROOT)),
        rows=rows if len(rows) <= inline_under else None,
        refinable=_refinable(rows, facets),
    )
    _record(receipt)
    return receipt


def load(result_id: str) -> list[dict]:
    """Every row of a stored result. In-process only — never a tool return."""
    path = RESULTS_DIR / f"{result_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no such result: {result_id}")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def store(rows: list[dict], query_id: str, params: dict,
          facets: tuple[str, ...] = (), parent_id: str | None = None,
          derived_by: str | None = None) -> Receipt:
    """Persist rows that were derived in-process (a filter, a sort) as a result
    in their own right, so a narrowed set can be narrowed again."""
    RESULTS_DIR.mkdir(exist_ok=True)
    result_id = f"res_{uuid.uuid4().hex[:10]}"
    path = RESULTS_DIR / f"{result_id}.jsonl"
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    columns = list(rows[0].keys()) if rows else []
    receipt = Receipt(
        result_id=result_id, query_id=query_id, params=params,
        row_count=len(rows), columns=columns, preview=rows[:PREVIEW_ROWS],
        facets=_facets(rows, columns, facets), path=str(path.relative_to(ROOT)),
        rows=rows if len(rows) <= 25 else None,
        parent_id=parent_id, derived_by=derived_by,
        refinable=_refinable(rows, facets),
    )
    _record(receipt)
    return receipt
