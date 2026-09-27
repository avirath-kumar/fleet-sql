"""The tools the agent actually calls.

Two variants of the same capability, selected by AGENT_VARIANT:

  v1  run_query returns every row inline. This is the naive shape, and the one
      the evals are meant to expose: correct answers, ruinous context, and a
      silent failure mode once the result no longer fits.
  v2  run_query returns a receipt and writes rows to a file; describe/filter/
      aggregate/page/export work on that file. Nothing that scales with row
      count ever reaches a message.

Both are here, in one module, because the diff between them is the argument.
"""
from __future__ import annotations

import json
import os
from typing import Annotated, Any, Literal

from langchain.tools import tool

from catalog import CATALOG
import results

VARIANT = os.environ.get("AGENT_VARIANT", "v2").lower()


class ToolInputError(Exception):
    """A tool call the model can fix by calling again.

    Raised, then converted to a returned payload by `_guard`, because a raised
    exception ends the run: the first v1 comparison died outright when the model
    omitted a required parameter, which is a correctable mistake and should cost
    a turn, not the answer.
    """


def _guard(fn):
    """Turn an input error into a result the model can read and retry from."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolInputError as exc:
            return {"error": str(exc), "retry": "Correct the arguments and call again."}
        except FileNotFoundError as exc:
            return {"error": str(exc), "retry": "Run the query again to get a fresh result id."}
    return wrapper


def _lookup(query_id: str):
    q = CATALOG.get(query_id)
    if q is None:
        raise ToolInputError(
            f"unknown query {query_id!r}. Approved queries: {', '.join(CATALOG)}"
        )
    return q


def _check_params(q, params: dict) -> dict:
    """Bind only declared parameters, and fail loudly on a missing one.

    Rejects rather than defaults: a missing `model` silently becoming every
    model is exactly how a 40-row question turns into a 1,700-row one.
    """
    params = params or {}
    expected = {p.name for p in q.params}
    missing = expected - set(params)
    if missing:
        raise ToolInputError(f"{q.id} needs parameter(s): {', '.join(sorted(missing))}")
    extra = set(params) - expected
    if extra:
        raise ToolInputError(f"{q.id} does not take: {', '.join(sorted(extra))}")
    return {k: params[k] for k in expected}


# --- v1: the naive tool -----------------------------------------------------

@tool(parse_docstring=True)
@_guard
def run_query_inline(query_id: str, params: dict | None = None) -> dict:
    """Run a pre-approved query and return all of its rows.

    Args:
        query_id: One of the approved query ids.
        params: Parameter values the query declares.
    """
    q = _lookup(query_id)
    bound = _check_params(q, params or {})
    receipt = results.materialize(q.id, bound, q.sql, q.facets, inline_under=10**9)
    return {"query_id": q.id, "params": bound,
            "row_count": receipt.row_count, "rows": receipt.rows}


# --- v2: receipt plus targeted tools ---------------------------------------

@tool(parse_docstring=True)
@_guard
def run_query(query_id: str, params: dict | None = None) -> dict:
    """Run a pre-approved query. Returns a receipt, not rows.

    The receipt carries the exact row count, the columns, a five-row preview and
    breakdowns of the low-cardinality columns. Rows are written to a file and
    reached through describe_result, filter_result, aggregate_result,
    page_result and export_report. A small result is returned inline.

    Args:
        query_id: One of the approved query ids.
        params: Parameter values the query declares.
    """
    q = _lookup(query_id)
    bound = _check_params(q, params or {})
    return results.materialize(q.id, bound, q.sql, q.facets).to_tool_result()


@tool(parse_docstring=True)
@_guard
def describe_result(result_id: str) -> dict:
    """Column-level shape of a stored result: types, distinct counts, nulls, ranges.

    Args:
        result_id: A result id from run_query or filter_result.
    """
    rows = results.load(result_id)
    if not rows:
        return {"result_id": result_id, "row_count": 0, "columns": {}}
    out: dict[str, Any] = {}
    for col in rows[0]:
        vals = [r.get(col) for r in rows]
        non_null = [v for v in vals if v is not None]
        info: dict[str, Any] = {
            "distinct": len(set(map(str, non_null))),
            "nulls": len(vals) - len(non_null),
        }
        if non_null and all(isinstance(v, (int, float)) for v in non_null):
            info["min"], info["max"] = min(non_null), max(non_null)
        elif non_null:
            ordered = sorted(map(str, non_null))
            info["min"], info["max"] = ordered[0], ordered[-1]
        out[col] = info
    return {"result_id": result_id, "row_count": len(rows), "columns": out}


@tool(parse_docstring=True)
@_guard
def filter_result(result_id: str, column: str,
                  op: Literal["eq", "ne", "in", "lt", "lte", "gt", "gte", "contains"],
                  value: str) -> dict:
    """Narrow a stored result. Returns a NEW receipt, not rows.

    Chainable: filter a filtered result to narrow again. For `in`, give a
    comma-separated list.

    Args:
        result_id: The result to narrow.
        column: Column to test.
        op: Comparison to apply.
        value: Value to compare against; comma-separated for `in`.
    """
    rows = results.load(result_id)
    if rows and column not in rows[0]:
        raise ToolInputError(f"no column {column!r}. Available: {', '.join(rows[0])}")

    def keep(r: dict) -> bool:
        cell = r.get(column)
        s = "" if cell is None else str(cell)
        if op == "eq":       return s == value
        if op == "ne":       return s != value
        if op == "in":       return s in {v.strip() for v in value.split(",")}
        if op == "contains": return value.lower() in s.lower()
        try:                 a, b = float(cell), float(value)      # numeric compare
        except (TypeError, ValueError): a, b = s, value            # else lexicographic
        return {"lt": a < b, "lte": a <= b, "gt": a > b, "gte": a >= b}[op]

    kept = [r for r in rows if keep(r)]
    facets = tuple(k for k in (rows[0] if rows else {}) if k != column)
    receipt = results.store(kept, f"filter({result_id})",
                            {"column": column, "op": op, "value": value}, facets[:4],
                            parent_id=result_id,
                            derived_by=f"{column} {op} {value}")
    return receipt.to_tool_result()


@tool(parse_docstring=True)
@_guard
def aggregate_result(result_id: str, group_by: str,
                     metric: Literal["count", "count_distinct"] = "count",
                     distinct_column: str | None = None) -> dict:
    """Group a stored result and count. Returns a small table whatever the input size.

    Args:
        result_id: The result to aggregate.
        group_by: Column to group on.
        metric: count of rows, or count_distinct of another column.
        distinct_column: Column to count distinctly when metric is count_distinct.
    """
    rows = results.load(result_id)
    if rows and group_by not in rows[0]:
        raise ToolInputError(f"no column {group_by!r}. Available: {', '.join(rows[0])}")
    buckets: dict[str, Any] = {}
    for r in rows:
        key = str(r.get(group_by))
        if metric == "count":
            buckets[key] = buckets.get(key, 0) + 1
        else:
            if not distinct_column:
                raise ToolInputError("count_distinct needs distinct_column")
            buckets.setdefault(key, set()).add(str(r.get(distinct_column)))
    table = {k: (v if isinstance(v, int) else len(v)) for k, v in buckets.items()}
    return {"result_id": result_id, "group_by": group_by, "metric": metric,
            "groups": dict(sorted(table.items(), key=lambda kv: -kv[1])),
            "total_rows": len(rows)}


@tool(parse_docstring=True)
@_guard
def page_result(result_id: str, offset: int = 0, limit: int = 20,
                columns: str | None = None) -> dict:
    """Read a bounded window of rows from a stored result.

    Never returns more than 50 rows in one call, whatever `limit` says. Ask for
    specific `columns` when you only need a few.

    Args:
        result_id: The result to read.
        offset: Zero-based row to start at.
        limit: Rows to return; capped at 50.
        columns: Comma-separated column subset.
    """
    rows = results.load(result_id)
    limit = max(1, min(int(limit), results.MAX_PAGE))
    window = rows[offset: offset + limit]
    if columns:
        keep = [c.strip() for c in columns.split(",")]
        window = [{k: r.get(k) for k in keep} for r in window]
    return {"result_id": result_id, "offset": offset, "returned": len(window),
            "row_count": len(rows),
            "has_more": offset + len(window) < len(rows), "rows": window}


@tool(parse_docstring=True)
@_guard
def export_report(result_id: str, title: str) -> dict:
    """Write a stored result to a CSV the user can open, and return its path.

    The honest answer to "there are 511 of these": hand over a file rather than
    read them aloud.

    Args:
        result_id: The result to export.
        title: Human-readable report title.
    """
    import csv
    rows = results.load(result_id)
    path = results.RESULTS_DIR / f"{result_id}.csv"
    if rows:
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return {"title": title, "row_count": len(rows),
            "path": str(path.relative_to(results.ROOT)),
            "note": "Full result written to file; cite the path in your answer."}


@tool(parse_docstring=True)
@_guard
def list_results() -> dict:
    """Every result produced so far, with its size, origin and lineage.

    Use this when a question refers to something already run -- "narrow that to
    ORD", "go back to the full set" -- instead of re-running the query. A
    result id from an earlier turn is still valid; the rows are on disk.
    """
    index = results.known_results()
    return {"count": len(index),
            "results": [{"result_id": rid, **info} for rid, info in index.items()]}


@tool(parse_docstring=True)
@_guard
def widen_result(result_id: str) -> dict:
    """Step back to what a filtered result was derived from.

    The inverse of filter_result. Narrowing too far is the common mistake and
    without this the only way back is to re-run the original query, which the
    agent often does with different parameters.

    Args:
        result_id: A result produced by filter_result.
    """
    index = results.known_results()
    info = index.get(result_id)
    if info is None:
        raise ToolInputError(f"no such result: {result_id}")
    parent = info.get("parent_id")
    if not parent:
        return {"result_id": result_id, "note": "already the original result; nothing to widen to",
                "row_count": info.get("row_count")}
    pinfo = index.get(parent, {})
    return {"result_id": parent, "row_count": pinfo.get("row_count"),
            "query_id": pinfo.get("query_id"), "params": pinfo.get("params"),
            "undid": info.get("derived_by"),
            "note": f"stepped back from {result_id}"}


V1_TOOLS = [run_query_inline]
V2_TOOLS = [run_query, describe_result, filter_result, aggregate_result,
            page_result, export_report, list_results, widen_result]
TOOLS = V1_TOOLS if VARIANT == "v1" else V2_TOOLS
