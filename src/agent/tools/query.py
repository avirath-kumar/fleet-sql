"""Bounded tools for approved queries and stored results."""
from __future__ import annotations

import functools
import json
from typing import Any

from langchain.tools import tool

import results
from catalog import CATALOG


class ToolInputError(Exception):
    """A tool call the model can fix by calling again."""


def _guard(fn):
    """Turn input and missing-result errors into retryable payloads."""
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
    query = CATALOG.get(query_id)
    if query is None:
        raise ToolInputError(
            f"unknown query {query_id!r}. Approved queries: {', '.join(CATALOG)}"
        )
    return query


def _check_params(query, params: dict) -> dict:
    """Bind only declared parameters and reject missing or extra values."""
    params = params or {}
    expected = {parameter.name for parameter in query.params}
    missing = expected - set(params)
    if missing:
        raise ToolInputError(f"{query.id} needs parameter(s): {', '.join(sorted(missing))}")
    extra = set(params) - expected
    if extra:
        raise ToolInputError(f"{query.id} does not take: {', '.join(sorted(extra))}")
    return {key: params[key] for key in expected}


def _stored_result(result_id: str) -> tuple[dict, list[dict]]:
    try:
        index = json.loads(results.INDEX.read_text(encoding="utf-8"))
        receipt = index[result_id]
    except (FileNotFoundError, KeyError):
        raise FileNotFoundError(f"result {result_id!r} was not found")
    return receipt, results.read(result_id)


@tool(parse_docstring=True)
@_guard
def run_query_inline(query_id: str, params: dict | None = None) -> dict:
    """Run a pre-approved query; large results return a receipt instead of rows.

    Args:
        query_id: One of the approved query ids.
        params: Parameter values the query declares.
    """
    query = _lookup(query_id)
    bound = _check_params(query, params or {})
    columns, rows = results.query(query.sql, bound)
    if len(rows) <= results.MAX_INLINE:
        return {"query_id": query.id, "params": bound, "row_count": len(rows), "columns": columns,
                "rows": rows}
    result = results.receipt_from_rows(query.id, bound, columns, rows)
    results.write(result)
    return result.receipt()


@tool(parse_docstring=True)
@_guard
def filter_result(result_id: str, column: str, value: Any) -> dict:
    """Store and return a receipt for rows whose column equals a value.

    Args:
        result_id: Stored result to filter.
        column: Column whose value must match.
        value: Value to compare against.
    """
    receipt, rows = _stored_result(result_id)
    if column not in receipt["columns"]:
        raise ToolInputError(f"unknown result column {column!r}")
    filtered = [row for row in rows if row.get(column) == value]
    result = results.receipt_from_rows(receipt["query_id"], receipt["params"], receipt["columns"], filtered)
    results.write(result)
    return result.receipt()


@tool(parse_docstring=True)
@_guard
def aggregate_result(result_id: str, group_by: str) -> dict:
    """Count stored result rows grouped by one column.

    Args:
        result_id: Stored result to aggregate.
        group_by: Column to group by.
    """
    receipt, rows = _stored_result(result_id)
    if group_by not in receipt["columns"]:
        raise ToolInputError(f"unknown result column {group_by!r}")
    groups: dict[str, int] = {}
    for row in rows:
        key = str(row[group_by])
        groups[key] = groups.get(key, 0) + 1
    return {"result_id": result_id, "group_by": group_by, "groups": groups, "total": len(rows)}


@tool(parse_docstring=True)
@_guard
def page_result(result_id: str, offset: int = 0, limit: int = results.MAX_PAGE) -> dict:
    """Return a bounded page from a stored result.

    Args:
        result_id: Stored result to page.
        offset: Zero-based row offset.
        limit: Requested page size, capped at the maximum.
    """
    receipt, rows = _stored_result(result_id)
    if offset < 0 or limit < 0:
        raise ToolInputError("offset and limit must be non-negative")
    page_limit = min(limit, results.MAX_PAGE)
    page = rows[offset:offset + page_limit]
    return {"result_id": result_id, "offset": offset, "limit": page_limit,
            "returned": len(page), "total": len(rows), "has_more": offset + len(page) < len(rows),
            "columns": receipt["columns"], "rows": page}


run_query = run_query_inline
TOOLS = [run_query_inline, filter_result, aggregate_result, page_result]
V1_TOOLS = TOOLS
