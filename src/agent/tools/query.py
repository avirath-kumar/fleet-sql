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

@tool
@_guard
def run_query_inline(query_id: str, params: dict | None = None) -> dict:
    """Run an approved query, returning small results inline and larger rows in rows_file."""
    q = _lookup(query_id)
    bound = _check_params(q, params or {})
    columns, rows = results.query(q.sql, bound)
    if len(rows) <= results.MAX_INLINE or q.size == "small":
        return {"query_id": q.id, "params": bound, "row_count": len(rows), "rows": rows}
    relative_path = results.write_rows(q.id, rows)
    return {
        "query_id": q.id,
        "params": bound,
        "row_count": len(rows),
        "columns": columns,
        "preview": rows[:results.PREVIEW_ROWS],
        "rows_file": relative_path,
        "note": "Full rows are newline-delimited JSON at rows_file; use grep or read_file on it.",
    }


TOOLS = [run_query_inline]
V1_TOOLS = TOOLS
