"""Fast, hermetic checks on the parts that must not drift.

No model calls: these test the result store and the tools, which is where the
architecture lives. The agent's behaviour is measured by the experiments.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/agent"))

import results  # noqa: E402
from catalog import CATALOG  # noqa: E402
from tools.query import (aggregate_result, filter_result, page_result,  # noqa: E402
                         run_query, run_query_inline)


@pytest.fixture(scope="module")
def big():
    return run_query.invoke({"query_id": "open_deferrals_by_fleet",
                             "params": {"model": "737-800"}})


def test_receipt_is_small_whatever_the_result_size(big):
    """The property the whole design rests on."""
    assert big["row_count"] > 400
    assert "rows" not in big                      # rows withheld
    assert len(json.dumps(big)) < 5_000           # receipt stays tiny


def test_receipt_carries_the_exact_count(big):
    """A truncated list would lose this; a receipt must not."""
    import sqlite3
    con = sqlite3.connect(ROOT / "data" / "fleet.db")
    truth = con.execute(
        "SELECT COUNT(*) FROM deferrals d JOIN aircraft a USING(tail_number) "
        "WHERE a.model='737-800' AND d.status IN ('open','extended')").fetchone()[0]
    assert big["row_count"] == truth


def test_small_results_come_back_inline():
    r = run_query.invoke({"query_id": "open_deferrals_by_tail",
                          "params": {"tail_number": "N101FL"}})
    assert r["row_count"] <= 25
    assert r.get("rows") is not None


def test_747_8_has_43_top_level_slots():
    """The figure the prototype reported, pinned so data changes cannot drift it."""
    r = run_query.invoke({"query_id": "top_level_config_slots", "params": {"model": "747-8"}})
    assert r["row_count"] == 43


def test_station_counts_are_preaggregated():
    r = run_query_inline.invoke({"query_id": "deferral_counts_by_station",
                                 "params": {"model": "737-800"}})
    assert r["rows"][0]["station_code"] == "ORD"
    assert r["rows"][0]["open_deferrals"] == 105


def test_page_result_cannot_be_talked_into_a_dump(big):
    p = page_result.invoke({"result_id": big["result_id"], "offset": 0, "limit": 10_000})
    assert p["returned"] == results.MAX_PAGE
    assert p["has_more"] is True


def test_aggregate_matches_sql(big):
    """The deterministic count is the thing the naive build got wrong."""
    import sqlite3
    con = sqlite3.connect(ROOT / "data" / "fleet.db")
    truth = dict(con.execute(
        "SELECT d.station_code, COUNT(*) FROM deferrals d JOIN aircraft a USING(tail_number) "
        "WHERE a.model='737-800' AND d.status IN ('open','extended') GROUP BY 1").fetchall())
    got = aggregate_result.invoke({"result_id": big["result_id"], "group_by": "station_code"})
    assert got["groups"] == {k: v for k, v in sorted(truth.items(), key=lambda kv: -kv[1])}


def test_filter_chains_and_returns_a_new_result(big):
    a = filter_result.invoke({"result_id": big["result_id"], "column": "category",
                              "op": "eq", "value": "C"})
    b = filter_result.invoke({"result_id": a["result_id"], "column": "station_code",
                              "op": "eq", "value": "ORD"})
    assert b["row_count"] <= a["row_count"] <= big["row_count"]
    assert b["result_id"] != a["result_id"]


def test_missing_parameter_is_correctable_not_fatal():
    """It used to raise and end the run."""
    out = run_query.invoke({"query_id": "open_deferrals_by_fleet", "params": {}})
    assert "error" in out and "model" in out["error"]


def test_unknown_query_is_rejected_with_the_catalogue():
    out = run_query.invoke({"query_id": "DROP TABLE aircraft"})
    assert "error" in out and "fleet_summary" in out["error"]


def test_catalogue_declares_params_for_every_placeholder():
    """A :placeholder with no declared Param would fail only at call time."""
    import re
    for q in CATALOG.values():
        placeholders = set(re.findall(r":(\w+)", q.sql))
        declared = {p.name for p in q.params}
        assert placeholders == declared, f"{q.id}: {placeholders} vs {declared}"


def test_v1_returns_every_row():
    """The naive shape, kept honest: if this ever stops being huge the
    comparison stops meaning anything."""
    r = run_query_inline.invoke({"query_id": "open_deferrals_by_fleet",
                                 "params": {"model": "737-800"}})
    assert len(r["rows"]) == r["row_count"] > 400
    assert len(json.dumps(r["rows"])) > 100_000


def test_filter_records_lineage_so_it_can_be_reverted():
    """The gap that made 'narrow that' a one-way door: without a parent, going
    back means re-running the query, often with different parameters."""
    from tools.query import list_results
    big = run_query.invoke({"query_id": "open_deferrals_by_fleet",
                            "params": {"model": "737-800"}})
    narrowed = filter_result.invoke({"result_id": big["result_id"],
                                     "column": "station_code", "op": "eq", "value": "ORD"})
    assert narrowed["parent_id"] == big["result_id"]
    assert narrowed["derived_by"] == "station_code eq ORD"
    listed = {r["result_id"]: r for r in list_results.invoke({})["results"]}
    assert listed[narrowed["result_id"]]["parent_id"] == big["result_id"]


def test_receipt_advertises_what_can_be_adjusted():
    """A receipt that only says '511 rows' reads as 'job done'. The breakdown
    keys are the columns you can filter on; the inner keys are the values."""
    big = run_query.invoke({"query_id": "open_deferrals_by_fleet",
                            "params": {"model": "737-800"}})
    assert set(big["breakdown"]) >= {"category", "station_code", "status"}
    assert set(big["breakdown"]["category"]) == {"A", "B", "C", "D"}


def test_breakdown_is_derived_from_data_not_declared():
    """`status` is refinable and no query declares it. The old design keyed off
    a per-query facet list, so a column nobody listed was invisible."""
    big = run_query.invoke({"query_id": "open_deferrals_by_fleet",
                            "params": {"model": "737-800"}})
    assert "status" in big["breakdown"]
    assert "deferral_id" not in big["breakdown"]      # 511 distinct: not a facet


def test_results_outlive_the_message_that_carried_them():
    """A handle from an earlier turn is still usable."""
    from tools.query import list_results
    big = run_query.invoke({"query_id": "open_deferrals_by_fleet",
                            "params": {"model": "737-800"}})
    known = list_results.invoke({})
    assert any(r["result_id"] == big["result_id"] for r in known["results"])


def test_an_original_result_has_no_parent():
    big = run_query.invoke({"query_id": "top_level_config_slots", "params": {"model": "747-8"}})
    assert big["parent_id"] is None


def test_distinct_counts_answer_how_many_aircraft_without_a_tool_call():
    """Simplifying the breakdown once dropped high-cardinality columns entirely,
    and v2 lost an answer it used to get: 'how many distinct aircraft have an
    open category C deferral' is exactly this number."""
    big = run_query.invoke({"query_id": "open_deferrals_by_fleet",
                            "params": {"model": "737-800"}})
    cat_c = filter_result.invoke({"result_id": big["result_id"], "column": "category",
                                  "op": "eq", "value": "C"})
    import sqlite3
    con = sqlite3.connect(ROOT / "data" / "fleet.db")
    truth = con.execute(
        "SELECT COUNT(DISTINCT d.tail_number) FROM deferrals d JOIN aircraft a "
        "USING(tail_number) WHERE a.model='737-800' AND d.status IN ('open','extended') "
        "AND d.category='C'").fetchone()[0]
    assert cat_c["distinct"]["tail_number"] == truth


def test_a_column_is_either_broken_down_or_counted_never_both():
    big = run_query.invoke({"query_id": "open_deferrals_by_fleet",
                            "params": {"model": "737-800"}})
    assert not (set(big["breakdown"]) & set(big["distinct"]))
