"""The pre-approved queries. The agent chooses one and supplies parameters; it
never writes SQL.

Two properties matter and both come from this being a fixed catalogue rather
than generated SQL:

  * every statement is reviewed once, by a human, and bound with parameters, so
    there is no injection surface and no way to reach a table nobody approved;
  * every query declares how big its result can get, which is what lets the
    runtime decide to offload BEFORE the rows exist rather than after they have
    already been pulled into a message.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ParamType = Literal["tail_number", "model", "station", "int", "category", "ata"]


@dataclass(frozen=True)
class Param:
    name: str
    type: ParamType
    description: str


@dataclass(frozen=True)
class Query:
    id: str
    summary: str
    sql: str
    params: tuple[Param, ...] = ()
    #: Rough upper bound on rows, used to plan. `unbounded` means the row count
    #: scales with fleet size and can exceed anything that belongs in context.
    size: Literal["small", "medium", "unbounded"] = "small"
    #: Columns worth grouping on when a result turns out to be too large.
    facets: tuple[str, ...] = field(default_factory=tuple)


CATALOG: dict[str, Query] = {q.id: q for q in (
    Query(
        id="fleet_summary",
        summary="One row per model: fleet size, in-service count, open deferrals.",
        size="small",
        sql="""
            SELECT m.model, m.manufacturer, m.body,
                   COUNT(DISTINCT a.tail_number) AS aircraft,
                   SUM(CASE WHEN a.status='in service' THEN 1 ELSE 0 END) AS in_service,
                   (SELECT COUNT(*) FROM deferrals d JOIN aircraft a2 USING (tail_number)
                     WHERE a2.model = m.model AND d.status IN ('open','extended')) AS open_deferrals
              FROM aircraft_models m LEFT JOIN aircraft a ON a.model = m.model
             GROUP BY m.model ORDER BY aircraft DESC
        """,
    ),
    Query(
        id="top_level_config_slots",
        summary="Top-level configuration slots for a model (parent slot is null).",
        params=(Param("model", "model", "Aircraft model, e.g. '747-8'"),),
        size="small",
        facets=("ata_chapter",),
        sql="""
            SELECT slot_id, slot_code, name, ata_chapter
              FROM config_slots
             WHERE model = :model AND parent_slot_id IS NULL
             ORDER BY ata_chapter, slot_code
        """,
    ),
    Query(
        id="config_for_tail",
        summary="Every configuration slot and its value for one aircraft.",
        params=(Param("tail_number", "tail_number", "Registration, e.g. 'N101FL'"),),
        size="medium",
        facets=("ata_chapter", "value"),
        sql="""
            SELECT s.slot_code, s.name, s.ata_chapter, c.value, c.effective_date
              FROM aircraft_config c JOIN config_slots s USING (slot_id)
             WHERE c.tail_number = :tail_number
             ORDER BY s.ata_chapter, s.slot_code
        """,
    ),
    Query(
        id="open_deferrals_by_tail",
        summary="Open and extended deferrals on one aircraft.",
        params=(Param("tail_number", "tail_number", "Registration, e.g. 'N101FL'"),),
        size="small",
        facets=("category", "ata_chapter"),
        sql="""
            SELECT deferral_id, mel_ref, category, ata_chapter, description,
                   opened_date, due_date, status, station_code
              FROM deferrals
             WHERE tail_number = :tail_number AND status IN ('open','extended')
             ORDER BY due_date
        """,
    ),
    Query(
        id="open_deferrals_by_fleet",
        summary="Open and extended deferrals across every aircraft of a model. "
                "Large: a mainline narrowbody fleet returns several hundred rows.",
        params=(Param("model", "model", "Aircraft model, e.g. '737-800'"),),
        size="unbounded",
        facets=("category", "ata_chapter", "station_code", "tail_number"),
        sql="""
            SELECT d.deferral_id, d.tail_number, d.mel_ref, d.category, d.ata_chapter,
                   d.description, d.opened_date, d.due_date, d.status, d.station_code
              FROM deferrals d JOIN aircraft a USING (tail_number)
             WHERE a.model = :model AND d.status IN ('open','extended')
             ORDER BY d.due_date
        """,
    ),
    Query(
        id="deferrals_due_within",
        summary="Open deferrals falling due within N days, fleet-wide. Large.",
        params=(Param("days", "int", "Horizon in days, e.g. 7"),),
        size="unbounded",
        facets=("category", "model", "station_code"),
        sql="""
            SELECT d.deferral_id, d.tail_number, a.model, d.category, d.ata_chapter,
                   d.description, d.due_date, d.status, d.station_code
              FROM deferrals d JOIN aircraft a USING (tail_number)
             WHERE d.status IN ('open','extended')
               AND julianday(d.due_date) - julianday('2026-09-27') BETWEEN 0 AND :days
             ORDER BY d.due_date
        """,
    ),
    Query(
        id="deferral_counts_by_category",
        summary="Open deferral counts grouped by MEL category for a model.",
        params=(Param("model", "model", "Aircraft model, e.g. '737-800'"),),
        size="small",
        sql="""
            SELECT d.category, COUNT(*) AS open_deferrals,
                   COUNT(DISTINCT d.tail_number) AS aircraft_affected
              FROM deferrals d JOIN aircraft a USING (tail_number)
             WHERE a.model = :model AND d.status IN ('open','extended')
             GROUP BY d.category ORDER BY d.category
        """,
    ),
    Query(
        id="aircraft_in_fleet",
        summary="Every aircraft of a model, with base and status.",
        params=(Param("model", "model", "Aircraft model, e.g. '747-8'"),),
        size="medium",
        facets=("status", "base_station", "operator_id"),
        sql="""
            SELECT a.tail_number, a.operator_id, a.base_station, a.delivery_date, a.status
              FROM aircraft a WHERE a.model = :model ORDER BY a.tail_number
        """,
    ),
    Query(
        id="deferrals_by_ata",
        summary="Open deferrals for a model in one ATA chapter.",
        params=(Param("model", "model", "Aircraft model"),
                Param("ata_chapter", "ata", "ATA chapter number, e.g. 25")),
        size="medium",
        facets=("category", "tail_number"),
        sql="""
            SELECT d.deferral_id, d.tail_number, d.mel_ref, d.category,
                   d.description, d.due_date, d.status
              FROM deferrals d JOIN aircraft a USING (tail_number)
             WHERE a.model = :model AND d.ata_chapter = :ata_chapter
               AND d.status IN ('open','extended')
             ORDER BY d.due_date
        """,
    ),
    Query(
        id="aircraft_at_station",
        summary="Aircraft based at a station, any model.",
        params=(Param("station_code", "station", "IATA code, e.g. 'MEM'"),),
        size="medium",
        facets=("model", "status"),
        sql="""
            SELECT a.tail_number, a.model, a.operator_id, a.status
              FROM aircraft a WHERE a.base_station = :station_code
             ORDER BY a.model, a.tail_number
        """,
    ),
)}


def describe_catalog() -> str:
    """The catalogue as the model sees it in its prompt."""
    lines = []
    for q in CATALOG.values():
        args = ", ".join(f"{p.name}: {p.type}" for p in q.params) or "no parameters"
        flag = "  [CAN BE LARGE]" if q.size == "unbounded" else ""
        lines.append(f"- {q.id}({args}){flag}\n    {q.summary}")
    return "\n".join(lines)
