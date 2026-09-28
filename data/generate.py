"""Build fleet.db. Deterministic: same seed, same database, every time.

Two numbers are load-bearing and asserted at the end:

  * the 747-8 has exactly 43 top-level configuration slots -- the figure the
    prototype returned, and the one a reviewer checks first;
  * `open_deferrals_by_fleet('737-800')` returns ~500 rows, which is the
    result size that does not fit in a model's context and is the whole
    reason this demo exists.

Neither is a coincidence in the data; both are tuned here on purpose.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import random
import sqlite3

ROOT = pathlib.Path(__file__).resolve().parent
DB = ROOT / "fleet.db"
SEED = 20260927
TODAY = dt.date(2026, 9, 27)

OPERATORS = [
    ("OP-ATL", "Atlantic Cargo", "NA"), ("OP-PAC", "Pacific Freight", "APAC"),
    ("OP-EUR", "Europa Air", "EMEA"), ("OP-NOR", "Nordic Link", "EMEA"),
    ("OP-AND", "Andes Connect", "LATAM"),
]
STATIONS = [
    ("ANC","Anchorage","US",1), ("MEM","Memphis","US",1), ("LAX","Los Angeles","US",0),
    ("JFK","New York","US",0), ("ORD","Chicago","US",1), ("FRA","Frankfurt","DE",1),
    ("AMS","Amsterdam","NL",0), ("CDG","Paris","FR",0), ("HKG","Hong Kong","HK",1),
    ("NRT","Tokyo","JP",0), ("SIN","Singapore","SG",1), ("DXB","Dubai","AE",0),
    ("GRU","Sao Paulo","BR",0), ("SCL","Santiago","CL",0), ("ARN","Stockholm","SE",0),
]
#: model -> (manufacturer, family, body, fleet size, top-level config slots)
#: 737-800 is deliberately the widest fleet: it is what makes an unfiltered
#: fleet-wide deferral query oversized.
MODELS = {
    "737-800":   ("Boeing", "737", "narrow", 134, 38),
    "747-8":     ("Boeing", "747", "wide",    12, 43),
    "787-9":     ("Boeing", "787", "wide",    38, 41),
    "777-300ER": ("Boeing", "777", "wide",    24, 45),
    "767-300F":  ("Boeing", "767", "wide",    16, 36),
    "A320-200":  ("Airbus", "A320", "narrow", 78, 37),
    "A350-900":  ("Airbus", "A350", "wide",   18, 40),
    "E175":      ("Embraer", "E-Jet", "narrow", 29, 31),
}
SLOT_AREAS = [
    (21, "Air Conditioning"), (22, "Auto Flight"), (23, "Communications"),
    (24, "Electrical Power"), (25, "Equipment & Furnishings"), (26, "Fire Protection"),
    (27, "Flight Controls"), (28, "Fuel"), (29, "Hydraulic Power"),
    (30, "Ice & Rain Protection"), (31, "Indicating & Recording"), (32, "Landing Gear"),
    (33, "Lights"), (34, "Navigation"), (35, "Oxygen"), (36, "Pneumatic"),
    (38, "Water & Waste"), (44, "Cabin Systems"), (45, "Central Maintenance"),
    (46, "Information Systems"), (49, "Auxiliary Power"), (52, "Doors"),
    (53, "Fuselage"), (56, "Windows"), (57, "Wings"), (71, "Power Plant"),
    (73, "Engine Fuel & Control"), (77, "Engine Indicating"), (78, "Exhaust"),
]
SLOT_KINDS = [
    "Configuration", "Option Group", "Installed Equipment", "Capability",
    "Provisioning", "Interface", "Modification State",
]
DEFERRAL_TEXT = [
    "Cabin reading light inoperative, seat {seat}",
    "Lavatory {lav} flush motor intermittent",
    "APU bleed valve slow to respond",
    "Cargo compartment smoke detector loop {n} faulted",
    "IFE seat box inoperative, seat {seat}",
    "Left windshield heat controller degraded",
    "Galley chiller {n} not maintaining temperature",
    "Wing anti-ice pressure switch out of tolerance",
    "Nosewheel steering shimmy damper seepage",
    "Overhead bin latch damaged, row {n}",
    "Passenger oxygen generator past service life, row {n}",
    "Hydraulic system {n} reservoir quantity indication erratic",
    "Cockpit door camera image degraded",
    "Potable water quantity indication inoperative",
    "Engine {n} oil filter bypass indication intermittent",
    "Weather radar tilt drive sluggish",
    "Fuel tank boost pump {n} low output",
    "Emergency exit sign panel dim, door {n}",
]
CATEGORY_DAYS = {"A": 3, "B": 3, "C": 10, "D": 120}


def main() -> int:
    rng = random.Random(SEED)
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript((ROOT / "schema.sql").read_text())

    con.executemany("INSERT INTO operators VALUES (?,?,?)", OPERATORS)
    con.executemany("INSERT INTO stations VALUES (?,?,?,?)", STATIONS)
    con.executemany("INSERT INTO aircraft_models VALUES (?,?,?,?)",
                    [(m, v[0], v[1], v[2]) for m, v in MODELS.items()])

    # --- aircraft -----------------------------------------------------------
    aircraft: list[tuple] = []
    seq = 100
    for model, (_, _, _, fleet_size, _) in MODELS.items():
        for _ in range(fleet_size):
            seq += 1
            tail = f"N{seq}FL"
            operator = rng.choice(OPERATORS)[0]
            base = rng.choice([s for s in STATIONS if s[3]])[0]
            delivered = TODAY - dt.timedelta(days=rng.randint(400, 6200))
            status = rng.choices(["in service", "maintenance", "stored"],
                                 weights=[88, 9, 3])[0]
            aircraft.append((tail, model, operator, base, delivered.isoformat(), status))
    con.executemany("INSERT INTO aircraft VALUES (?,?,?,?,?,?)", aircraft)

    # --- configuration slots ------------------------------------------------
    # Top level first, so the per-model count is exact; then 0-3 children each.
    slot_id = 0
    slots: list[tuple] = []
    tops_by_model: dict[str, list[int]] = {}
    for model, (_, _, _, _, top_count) in MODELS.items():
        tops: list[int] = []
        areas = rng.sample(SLOT_AREAS, k=min(top_count, len(SLOT_AREAS)))
        while len(areas) < top_count:            # reuse areas with a new kind
            areas.append(rng.choice(SLOT_AREAS))
        for i, (ata, area) in enumerate(areas, start=1):
            slot_id += 1
            kind = SLOT_KINDS[i % len(SLOT_KINDS)]
            slots.append((slot_id, model, f"{ata}-{i:02d}", f"{area} {kind}", None, ata))
            tops.append(slot_id)
        tops_by_model[model] = tops
        for parent in tops:
            for j in range(rng.randint(0, 3)):
                slot_id += 1
                parent_row = next(s for s in slots if s[0] == parent)
                slots.append((slot_id, model, f"{parent_row[2]}.{j+1}",
                              f"{parent_row[3]} — detail {j+1}", parent, parent_row[5]))
    con.executemany("INSERT INTO config_slots VALUES (?,?,?,?,?,?)", slots)

    # --- per-tail configuration values --------------------------------------
    by_model_slots: dict[str, list[int]] = {}
    for s in slots:
        by_model_slots.setdefault(s[1], []).append(s[0])
    values = ["fitted", "not fitted", "provisioned only", "deactivated",
              "standard", "extended range", "option A", "option B"]
    rows = []
    for tail, model, *_ in aircraft:
        for sid in by_model_slots[model]:
            rows.append((tail, sid, rng.choice(values),
                         (TODAY - dt.timedelta(days=rng.randint(30, 3000))).isoformat()))
    con.executemany("INSERT INTO aircraft_config VALUES (?,?,?,?)", rows)

    # --- deferrals ----------------------------------------------------------
    # ~4 open per in-service aircraft, which puts the 737-800 fleet near 500.
    deferrals = []
    n = 0
    for tail, model, _, base, _, status in aircraft:
        count = rng.randint(2, 9) if status == "in service" else rng.randint(0, 3)
        for _ in range(count):
            n += 1
            cat = rng.choices(["A", "B", "C", "D"], weights=[6, 14, 45, 35])[0]
            opened = TODAY - dt.timedelta(days=rng.randint(0, 150))
            due = opened + dt.timedelta(days=CATEGORY_DAYS[cat])
            # Most are open; a closed one has a date, an extended one gets more time.
            state = rng.choices(["open", "closed", "extended"], weights=[64, 26, 10])[0]
            closed = None
            if state == "closed":
                closed = (opened + dt.timedelta(days=rng.randint(1, CATEGORY_DAYS[cat]))).isoformat()
            if state == "extended":
                due = due + dt.timedelta(days=CATEGORY_DAYS[cat])
            ata, _area = rng.choice(SLOT_AREAS)
            text = rng.choice(DEFERRAL_TEXT).format(
                seat=f"{rng.randint(1,48)}{rng.choice('ABCDEFGHJK')}",
                lav=rng.choice("ABCDEF"), n=rng.randint(1, 4))
            deferrals.append((f"DEF-{n:05d}", tail, f"MEL {ata}-{rng.randint(10,99)}-{rng.randint(1,9)}",
                              cat, ata, text, opened.isoformat(), due.isoformat(),
                              state, closed, base))
    con.executemany("INSERT INTO deferrals VALUES (?,?,?,?,?,?,?,?,?,?,?)", deferrals)
    con.commit()

    # --- the two load-bearing numbers ---------------------------------------
    top_747 = con.execute(
        "SELECT COUNT(*) FROM config_slots WHERE model='747-8' AND parent_slot_id IS NULL"
    ).fetchone()[0]
    open_737 = con.execute(
        "SELECT COUNT(*) FROM deferrals d JOIN aircraft a USING (tail_number) "
        "WHERE a.model='737-800' AND d.status IN ('open','extended')"
    ).fetchone()[0]
    assert top_747 == 43, f"747-8 top-level slots: expected 43, got {top_747}"
    assert 430 <= open_737 <= 620, f"737-800 open deferrals: {open_737} outside the demo range"

    print(f"  aircraft        {len(aircraft)}")
    print(f"  config slots    {len(slots)}  ({top_747} top level on the 747-8)")
    print(f"  config values   {len(rows)}")
    print(f"  deferrals       {len(deferrals)}  ({open_737} open on the 737-800 fleet)")
    print(f"  -> {DB}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
