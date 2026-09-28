-- Fleet, configuration and deferrals. A development database, deliberately
-- shaped like the real thing: a handful of wide fleets, a long tail of small
-- ones, and a deferrals table big enough that an unfiltered fleet query does
-- not fit in a model's context.

CREATE TABLE operators (
  operator_id   TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  region        TEXT NOT NULL
);

CREATE TABLE stations (
  station_code  TEXT PRIMARY KEY,   -- IATA
  city          TEXT NOT NULL,
  country       TEXT NOT NULL,
  is_maintenance_base INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE aircraft_models (
  model         TEXT PRIMARY KEY,   -- '747-8', '737-800'
  manufacturer  TEXT NOT NULL,
  family        TEXT NOT NULL,      -- '747', '737'
  body          TEXT NOT NULL       -- 'wide' | 'narrow'
);

CREATE TABLE aircraft (
  tail_number   TEXT PRIMARY KEY,
  model         TEXT NOT NULL REFERENCES aircraft_models(model),
  operator_id   TEXT NOT NULL REFERENCES operators(operator_id),
  base_station  TEXT NOT NULL REFERENCES stations(station_code),
  delivery_date TEXT NOT NULL,
  status        TEXT NOT NULL       -- 'in service' | 'maintenance' | 'stored'
);

-- Configuration slots are per MODEL and form a tree. A slot with a NULL parent
-- is top level; the 747-8 has 43 of them, which is the number the prototype
-- returned and the one a reviewer will check.
CREATE TABLE config_slots (
  slot_id        INTEGER PRIMARY KEY,
  model          TEXT NOT NULL REFERENCES aircraft_models(model),
  slot_code      TEXT NOT NULL,
  name           TEXT NOT NULL,
  parent_slot_id INTEGER REFERENCES config_slots(slot_id),
  ata_chapter    INTEGER NOT NULL,
  UNIQUE (model, slot_code)
);

-- What a given tail actually has in a slot.
CREATE TABLE aircraft_config (
  tail_number    TEXT NOT NULL REFERENCES aircraft(tail_number),
  slot_id        INTEGER NOT NULL REFERENCES config_slots(slot_id),
  value          TEXT NOT NULL,
  effective_date TEXT NOT NULL,
  PRIMARY KEY (tail_number, slot_id)
);

-- Deferred maintenance (MEL). The big one.
CREATE TABLE deferrals (
  deferral_id   TEXT PRIMARY KEY,
  tail_number   TEXT NOT NULL REFERENCES aircraft(tail_number),
  mel_ref       TEXT NOT NULL,
  category      TEXT NOT NULL,      -- A | B | C | D, shortest interval first
  ata_chapter   INTEGER NOT NULL,
  description   TEXT NOT NULL,
  opened_date   TEXT NOT NULL,
  due_date      TEXT NOT NULL,
  status        TEXT NOT NULL,      -- 'open' | 'closed' | 'extended'
  closed_date   TEXT,
  station_code  TEXT NOT NULL REFERENCES stations(station_code)
);

CREATE INDEX idx_aircraft_model     ON aircraft(model);
CREATE INDEX idx_deferrals_tail     ON deferrals(tail_number);
CREATE INDEX idx_deferrals_status   ON deferrals(status);
CREATE INDEX idx_config_slots_model ON config_slots(model, parent_slot_id);
CREATE INDEX idx_aircraft_config_t  ON aircraft_config(tail_number);
