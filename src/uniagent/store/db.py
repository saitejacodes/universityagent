"""SQLite store with full provenance and history.

`facts` is append-only: every run writes its values with the snapshot hash and quote they came
from, so the database can answer "what did we believe about X on date D, and why?". The
`current_facts` view exposes the latest accepted value per (university, field).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    model       TEXT NOT NULL,
    options     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    university  TEXT NOT NULL,
    page_type   TEXT NOT NULL,
    sha         TEXT NOT NULL,
    url         TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    method      TEXT NOT NULL,
    PRIMARY KEY (university, page_type, sha)
);
CREATE TABLE IF NOT EXISTS facts (
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    university   TEXT NOT NULL,
    field        TEXT NOT NULL,
    value        TEXT,            -- JSON
    status       TEXT NOT NULL,
    confidence   REAL NOT NULL,
    quote        TEXT,
    note         TEXT,
    page_type    TEXT,
    url          TEXT,
    snapshot_sha TEXT,
    PRIMARY KEY (run_id, university, field)
);
CREATE INDEX IF NOT EXISTS facts_by_key ON facts(university, field);
CREATE VIEW IF NOT EXISTS current_facts AS
    SELECT f.* FROM facts f
    JOIN runs r ON r.run_id = f.run_id
    WHERE r.started_at = (
        SELECT MAX(r2.started_at) FROM facts f2 JOIN runs r2 ON r2.run_id = f2.run_id
        WHERE f2.university = f.university AND f2.field = f.field AND f2.value IS NOT NULL);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def add_run(self, run_id: str, started_at: str, model: str, options: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?)",
            (run_id, started_at, model, json.dumps(options)),
        )
        self.conn.commit()

    def add_snapshot(
        self, university: str, page_type: str, sha: str, url: str, fetched_at: str, method: str
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?,?,?)",
            (university, page_type, sha, url, fetched_at, method),
        )
        self.conn.commit()

    def add_facts(self, run_id: str, rows: list[dict]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    run_id,
                    r["university"],
                    r["field"],
                    json.dumps(r["value"]) if r["value"] is not None else None,
                    r["status"],
                    r["confidence"],
                    r.get("quote"),
                    r.get("note"),
                    r.get("page_type"),
                    r.get("url"),
                    r.get("snapshot_sha"),
                )
                for r in rows
            ],
        )
        self.conn.commit()

    def current(self, university: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT field, value, confidence, quote, url, snapshot_sha, run_id FROM current_facts "
            "WHERE university = ? ORDER BY field",
            (university,),
        )
        cols = [d[0] for d in cur.description]
        out = []
        for row in cur.fetchall():
            d = dict(zip(cols, row, strict=True))
            d["value"] = json.loads(d["value"]) if d["value"] else None
            out.append(d)
        return out
