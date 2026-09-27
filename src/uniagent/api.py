"""Read-only HTTP API over the extracted database.

    GET /universities                       ids + names
    GET /universities/{id}                  current value of every field, with evidence
    GET /universities/{id}/history/{field}  every value ever extracted, run by run

Every value is returned with the quote and URL it came from, so a consumer can show "source"
links and decide per field whether the confidence is high enough to use.
"""

from __future__ import annotations

import json
import sqlite3

from fastapi import FastAPI, HTTPException

from uniagent.config import get_settings, load_universities

app = FastAPI(title="uniagent — evidence-backed university facts", version="2.0.0")


def _conn() -> sqlite3.Connection:
    path = get_settings().db_path
    if not path.exists():
        raise HTTPException(503, "no data yet: run `uniagent crawl` and `uniagent extract`")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/universities")
def universities():
    return [{"id": u.id, "name": u.name, "country": u.country} for u in load_universities()]


@app.get("/universities/{uid}")
def university(uid: str, min_confidence: float = 0.0):
    rows = (
        _conn()
        .execute(
            "SELECT field, value, confidence, status, quote, url, snapshot_sha, run_id "
            "FROM current_facts WHERE university = ? ORDER BY field",
            (uid,),
        )
        .fetchall()
    )
    if not rows:
        raise HTTPException(404, f"no facts for '{uid}'")
    return {
        "university": uid,
        "fields": {
            r["field"]: {
                "value": json.loads(r["value"]) if r["value"] else None,
                "confidence": r["confidence"],
                "evidence": {"quote": r["quote"], "url": r["url"], "snapshot": r["snapshot_sha"]},
                "run": r["run_id"],
            }
            for r in rows
            if r["confidence"] >= min_confidence
        },
    }


@app.get("/universities/{uid}/history/{field}")
def history(uid: str, field: str):
    rows = (
        _conn()
        .execute(
            "SELECT r.started_at, f.run_id, f.value, f.confidence, f.status, f.quote, f.url "
            "FROM facts f JOIN runs r ON r.run_id = f.run_id "
            "WHERE f.university = ? AND f.field = ? ORDER BY r.started_at",
            (uid, field),
        )
        .fetchall()
    )
    return [
        {
            "at": r["started_at"],
            "run": r["run_id"],
            "status": r["status"],
            "value": json.loads(r["value"]) if r["value"] else None,
            "confidence": r["confidence"],
            "quote": r["quote"],
            "url": r["url"],
        }
        for r in rows
    ]
