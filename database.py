import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
DB_PATH = Path("data/universities.db")


class Database:
    def __init__(self) -> None:
        DB_PATH.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH))
        self._create_tables()

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS universities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                scraped_at TEXT,
                overall_confidence REAL,
                data_json TEXT          -- full UniversityData as JSON
            );

            CREATE TABLE IF NOT EXISTS raw_fields (
                university_id TEXT,
                field_name TEXT,
                value_json TEXT,        -- raw LLM extraction result
                confidence REAL,
                source_url TEXT,
                needs_review INTEGER,
                notes TEXT,
                scraped_at TEXT,
                PRIMARY KEY (university_id, field_name)
            );

            CREATE TABLE IF NOT EXISTS validation_reports (
                university_id TEXT PRIMARY KEY,
                report_json TEXT,
                quality_score REAL,
                ready_for_storage INTEGER,
                created_at TEXT
            );
        """)
        self.conn.commit()

    def upsert_raw_fields(self, university_id: str, raw_data: dict) -> None:
        """Store all raw LLM extractions per field."""
        now = datetime.utcnow().isoformat()
        for field_name, field_data in raw_data.items():
            if not isinstance(field_data, dict):
                continue
            self.conn.execute("""
                INSERT OR REPLACE INTO raw_fields
                (university_id, field_name, value_json, confidence, source_url,
                 needs_review, notes, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                university_id,
                field_name,
                json.dumps(field_data.get("value")),
                field_data.get("confidence", 0.0),
                field_data.get("source_url"),
                int(field_data.get("needs_review", False)),
                field_data.get("notes"),
                now,
            ))
        self.conn.commit()
        logger.info(f"Stored {len(raw_data)} raw fields for {university_id}")

    def upsert_university(self, university_id: str, name: str, data: dict, confidence: float) -> None:
        """Store the full university record as JSON."""
        self.conn.execute("""
            INSERT OR REPLACE INTO universities (id, name, scraped_at, overall_confidence, data_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            university_id,
            name,
            datetime.utcnow().isoformat(),
            confidence,
            json.dumps(data),
        ))
        self.conn.commit()
        logger.info(f"Upserted university record: {university_id}")

    def save_validation_report(self, university_id: str, report: dict) -> None:
        """Persist validation report for the eval step."""
        self.conn.execute("""
            INSERT OR REPLACE INTO validation_reports
            (university_id, report_json, quality_score, ready_for_storage, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            university_id,
            json.dumps(report),
            report.get("overall_quality_score", 0.0),
            int(report.get("ready_for_storage", False)),
            datetime.utcnow().isoformat(),
        ))
        self.conn.commit()

    def get_all_universities(self) -> list[dict]:
        """Return all stored university records."""
        rows = self.conn.execute(
            "SELECT id, name, scraped_at, overall_confidence FROM universities"
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "scraped_at": r[2], "confidence": r[3]}
            for r in rows
        ]

    def get_university_data(self, university_id: str) -> dict | None:
        """Return full data JSON for a university."""
        row = self.conn.execute(
            "SELECT data_json FROM universities WHERE id = ?", (university_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self.conn.close()
