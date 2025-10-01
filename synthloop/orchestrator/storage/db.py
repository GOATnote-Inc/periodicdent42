from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./synthloop.db")


def _sqlite_path() -> Path:
    if _DB_URL.startswith("sqlite:///"):
        return Path(_DB_URL.replace("sqlite:///", ""))
    raise ValueError("Only sqlite URLs are supported in this demo")


_DB_PATH = _sqlite_path()
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def reset_engine():
    global _DB_PATH
    _DB_PATH = _sqlite_path()
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _connect():
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY,
                plan_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                operator TEXT,
                code_sha TEXT,
                image_digest TEXT,
                backend TEXT,
                real_device INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                name TEXT,
                params_json TEXT,
                started_at TEXT,
                ended_at TEXT,
                status TEXT,
                error TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                step_id INTEGER,
                type TEXT,
                parquet_path TEXT,
                summary_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                ts TEXT,
                type TEXT,
                payload_json TEXT,
                payload_sha256 TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS qc_reports (
                run_id TEXT PRIMARY KEY,
                overall_status TEXT,
                rules_summary_json TEXT,
                created_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS outcomes (
                run_id TEXT PRIMARY KEY,
                success INTEGER,
                failure_mode TEXT,
                notes TEXT,
                evidence_json TEXT,
                created_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                path TEXT,
                sha256 TEXT,
                media_type TEXT
            )
            """
        )
        conn.commit()


def insert_experiment(run_id: str, plan_json: str, operator: str, backend: str, real_device: bool):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO experiments(run_id, plan_json, created_at, operator, backend, real_device) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, plan_json, datetime.utcnow().isoformat(), operator, backend, int(real_device)),
        )
        conn.commit()


def insert_step(run_id: str, name: str, params_json: str, started_at: datetime, ended_at: datetime, status: str, error: Optional[str]) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO steps(run_id, name, params_json, started_at, ended_at, status, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                name,
                params_json,
                started_at.isoformat() if started_at else None,
                ended_at.isoformat() if ended_at else None,
                status,
                error,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def insert_measurement(run_id: str, step_id: int, path: str, summary_json: str):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO measurements(run_id, step_id, type, parquet_path, summary_json) VALUES (?, ?, ?, ?, ?)",
            (run_id, step_id, "telemetry", path, summary_json),
        )
        conn.commit()


def record_event(run_id: str, event_type: str, payload: dict):
    with _connect() as conn:
        payload_json = json.dumps(payload, default=str)
        conn.execute(
            "INSERT INTO events(run_id, ts, type, payload_json, payload_sha256) VALUES (?, ?, ?, ?, ?)",
            (
                run_id,
                datetime.utcnow().isoformat(),
                event_type,
                payload_json,
                hashlib.sha256(payload_json.encode()).hexdigest(),
            ),
        )
        conn.commit()


def write_qc_report(run_id: str, overall: str, rules: List[dict]):
    with _connect() as conn:
        conn.execute(
            "REPLACE INTO qc_reports(run_id, overall_status, rules_summary_json, created_at) VALUES (?, ?, ?, ?)",
            (run_id, overall, json.dumps(rules, default=str), datetime.utcnow().isoformat()),
        )
        conn.commit()


def write_outcome(run_id: str, success: bool, failure_mode: Optional[str], notes: Optional[str], evidence: dict):
    with _connect() as conn:
        cur = conn.execute("SELECT run_id FROM outcomes WHERE run_id = ?", (run_id,))
        if cur.fetchone():
            raise ValueError("Outcome already recorded")
        conn.execute(
            "INSERT INTO outcomes(run_id, success, failure_mode, notes, evidence_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                int(success),
                failure_mode,
                notes,
                json.dumps(evidence, default=str),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()


def list_experiments() -> List[sqlite3.Row]:
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM experiments")
        return cur.fetchall()


def list_outcomes() -> Dict[str, sqlite3.Row]:
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM outcomes")
        return {row["run_id"]: row for row in cur.fetchall()}


def list_steps(run_id: str) -> List[sqlite3.Row]:
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM steps WHERE run_id = ?", (run_id,))
        return cur.fetchall()


def get_qc_report(run_id: str) -> Optional[sqlite3.Row]:
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM qc_reports WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        return row


def add_attachment(run_id: str, path: Path, media_type: str):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO attachments(run_id, path, sha256, media_type) VALUES (?, ?, ?, ?)",
            (run_id, str(path), "", media_type),
        )
        conn.commit()

