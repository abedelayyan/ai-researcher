"""SQLite access for Signal Zero.

Two ways in:

``connect``              full access, used by ingest, outcomes and rendering.
``open_feature_scoped``  a connection with an authoriser that refuses to read the
                         outcome tables at all. Feature extraction and scoring use
                         this, so leakage fails loudly at the SQLite layer rather
                         than quietly corrupting the prediction log.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

#: Tables that hold information which did not exist at publication time.
OUTCOME_TABLES = frozenset({"paper_outcomes", "outcome_labels"})


class LeakageError(sqlite3.DatabaseError):
    """Raised when day-zero code reaches for an outcome table."""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(db_path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the database, creating the file and applying migrations if needed."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if read_only and path.exists():
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    if not read_only:
        migrate(conn)
    return conn


def open_feature_scoped(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection that cannot see outcome data.

    The authoriser denies every read, write and schema operation touching a table in
    :data:`OUTCOME_TABLES`. Day-zero code gets this connection and nothing else, which
    is what makes the anti-leakage rule an enforced property rather than a habit.
    """
    conn = connect(db_path)

    def authorizer(action: int, arg1: Any, arg2: Any, db_name: Any, trigger: Any) -> int:
        for arg in (arg1, arg2):
            if isinstance(arg, str) and arg.lower() in OUTCOME_TABLES:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(authorizer)
    return conn


def close(conn: sqlite3.Connection) -> None:
    """Checkpoint and close.

    The database file is committed to the repo, so any write still sitting in the
    write-ahead log would be lost when the runner is torn down.
    """
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.DatabaseError:  # a feature-scoped connection may refuse
        pass
    conn.close()


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Apply any migration files that have not run yet. Safe to call repeatedly."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {row[0] for row in conn.execute("SELECT name FROM schema_migrations")}
    ran: list[str] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name in applied:
            continue
        conn.executescript(path.read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)",
            (path.name, _utcnow()),
        )
        ran.append(path.name)
    conn.commit()
    return ran


# --- small helpers so callers do not sprinkle json.dumps around ------------------


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def upsert(
    conn: sqlite3.Connection,
    table: str,
    row: dict[str, Any],
    *,
    keys: Sequence[str],
) -> None:
    """Insert a row, replacing the non-key columns when the keys already exist."""
    cols = list(row)
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in keys)
    conflict = ", ".join(keys)
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    sql += f" ON CONFLICT({conflict}) DO UPDATE SET {updates}" if updates else " ON CONFLICT DO NOTHING"
    conn.execute(sql, [row[c] for c in cols])


def start_run(
    conn: sqlite3.Connection,
    kind: str,
    run_date: str,
    *,
    window_from: str | None = None,
    window_to: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO runs (kind, run_date, started_at, status, window_from, window_to)"
        " VALUES (?, ?, ?, 'running', ?, ?)",
        (kind, run_date, _utcnow(), window_from, window_to),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    stats: dict[str, Any] | None = None,
    notes: str | None = None,
) -> None:
    conn.execute(
        "UPDATE runs SET finished_at = ?, status = ?, stats = ?, notes = ? WHERE run_id = ?",
        (_utcnow(), status, dumps(stats or {}), notes, run_id),
    )
    conn.commit()


def last_successful_run(conn: sqlite3.Connection, kind: str) -> sqlite3.Row | None:
    cur = conn.execute(
        "SELECT * FROM runs WHERE kind = ? AND status IN ('ok', 'partial')"
        " ORDER BY started_at DESC LIMIT 1",
        (kind,),
    )
    return cur.fetchone()


def record_llm_calls(conn: sqlite3.Connection, run_id: int | None, calls: Iterable[dict]) -> None:
    for call in calls:
        conn.execute(
            "INSERT INTO llm_calls (run_id, ts, purpose, provider, model, input_tokens,"
            " output_tokens, cost_usd, ok, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                call.get("ts", _utcnow()),
                call.get("purpose", ""),
                call.get("provider", ""),
                call.get("model", ""),
                int(call.get("input_tokens", 0)),
                int(call.get("output_tokens", 0)),
                float(call.get("cost_usd", 0.0)),
                1 if call.get("ok", True) else 0,
                call.get("error"),
            ),
        )
    conn.commit()
