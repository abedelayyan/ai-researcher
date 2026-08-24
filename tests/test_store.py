"""The storage layer: migrations, upserts and the committed-file guarantees."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.store import db


class TestMigrations:
    def test_first_connect_applies_them(self, tmp_path: Path):
        conn = db.connect(tmp_path / "m.db")
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        db.close(conn)
        assert {"papers", "paper_features", "paper_scores", "paper_outcomes",
                "outcome_labels", "runs", "llm_calls"} <= tables

    def test_running_them_again_changes_nothing(self, tmp_path: Path):
        path = tmp_path / "m.db"
        conn = db.connect(path)
        assert db.migrate(conn) == []
        db.close(conn)
        conn = db.connect(path)
        assert db.migrate(conn) == []
        db.close(conn)

    def test_foreign_keys_are_enforced(self, tmp_path: Path):
        with db.connect(tmp_path / "m.db") as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO paper_features (arxiv_id, features_version, extracted_at)"
                    " VALUES ('nope', 'v1', 'now')"
                )


class TestUpsert:
    def test_it_replaces_the_non_key_columns(self, tmp_path: Path):
        with db.connect(tmp_path / "u.db") as conn:
            for title in ("first", "second"):
                db.upsert(conn, "papers", {
                    "arxiv_id": "2608.00001", "title": title, "abstract": "a",
                    "announced_date": "2026-08-20", "source": "test",
                    "ingested_at": "2026-08-20T00:00:00+00:00",
                }, keys=["arxiv_id"])
            rows = conn.execute("SELECT title FROM papers").fetchall()
        assert [row[0] for row in rows] == ["second"]

    def test_composite_keys_work(self, tmp_path: Path):
        with db.connect(tmp_path / "u.db") as conn:
            conn.execute(
                "INSERT INTO papers (arxiv_id, title, abstract, announced_date, source, ingested_at)"
                " VALUES ('2608.00002', 't', 'a', '2026-08-20', 'test', 'now')"
            )
            for stars in (10, 40):
                db.upsert(conn, "paper_outcomes", {
                    "arxiv_id": "2608.00002", "horizon_days": 7,
                    "observed_at": "2026-08-27", "github_stars": stars,
                }, keys=["arxiv_id", "horizon_days"])
            row = conn.execute("SELECT github_stars FROM paper_outcomes").fetchone()
        assert row[0] == 40


class TestRuns:
    def test_a_finished_run_becomes_the_last_successful_one(self, tmp_path: Path):
        with db.connect(tmp_path / "r.db") as conn:
            run_id = db.start_run(conn, "daily", "2026-08-21")
            assert db.last_successful_run(conn, "daily") is None
            db.finish_run(conn, run_id, "ok", {"scored": 12})
            row = db.last_successful_run(conn, "daily")
        assert row["status"] == "ok"
        assert db.loads(row["stats"])["scored"] == 12

    def test_a_failed_run_does_not_move_the_window(self, tmp_path: Path):
        with db.connect(tmp_path / "r.db") as conn:
            db.finish_run(conn, db.start_run(conn, "daily", "2026-08-21"), "failed")
            assert db.last_successful_run(conn, "daily") is None


def test_close_leaves_no_write_ahead_log_behind(tmp_path: Path):
    """The database file is committed, so a write stranded in the WAL would be lost."""
    path = tmp_path / "w.db"
    conn = db.connect(path)
    conn.execute(
        "INSERT INTO papers (arxiv_id, title, abstract, announced_date, source, ingested_at)"
        " VALUES ('2608.00003', 't', 'a', '2026-08-20', 'test', 'now')"
    )
    conn.commit()
    db.close(conn)
    wal = path.with_name(path.name + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0
    reopened = sqlite3.connect(path)
    assert reopened.execute("SELECT count(*) FROM papers").fetchone()[0] == 1
    reopened.close()
