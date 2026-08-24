"""End to end on a fixture, with no network and no model.

This is the run that has to work on a bad day: arXiv reachable, everything else down.
It also checks the property the prediction log depends on, which is that every paper
scored gets a row and not only the shortlist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.arxiv import parse_atom
from src.pipelines import daily
from src.store import db
from src.util.config import db_path


@pytest.fixture
def offline_run(monkeypatch, config, sample_atom):
    papers = parse_atom(sample_atom)
    monkeypatch.setattr(daily, "_ingest", lambda fetcher, cfg, start, end: (papers, []))

    # Two authors have a record, everyone else is unknown.
    conn = db.connect(db_path(config))
    for name, prior in (("wei zhang", 0.72), ("marta silva", 0.55)):
        conn.execute(
            "INSERT INTO author_priors (author_key, cutoff_date, display_name, works_count,"
            " citations_total, citations_per_year, prior, coverage, computed_at)"
            " VALUES (?, '', ?, 12, 400, 30.0, ?, 1, ?)",
            (name, name.title(), prior, "2026-08-20T00:00:00+00:00"),
        )
    conn.commit()
    conn.close()
    return daily.run(config=config, run_date="2026-08-21", offline=True)


class TestDailyRun:
    def test_it_writes_a_digest(self, offline_run, config):
        markdown = Path(offline_run.paths["markdown"])
        assert markdown.exists()
        text = markdown.read_text(encoding="utf-8")
        assert "## Confidence" in text and "## High variance" in text
        assert Path(offline_run.paths["index"]).exists()

    def test_every_paper_scored_gets_a_row(self, offline_run, config):
        """Including the low ones. Without the negatives the log cannot be calibrated."""
        conn = db.connect(db_path(config))
        papers = conn.execute("SELECT count(*) FROM papers").fetchone()[0]
        logged = conn.execute("SELECT count(*) FROM paper_scores").fetchone()[0]
        shortlisted = conn.execute(
            "SELECT count(*) FROM paper_scores WHERE shortlisted = 1"
        ).fetchone()[0]
        conn.close()
        assert papers == 12
        assert logged == papers
        assert 0 < shortlisted < papers

    def test_features_are_extracted_for_every_paper(self, offline_run, config):
        conn = db.connect(db_path(config))
        rows = conn.execute(
            "SELECT arxiv_id, compute_band, capability_source FROM paper_features"
        ).fetchall()
        conn.close()
        assert len(rows) == 12
        assert all(row["capability_source"] == "heuristic:rules" for row in rows)
        assert {row["compute_band"] for row in rows} & {"consumer", "single_node", "frontier"}

    def test_unknown_authors_keep_a_null_prior(self, offline_run, config):
        conn = db.connect(db_path(config))
        rows = conn.execute(
            "SELECT author_prior, author_prior_coverage FROM paper_features"
        ).fetchall()
        conn.close()
        assert any(row["author_prior"] is None for row in rows)
        assert all(row["author_prior"] is not None or row["author_prior_coverage"] == 0
                   for row in rows)

    def test_the_frontier_scale_paper_is_not_shortlisted_above_deployable_work(
        self, offline_run, config
    ):
        """Work needing a frontier cluster is a lab result, not a startup seed."""
        conn = db.connect(db_path(config))
        row = conn.execute(
            "SELECT s.score FROM paper_scores s WHERE s.arxiv_id = '2608.01004'"
        ).fetchone()
        consumer = conn.execute(
            "SELECT s.score FROM paper_scores s WHERE s.arxiv_id = '2608.01001'"
        ).fetchone()
        conn.close()
        assert row["score"] < consumer["score"]

    def test_rerunning_the_same_day_is_a_no_op(self, offline_run, config):
        second = daily.run(config=config, run_date="2026-08-21", offline=True, skip_ingest=True)
        assert second.extracted["extracted"] == 0
        conn = db.connect(db_path(config))
        logged = conn.execute("SELECT count(*) FROM paper_scores").fetchone()[0]
        conn.close()
        assert logged == 12

    def test_a_truncated_ingest_does_not_move_the_window_on(self, monkeypatch, config, sample_atom):
        """A partial run is re-covered by the next one rather than skipped over."""
        from src.ingest.arxiv import parse_atom

        papers = parse_atom(sample_atom)
        monkeypatch.setattr(
            daily, "_ingest", lambda fetcher, cfg, start, end: (papers, ["page 2: 429"])
        )
        daily.run(config=config, run_date="2026-08-21", offline=True)
        conn = db.connect(db_path(config))
        row = conn.execute("SELECT * FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
        clean = db.last_successful_run(conn, "daily")
        db.close(conn)
        assert row["status"] == "partial"
        assert "429" in (row["notes"] or "")
        assert clean is None

    def test_the_run_is_recorded(self, offline_run, config):
        conn = db.connect(db_path(config))
        row = conn.execute("SELECT * FROM runs WHERE kind = 'daily' ORDER BY run_id").fetchone()
        conn.close()
        assert row["status"] in ("ok", "partial")
        assert row["window_from"] and row["window_to"]


class TestDigestStyle:
    """Section 17 of the spec is a house style, so it is worth a test."""

    BANNED_WORDS = ("leverage", "seamless", "revolutionary", "remarkable", "robust", "deepen")

    def test_no_em_dashes_or_filler(self, offline_run):
        text = Path(offline_run.paths["markdown"]).read_text(encoding="utf-8")
        assert "—" not in text
        lowered = text.lower()
        for word in self.BANNED_WORDS:
            assert word not in lowered, f"digest contains {word!r}"
        assert "not just" not in lowered
