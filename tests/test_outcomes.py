"""Outcome collection and calibration: the measuring half of the system."""

from __future__ import annotations

from pathlib import Path

from src.outcomes import calibration, collect
from src.outcomes.github import repo_from_url
from src.store import db
from src.util.config import load as load_config


def seed(conn, arxiv_id: str, announced: str) -> None:
    conn.execute(
        "INSERT INTO papers (arxiv_id, title, abstract, announced_date, source, ingested_at)"
        " VALUES (?, 'title', 'abstract', ?, 'test', '2026-01-01T00:00:00+00:00')",
        (arxiv_id, announced),
    )


class TestDueObservations:
    def test_a_horizon_that_has_arrived_is_due(self, tmp_path: Path):
        with db.connect(tmp_path / "o.db") as conn:
            seed(conn, "2601.00001", "2026-01-01")
            due = collect.due_observations(conn, "2026-01-09", [7, 30, 90])
        assert [horizon for _, horizon in due] == [7]

    def test_an_overdue_observation_is_still_collected(self, tmp_path: Path):
        """A missed run must not lose the observation."""
        with db.connect(tmp_path / "o.db") as conn:
            seed(conn, "2601.00002", "2026-01-01")
            due = collect.due_observations(conn, "2026-02-20", [7, 30, 90])
        assert sorted(h for _, h in due) == [7, 30]

    def test_an_existing_observation_is_not_repeated(self, tmp_path: Path):
        with db.connect(tmp_path / "o.db") as conn:
            seed(conn, "2601.00003", "2026-01-01")
            conn.execute(
                "INSERT INTO paper_outcomes (arxiv_id, horizon_days, observed_at)"
                " VALUES ('2601.00003', 7, '2026-01-08')"
            )
            due = collect.due_observations(conn, "2026-01-09", [7, 30, 90])
        assert due == []


class TestComposite:
    def test_signals_combine_and_saturate(self):
        config = load_config()
        low, _ = calibration.composite({"hf_upvotes": 0, "github_stars": 0}, config)
        high, parts = calibration.composite(
            {"hf_upvotes": 200, "github_stars": 4000, "citations_openalex": 40, "hn_points": 500},
            config,
        )
        assert low == 0.0
        assert high == 1.0
        assert all(value == 1.0 for value in parts.values())

    def test_the_larger_citation_source_wins(self):
        config = load_config()
        value, parts = calibration.composite(
            {"citations_openalex": 2, "citations_s2": 9}, config
        )
        assert parts["citations"] > 0.8


class TestCalibration:
    def _populated(self, tmp_path: Path):
        conn = db.connect(tmp_path / "c.db")
        for index in range(20):
            arxiv_id = f"2601.{index:05d}"
            seed(conn, arxiv_id, "2026-01-01")
            # The scorer put the first five at the top, and four of them landed.
            score = 0.9 - index * 0.02
            hit = 1 if index < 4 else 0
            conn.execute(
                "INSERT INTO paper_scores (arxiv_id, run_date, scorer_version, score,"
                " claim_score, bucket, rank, components, shortlisted, created_at)"
                " VALUES (?, '2026-01-01', 'v1', ?, ?, ?, ?, '{}', ?, '2026-01-01T00:00:00+00:00')",
                (arxiv_id, score, score, "confidence" if index < 5 else None,
                 index + 1 if index < 5 else None, 1 if index < 5 else 0),
            )
            conn.execute(
                "INSERT INTO outcome_labels (arxiv_id, horizon_days, composite, hit, parts, computed_at)"
                " VALUES (?, 30, ?, ?, '{}', '2026-02-01T00:00:00+00:00')",
                (arxiv_id, 0.5 if hit else 0.05, hit),
            )
        conn.commit()
        return conn

    def test_precision_at_k_beats_the_base_rate_when_the_scorer_is_right(self, tmp_path: Path):
        conn = self._populated(tmp_path)
        rows = calibration.precision_at_k(conn, k=5)
        assert len(rows) == 1
        assert rows[0].precision == 0.8
        assert rows[0].base_rate == 0.2
        assert rows[0].lift == 4.0

    def test_labels_are_derived_from_observations(self, tmp_path: Path):
        config = load_config()
        with db.connect(tmp_path / "l.db") as conn:
            seed(conn, "2601.00099", "2026-01-01")
            conn.execute(
                "INSERT INTO paper_outcomes (arxiv_id, horizon_days, observed_at, hf_upvotes,"
                " github_stars, citations_openalex, hn_points)"
                " VALUES ('2601.00099', 30, '2026-01-31', 90, 900, 12, 200)"
            )
            assert calibration.compute_labels(conn, config) == 1
            row = conn.execute("SELECT * FROM outcome_labels").fetchone()
        assert row["hit"] == 1
        assert row["composite"] == 1.0

    def test_missed_winners_are_reported(self, tmp_path: Path):
        conn = self._populated(tmp_path)
        conn.execute("UPDATE outcome_labels SET hit = 1, composite = 0.9 WHERE arxiv_id = '2601.00015'")
        conn.commit()
        missed = calibration.missed_winners(conn)
        assert missed and missed[0]["arxiv_id"] == "2601.00015"


def test_repo_url_parsing():
    assert repo_from_url("https://github.com/foo/bar.git") == ("foo", "bar")
    assert repo_from_url("https://example.com/foo") is None
