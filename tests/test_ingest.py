"""arXiv parsing and window handling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.ingest.arxiv import _split_days, build_query, parse_atom
from src.ingest.models import split_arxiv_id
from src.ingest.store import save_papers
from src.store import db
from src.util.dates import lookback_window


class TestParsing:
    def test_every_entry_becomes_a_paper(self, sample_atom):
        papers = parse_atom(sample_atom)
        assert len(papers) == 12

    def test_metadata_survives(self, sample_atom):
        paper = {p.arxiv_id: p for p in parse_atom(sample_atom)}["2608.01001"]
        assert paper.version == 1
        assert paper.primary_category == "cs.LG"
        assert paper.categories == ["cs.LG", "cs.CL"]
        assert paper.authors[0] == "Wei Zhang"
        assert paper.announced_date == "2026-08-20"
        assert "github.com/flashkv" in (paper.comments or "")

    def test_version_suffixes_are_split_off(self):
        assert split_arxiv_id("http://arxiv.org/abs/2408.01234v2") == ("2408.01234", 2)
        assert split_arxiv_id("2408.01234") == ("2408.01234", None)


class TestQueryWindow:
    def test_query_covers_every_category_and_the_window(self):
        start = datetime(2026, 8, 20, tzinfo=UTC)
        end = datetime(2026, 8, 21, tzinfo=UTC)
        query = build_query(["cs.AI", "cs.LG"], start, end)
        assert "cat:cs.AI OR cat:cs.LG" in query
        assert "submittedDate:[202608200000 TO 202608210000]" in query

    def test_long_windows_are_cut_at_day_boundaries(self):
        chunks = _split_days(
            datetime(2026, 8, 20, 14, tzinfo=UTC),
            datetime(2026, 8, 23, 2, tzinfo=UTC),
        )
        assert len(chunks) == 4
        assert all(end - start <= timedelta(days=1) for start, end in chunks)

    def test_a_missed_run_widens_the_window(self):
        now = datetime(2026, 8, 23, 1, 0, tzinfo=UTC)
        normal_start, _ = lookback_window(None, now=now, default_hours=30)
        recovered_start, _ = lookback_window("2026-08-19T01:00:00+00:00", now=now, default_hours=30)
        assert recovered_start < normal_start

    def test_the_widened_window_is_capped(self):
        now = datetime(2026, 8, 23, 1, 0, tzinfo=UTC)
        start, end = lookback_window(
            "2025-01-01T00:00:00+00:00", now=now, default_hours=30, max_hours=168
        )
        assert (end - start) <= timedelta(hours=168)


class TestStore:
    def test_saving_twice_adds_nothing(self, tmp_path, sample_atom):
        papers = parse_atom(sample_atom)
        with db.connect(tmp_path / "i.db") as conn:
            first = save_papers(conn, papers)
            second = save_papers(conn, papers)
            total = conn.execute("SELECT count(*) FROM papers").fetchone()[0]
        assert first["new"] == 12
        assert second["new"] == 0
        assert total == 12

    def test_corpus_terms_record_first_appearance(self, tmp_path, sample_atom):
        papers = parse_atom(sample_atom)
        with db.connect(tmp_path / "i.db") as conn:
            save_papers(conn, papers)
            row = conn.execute(
                "SELECT * FROM corpus_terms WHERE term = 'flashkv'"
            ).fetchone()
        assert row is not None
        assert row["first_seen_date"] == "2026-08-20"
