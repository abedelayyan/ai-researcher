"""Digest rendering: the markdown a human reads and the pages Pages serves."""

from __future__ import annotations

import re
from pathlib import Path

from src.render import digest as render_digest
from src.score.rank import ScoredPaper


def make_paper(**overrides) -> ScoredPaper:
    row = {
        "arxiv_id": "2608.00001", "title": "A paper", "authors": '["Ada L", "Bo N"]',
        "categories": '["cs.LG"]', "abs_url": "https://arxiv.org/abs/2608.00001",
        "capability_total": 7, "cost_curve": 3, "cost_curve_why": "cuts inference cost 10x",
        "compute_band": "single_node", "compute_band_why": "matched '8 A100'",
        "max_point_gain": 12.0, "code_url": "https://github.com/a/b",
        "author_prior": 0.6, "author_prior_coverage": 1,
    }
    row.update(overrides)
    return ScoredPaper(arxiv_id=row["arxiv_id"], title=row["title"], score=0.7,
                       claim_score=0.65, row=row)


def context(**overrides) -> dict:
    base = dict(
        date="2026-08-21",
        buckets={"confidence": [make_paper()], "high_variance": []},
        summaries={"2608.00001": "Cuts long-context serving cost. Fits one node."},
        scored_count=12,
        config={"ingest": {"categories": ["cs.LG"]}, "outcomes": {"composite": {"horizon_days": 30}}},
        calibration={"k": 10, "buckets": {}, "labelled_total": 0},
    )
    base.update(overrides)
    return render_digest.build_context(**base)


class TestMarkdown:
    def test_both_buckets_always_appear(self):
        text = render_digest.render_markdown(context())
        assert "## Confidence" in text
        assert "## High variance" in text
        assert "No paper met the claim threshold today." in text

    def test_a_paper_carries_its_evidence(self):
        text = render_digest.render_markdown(context())
        assert "Cuts long-context serving cost." in text
        assert "12 points over the stated prior art" in text
        assert "single node" in text
        assert "github.com/a/b" in text
        assert "cost: cuts inference cost 10x" in text

    def test_an_unknown_prior_says_so_rather_than_showing_zero(self):
        paper = make_paper(author_prior=None, author_prior_coverage=0)
        text = render_digest.render_markdown(
            context(buckets={"confidence": [], "high_variance": [paper]})
        )
        assert "author prior unknown" in text

    def test_a_rules_scored_run_is_marked_provisional(self):
        paper = make_paper(capability_source="heuristic:rules")
        text = render_digest.render_markdown(context(buckets={"confidence": [paper],
                                                             "high_variance": []}))
        assert "provisional" in text

    def test_a_model_scored_run_carries_no_warning(self):
        paper = make_paper(capability_source="github_models:openai/gpt-4o-mini")
        text = render_digest.render_markdown(context(buckets={"confidence": [paper],
                                                             "high_variance": []}))
        assert "provisional" not in text


class TestSite:
    def test_archive_links_resolve_from_both_locations(self, tmp_path: Path):
        """index.html sits one level above the archive pages, so it needs a prefix."""
        config = {"render": {"digest_dir": str(tmp_path / "digest"),
                             "site_dir": str(tmp_path / "site")}}
        paths = render_digest.write(context(), config, recent_dates=["2026-08-21"])

        page = Path(paths["html"]).read_text(encoding="utf-8")
        index = Path(paths["index"]).read_text(encoding="utf-8")
        assert 'href="2026-08-21.html"' in page
        assert 'href="digest/2026-08-21.html"' in index

        # Every local link must resolve from the file it appears in.
        for text, base in ((page, Path(paths["html"]).parent), (index, Path(paths["index"]).parent)):
            local = [h for h in re.findall(r'href="([^"]+\.html)"', text)
                     if not h.startswith("http")]
            assert local
            for href in local:
                assert (base / href).exists(), f"{href} is broken from {base}"

    def test_writing_twice_overwrites_rather_than_appending(self, tmp_path: Path):
        config = {"render": {"digest_dir": str(tmp_path / "digest"),
                             "site_dir": str(tmp_path / "site")}}
        first = render_digest.write(context(), config)
        second = render_digest.write(context(), config)
        assert first["markdown"] == second["markdown"]
        assert Path(first["markdown"]).read_text().count("# Signal Zero") == 1
