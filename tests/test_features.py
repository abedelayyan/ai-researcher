"""Day-zero feature extraction: the parts that must behave the same on every replay."""

from __future__ import annotations

import pytest

from src.features import structural
from src.features.author_prior import citations_by, select_authors, summarise_works
from src.features.capability import heuristic_scores, parse_response


class TestCodeLink:
    def test_finds_a_link_in_the_abstract(self):
        link = structural.find_code_link("Code at https://github.com/foo/bar.", None, [])
        assert link.url == "https://github.com/foo/bar"
        assert link.source == "abstract"

    def test_prefers_the_comments_field(self):
        link = structural.find_code_link("no link here", "Code: https://gitlab.com/a/b", [])
        assert link.source == "comments"

    def test_records_a_promise_without_a_url(self):
        link = structural.find_code_link("Code will be released upon acceptance.", None, [])
        assert link.present and link.url is None and link.source == "promise"

    def test_absent(self):
        assert not structural.find_code_link("A theory paper.", None, []).present


class TestBenchmarkClaims:
    def test_from_to(self):
        claims = structural.parse_benchmark_claims("accuracy improves from 45.2 to 61.8")
        assert claims.max_point_gain == pytest.approx(16.6)

    def test_multiple(self):
        claims = structural.parse_benchmark_claims("achieves 12x cheaper inference")
        assert claims.max_relative_gain == pytest.approx(12.0)

    def test_percentage_reduction_becomes_a_multiple(self):
        claims = structural.parse_benchmark_claims("reduces memory by 90%")
        assert claims.max_relative_gain == pytest.approx(10.0)

    def test_no_numbers_is_recorded(self):
        claims = structural.parse_benchmark_claims("We prove a tighter bound.")
        assert not claims.has_numbers
        assert claims.note == "no numeric claim in the abstract"

    def test_small_and_large_gains_are_different_species(self):
        small = structural.parse_benchmark_claims("improves by 0.4 points")
        large = structural.parse_benchmark_claims("improves by 30 points")
        assert small.max_point_gain < large.max_point_gain


class TestComputeBand:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("runs on a single RTX 4090", "consumer"),
            ("trained on 8 A100 GPUs", "single_node"),
            ("we use 64 A100 GPUs", "small_cluster"),
            ("training used 512 A100 GPUs for three weeks", "frontier"),
            ("we prove a bound", "unknown"),
        ],
    )
    def test_bands(self, text, expected):
        band, _ = structural.estimate_compute_band(text)
        assert band == expected

    def test_largest_requirement_wins(self):
        band, _ = structural.estimate_compute_band(
            "inference runs on a single RTX 4090 after training on 1024 GPUs"
        )
        assert band == "frontier"


class TestAuthorPrior:
    def test_citations_are_bounded_by_the_cutoff_year(self):
        work = {"counts_by_year": [
            {"year": 2023, "cited_by_count": 40},
            {"year": 2024, "cited_by_count": 90},
            {"year": 2026, "cited_by_count": 500},
        ]}
        assert citations_by(work, 2024) == 130

    def test_an_unknown_author_gets_a_null_prior_not_a_zero(self):
        prior = summarise_works("New Person", "new person", [], 2026, saturation=40, min_works=3)
        assert prior.prior is None
        assert prior.coverage is False

    def test_a_cited_author_scores(self):
        works = [
            {"publication_year": 2024, "counts_by_year": [{"year": 2025, "cited_by_count": 120}]},
            {"publication_year": 2023, "counts_by_year": [{"year": 2024, "cited_by_count": 60}]},
            {"publication_year": 2022, "counts_by_year": [{"year": 2023, "cited_by_count": 30}]},
        ]
        prior = summarise_works("Known", "known", works, 2026, saturation=40, min_works=3)
        assert prior.prior is not None and prior.prior > 0
        assert prior.coverage is True

    def test_both_ends_of_the_author_list_are_kept(self):
        authors = [f"A{i}" for i in range(10)]
        picked = select_authors(authors, 3)
        assert picked[0] == "A0" and picked[-1] == "A9" and len(picked) == 3


class TestCapability:
    def test_scores_are_clamped_and_bands_validated(self):
        score = parse_response(
            '{"cost_curve": "5", "constraint_removal": -2, "compute_band": "nonsense"}',
            source="test:model", prompt_version="v1",
        )
        assert score.scores["cost_curve"] == 3
        assert score.scores["constraint_removal"] == 0
        assert score.compute_band == "unknown"

    def test_the_rules_fallback_stays_below_the_top_of_each_axis(self):
        text = " ".join(["training-free"] * 5 + ["3x faster cheaper distill quantise"] * 5)
        score = heuristic_scores(text)
        assert max(score.scores.values()) <= 2
        assert score.source == "heuristic:rules"

    def test_a_paper_that_moves_nothing_scores_zero(self):
        score = heuristic_scores("We prove a tighter convergence bound.")
        assert score.total == 0
