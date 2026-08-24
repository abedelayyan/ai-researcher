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

    @pytest.mark.parametrize(
        "text",
        [
            "We train on 8 A100 GPUs over 3 weeks.",
            "We evaluate 12 models against 4 baselines.",
            "The study covers 60 APIs versus 12 in prior benchmarks.",
            "We survey 40 papers published over 5 years.",
            "We train from 3 to 8 epochs.",
            "Data spans from 2019 to 2024.",
        ],
    )
    def test_ordinary_counts_are_not_read_as_a_jump(self, text):
        """Two counts in one sentence are not a benchmark result.

        Without the guard, "60 APIs versus 12" scores as a 48 point gain and pushes a
        dataset paper straight into the shortlist.
        """
        claims = structural.parse_benchmark_claims(text)
        assert claims.max_point_gain is None

    @pytest.mark.parametrize(
        "text,points",
        [
            ("Accuracy reaches 91.2% versus 84.5% for the previous best.", 6.7),
            ("reaching a 91% success rate against 47% for the previous best", 44.0),
            ("accuracy improves from 45.2 to 61.8 on MMLU", 16.6),
        ],
    )
    def test_real_comparisons_still_parse(self, text, points):
        assert structural.parse_benchmark_claims(text).max_point_gain == pytest.approx(points)

    def test_a_rate_needs_rate_words_to_count(self):
        with_units = structural.parse_benchmark_claims(
            "improves throughput from 340 to 1180 tokens per second"
        )
        bare = structural.parse_benchmark_claims("the corpus grew from 340 to 1180")
        assert with_units.max_relative_gain == pytest.approx(3.471)
        assert bare.max_relative_gain is None

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


class TestModelBudget:
    """A heavy arXiv day must not exhaust the free tier or run up a bill."""

    def test_triage_puts_the_strongest_structural_signals_first(self):
        from src.features.capability import triage

        papers = [{"arxiv_id": "thin"}, {"arxiv_id": "strong"}, {"arxiv_id": "middling"}]
        rows = {
            "thin": {},
            "strong": {"code_url": "https://github.com/a/b", "claims_numbers": 1,
                       "claims_system": 1, "max_point_gain": 20.0},
            "middling": {"claims_numbers": 1, "max_relative_gain": 3.0},
        }
        assert [p["arxiv_id"] for p in triage(papers, rows)] == ["strong", "middling", "thin"]

    def test_the_client_stops_at_the_call_budget(self):
        from src.llm.client import Client, HeuristicProvider

        client = Client(config={"llm": {"max_calls_per_run": 2, "fallback_order": [],
                                        "tiers": {"cheap": {}}}})
        recorded = []

        class Counting(HeuristicProvider):
            name = "fake"

            def available(self) -> bool:
                return True

            def complete(self, system, user, *, model, max_tokens, temperature, purpose):
                from src.llm.client import LLMResult

                recorded.append(purpose)
                return LLMResult(text="{}", provider="fake", model="m", purpose=purpose)

        client.providers = [Counting()]
        results = [client.complete("test", "s", "u") for _ in range(4)]
        assert len(recorded) == 2
        assert [r.ok for r in results] == [True, True, False, False]
        assert "budget" in (results[-1].error or "")
