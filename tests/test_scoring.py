"""Ranking rules: how components combine and how the two buckets are drawn."""

from __future__ import annotations

import pytest

from src.score import rank


WEIGHTS = {
    "capability_delta": 0.40, "benchmark_jump": 0.20, "compute_band": 0.15,
    "code_link": 0.10, "structural": 0.05, "author_prior": 0.10,
}


def make_row(**overrides):
    row = {
        "arxiv_id": "2608.00001", "title": "A paper", "capability_total": 6,
        "max_point_gain": None, "max_relative_gain": None, "compute_band": "single_node",
        "code_url": None, "code_link_source": None, "cross_listed": 0, "claims_system": 0,
        "application_domain": None, "novel_terms": "[]", "author_prior": 0.5,
        "author_prior_coverage": 1,
    }
    row.update(overrides)
    return row


class TestComponents:
    def test_benchmark_component_separates_a_small_gain_from_a_large_one(self):
        cfg = {"benchmark_saturation_points": 20.0, "benchmark_saturation_multiple": 5.0}
        assert rank.benchmark_component(0.4, None, cfg) < rank.benchmark_component(18.0, None, cfg)

    def test_a_large_multiple_saturates(self):
        cfg = {"benchmark_saturation_points": 20.0, "benchmark_saturation_multiple": 5.0}
        assert rank.benchmark_component(None, 12.0, cfg) == 1.0

    def test_a_promised_repo_counts_for_less_than_a_link(self):
        assert rank.code_component({"code_url": "https://github.com/a/b"}) == 1.0
        assert rank.code_component({"code_link_source": "promise"}) == 0.5
        assert rank.code_component({}) == 0.0

    def test_frontier_compute_is_penalised(self):
        table = {"consumer": 1.0, "frontier": 0.05, "unknown": 0.4}
        assert rank._band_score("frontier", table) < rank._band_score("consumer", table)


class TestCombine:
    def test_a_null_prior_redistributes_rather_than_scoring_zero(self):
        components = {"capability_delta": 0.5, "benchmark_jump": 0.5, "compute_band": 0.5,
                      "code_link": 0.5, "structural": 0.5}
        unknown, unknown_claim = rank.combine(components, None, WEIGHTS)
        zeroed, _ = rank.combine(components, 0.0, WEIGHTS)
        assert unknown == unknown_claim
        assert unknown > zeroed

    def test_the_claim_score_ignores_the_prior_entirely(self):
        components = {"capability_delta": 0.8, "benchmark_jump": 0.6, "compute_band": 0.8,
                      "code_link": 1.0, "structural": 0.4}
        strong, claim_strong = rank.combine(components, 0.9, WEIGHTS)
        weak, claim_weak = rank.combine(components, 0.1, WEIGHTS)
        assert strong > weak
        assert claim_strong == claim_weak


class TestBuckets:
    def test_the_buckets_never_merge_and_never_overlap(self):
        rows = [
            make_row(arxiv_id="known-1", author_prior=0.8, capability_total=9),
            make_row(arxiv_id="known-2", author_prior=0.6, capability_total=8),
            make_row(arxiv_id="unknown-1", author_prior=None, author_prior_coverage=0,
                     capability_total=12, max_point_gain=30.0, compute_band="consumer",
                     code_url="https://github.com/a/b"),
        ]
        scored = rank.score_rows(rows, {"score": {"weights": WEIGHTS}})
        buckets = rank.assign_buckets(scored, {"score": {"buckets": {
            "confidence": {"size": 8},
            "high_variance": {"size": 6, "max_author_prior": 0.35, "min_claim_score": 0.45},
        }}})
        confidence = {p.arxiv_id for p in buckets["confidence"]}
        variance = {p.arxiv_id for p in buckets["high_variance"]}
        assert confidence == {"known-1", "known-2"}
        assert variance == {"unknown-1"}
        assert not confidence & variance

    def test_an_unknown_author_with_a_big_claim_is_not_buried(self):
        """The point of the second bucket. A merged ranking would lose this paper."""
        rows = [
            make_row(arxiv_id=f"lab-{i}", author_prior=0.95, capability_total=11,
                     max_point_gain=25.0, compute_band="consumer", claims_system=1,
                     code_url="https://github.com/lab/x")
            for i in range(8)
        ] + [
            make_row(arxiv_id="outsider", author_prior=None, author_prior_coverage=0,
                     capability_total=9, max_relative_gain=6.0, compute_band="single_node",
                     code_url="https://github.com/a/b"),
        ]
        config = {"score": {"weights": WEIGHTS, "buckets": {
            "confidence": {"size": 8},
            "high_variance": {"size": 6, "max_author_prior": 0.35, "min_claim_score": 0.45},
        }}}
        scored = rank.score_rows(rows, config)
        merged_top8 = {p.arxiv_id for p in sorted(scored, key=lambda p: p.score, reverse=True)[:8]}
        buckets = rank.assign_buckets(scored, config)
        assert "outsider" not in merged_top8
        assert "outsider" in {p.arxiv_id for p in buckets["high_variance"]}

    def test_ranks_are_assigned_within_each_bucket(self):
        rows = [make_row(arxiv_id=f"p{i}", author_prior=0.9 - i / 10) for i in range(3)]
        config = {"score": {"weights": WEIGHTS, "buckets": {"confidence": {"size": 8}}}}
        buckets = rank.assign_buckets(rank.score_rows(rows, config), config)
        assert [p.rank for p in buckets["confidence"]] == [1, 2, 3]
        assert buckets["confidence"][0].score >= buckets["confidence"][1].score
