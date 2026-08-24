"""Combine day-zero features into two rankings.

Every component here is computable on the morning a paper appears. Nothing that
depends on how the paper was received exists in this module, and the connection it is
handed cannot read those tables anyway.

Two rankings, never merged:

* **Confidence** weights the author prior in. These are the papers most likely to matter.
* **High variance** takes papers with a null or low prior and a large claim, ranked on
  claim size alone. Either noise or the thing worth finding. A single prior-weighted
  ranking would bury them under known labs every time, which is why the split exists.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..store import db
from ..util import dates
from ..util import logging as log
from ..util.config import load as load_config

LOG = log.get("score.rank")

MAX_CAPABILITY = 12.0  # four axes, three points each


@dataclass
class ScoredPaper:
    arxiv_id: str
    title: str
    score: float
    claim_score: float
    components: dict = field(default_factory=dict)
    bucket: str | None = None
    rank: int | None = None
    row: dict = field(default_factory=dict)


def _band_score(band: str | None, table: dict) -> float:
    return float(table.get(band or "unknown", table.get("unknown", 0.4)))


def benchmark_component(points: float | None, multiple: float | None, cfg: dict) -> float:
    """A 0.4 point gain and a 30 point gain are different species of result."""
    point_ceiling = float(cfg.get("benchmark_saturation_points", 20.0))
    multiple_ceiling = float(cfg.get("benchmark_saturation_multiple", 5.0))
    from_points = min(1.0, (points or 0.0) / point_ceiling) if points else 0.0
    from_multiple = 0.0
    if multiple and multiple > 1.0:
        from_multiple = min(1.0, (multiple - 1.0) / max(0.001, multiple_ceiling - 1.0))
    return max(from_points, from_multiple)


def structural_component(row: dict) -> float:
    """Small signals, worth little individually."""
    parts = [
        0.30 if row.get("cross_listed") else 0.0,
        0.30 if row.get("claims_system") else 0.0,
        0.20 if row.get("application_domain") else 0.0,
        0.20 if db.loads(row.get("novel_terms"), []) else 0.0,
    ]
    return min(1.0, sum(parts))


def code_component(row: dict) -> float:
    if row.get("code_url"):
        return 1.0
    if row.get("code_link_source") == "promise":
        return 0.5
    return 0.0


def components_for(row: dict, cfg: dict) -> dict[str, float]:
    capability_total = row.get("capability_total")
    capability = (capability_total or 0) / MAX_CAPABILITY
    return {
        "capability_delta": round(capability, 4),
        "benchmark_jump": round(
            benchmark_component(row.get("max_point_gain"), row.get("max_relative_gain"), cfg), 4
        ),
        "compute_band": round(_band_score(row.get("compute_band"), cfg.get("compute_band_scores", {})), 4),
        "code_link": round(code_component(row), 4),
        "structural": round(structural_component(row), 4),
    }


def combine(components: dict[str, float], prior: float | None, weights: dict) -> tuple[float, float]:
    """Return (score with the prior, claim score without it).

    A null prior does not score zero. The prior's weight is redistributed across the
    other components instead, so an unknown author is neither rewarded nor punished.
    """
    prior_weight = float(weights.get("author_prior", 0.0))
    claim_weights = {k: float(v) for k, v in weights.items() if k != "author_prior"}
    claim_total = sum(claim_weights.values()) or 1.0
    claim_score = sum(components.get(k, 0.0) * w for k, w in claim_weights.items()) / claim_total

    if prior is None:
        return round(claim_score, 4), round(claim_score, 4)
    full = claim_score * (1.0 - prior_weight) + prior * prior_weight
    return round(full, 4), round(claim_score, 4)


def score_rows(rows: Sequence[sqlite3.Row | dict], config: dict | None = None) -> list[ScoredPaper]:
    config = config or load_config()
    cfg = config.get("score", {})
    weights = cfg.get("weights", {})
    scored: list[ScoredPaper] = []
    for raw in rows:
        row = dict(raw)
        components = components_for(row, cfg)
        prior = row.get("author_prior")
        score, claim = combine(components, prior, weights)
        components["author_prior"] = prior
        scored.append(
            ScoredPaper(
                arxiv_id=row["arxiv_id"],
                title=row.get("title", ""),
                score=score,
                claim_score=claim,
                components=components,
                row=row,
            )
        )
    return scored


def assign_buckets(scored: Sequence[ScoredPaper], config: dict | None = None) -> dict[str, list[ScoredPaper]]:
    config = config or load_config()
    buckets = config.get("score", {}).get("buckets", {})
    conf_cfg = buckets.get("confidence", {})
    var_cfg = buckets.get("high_variance", {})

    confident = [p for p in scored if p.row.get("author_prior_coverage")]
    confident.sort(key=lambda p: p.score, reverse=True)
    confidence = confident[: int(conf_cfg.get("size", 8))]
    chosen = {p.arxiv_id for p in confidence}

    max_prior = float(var_cfg.get("max_author_prior", 0.35))
    min_claim = float(var_cfg.get("min_claim_score", 0.45))
    candidates = [
        p for p in scored
        if p.arxiv_id not in chosen
        and (p.row.get("author_prior") is None or (p.row.get("author_prior") or 0.0) <= max_prior)
        and p.claim_score >= min_claim
    ]
    candidates.sort(key=lambda p: p.claim_score, reverse=True)
    high_variance = candidates[: int(var_cfg.get("size", 6))]

    for index, paper in enumerate(confidence, start=1):
        paper.bucket, paper.rank = "confidence", index
    for index, paper in enumerate(high_variance, start=1):
        paper.bucket, paper.rank = "high_variance", index
    return {"confidence": confidence, "high_variance": high_variance}


def persist(
    conn: sqlite3.Connection,
    scored: Sequence[ScoredPaper],
    run_date: str,
    config: dict | None = None,
) -> int:
    """Write the prediction log.

    Every paper scored gets a row, including the ones scored low. Without the negatives
    the dataset cannot be calibrated later.
    """
    config = config or load_config()
    version = config.get("score", {}).get("scorer_version", "v1")
    now = dates.utcnow().isoformat(timespec="seconds")
    for paper in scored:
        db.upsert(
            conn,
            "paper_scores",
            {
                "arxiv_id": paper.arxiv_id,
                "run_date": run_date,
                "scorer_version": version,
                "score": paper.score,
                "claim_score": paper.claim_score,
                "bucket": paper.bucket,
                "rank": paper.rank,
                "components": db.dumps(paper.components),
                "shortlisted": 1 if paper.bucket else 0,
                "summary": None,
                "created_at": now,
            },
            keys=["arxiv_id", "run_date", "scorer_version"],
        )
    conn.commit()
    return len(scored)


def _load_rows(conn: sqlite3.Connection, where: str, params: Sequence[object]) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT p.arxiv_id, p.title, p.abstract, p.authors, p.categories, p.primary_category,"
        " p.abs_url, p.announced_date, f.* FROM papers p JOIN paper_features f USING (arxiv_id)"
        f" WHERE {where}",
        list(params),
    ).fetchall()


def rank_ids(
    conn: sqlite3.Connection,
    arxiv_ids: Sequence[str],
    run_date: str,
    config: dict | None = None,
    *,
    persist_log: bool = True,
) -> dict[str, list[ScoredPaper]]:
    """Score a pool of papers and split it into the two buckets.

    The pool is whatever the run ingested, which after a missed day may span more than
    one announcement date. That is deliberate: a missed run should produce a fuller
    digest, not a gap.
    """
    if not arxiv_ids:
        return {"confidence": [], "high_variance": []}
    ids = list(arxiv_ids)
    placeholders = ", ".join("?" for _ in ids)
    rows = _load_rows(conn, f"p.arxiv_id IN ({placeholders})", ids)
    scored = score_rows(rows, config)
    result = assign_buckets(scored, config)
    if persist_log and scored:
        persist(conn, scored, run_date, config)
    LOG.info(
        "run %s: scored %d papers, %d confidence, %d high variance",
        run_date, len(scored), len(result["confidence"]), len(result["high_variance"]),
    )
    return result


def rank_day(
    conn: sqlite3.Connection,
    announced_date: str,
    config: dict | None = None,
    *,
    run_date: str | None = None,
    persist_log: bool = True,
) -> dict[str, list[ScoredPaper]]:
    """Score every paper announced on one date."""
    rows = _load_rows(conn, "p.announced_date = ?", [announced_date])
    scored = score_rows(rows, config)
    result = assign_buckets(scored, config)
    if persist_log and scored:
        persist(conn, scored, run_date or announced_date, config)
    LOG.info(
        "%s: scored %d papers, %d confidence, %d high variance",
        announced_date, len(scored), len(result["confidence"]), len(result["high_variance"]),
    )
    return result


def save_summary(conn: sqlite3.Connection, arxiv_id: str, run_date: str, summary: str,
                 config: dict | None = None) -> None:
    config = config or load_config()
    version = config.get("score", {}).get("scorer_version", "v1")
    conn.execute(
        "UPDATE paper_scores SET summary = ? WHERE arxiv_id = ? AND run_date = ? AND scorer_version = ?",
        (summary, arxiv_id, run_date, version),
    )
    conn.commit()
