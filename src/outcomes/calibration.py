"""Was the day-zero score right?

The composite label collapses the outcome signals into one number per paper once the
t+30 observation is in. Precision-at-k over the prediction log then says whether the
shortlist was better than the pile it was drawn from.

This lives on the outcome side of the wall on purpose. It reads both the predictions
and what happened, which is exactly the pairing the scorer is not allowed to see.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..store import db
from ..util import dates
from ..util import logging as log

LOG = log.get("outcomes.calibration")


@dataclass
class Calibration:
    run_date: str
    bucket: str
    k: int
    hits: int
    labelled: int
    base_rate: float
    precision: float
    lift: float


def composite(row: dict | sqlite3.Row, config: dict) -> tuple[float, dict]:
    """Normalise each outcome signal, then take the weighted sum."""
    cfg = config.get("outcomes", {}).get("composite", {})
    weights = cfg.get("weights", {})
    ceilings = {
        "hf_upvotes": float(cfg.get("hf_upvotes_ceiling", 60.0)),
        "github_stars": float(cfg.get("github_stars_ceiling", 500.0)),
        "citations": float(cfg.get("citations_ceiling", 10.0)),
        "hn_points": float(cfg.get("hn_points_ceiling", 100.0)),
    }
    data = dict(row)
    citations = max(int(data.get("citations_openalex") or 0), int(data.get("citations_s2") or 0))
    raw = {
        "hf_upvotes": float(data.get("hf_upvotes") or 0),
        "github_stars": float(data.get("github_stars") or 0),
        "citations": float(citations),
        "hn_points": float(data.get("hn_points") or 0),
    }
    parts = {key: min(1.0, value / ceilings[key]) if ceilings[key] else 0.0 for key, value in raw.items()}
    total = sum(parts[key] * float(weights.get(key, 0.0)) for key in parts)
    weight_sum = sum(float(w) for w in weights.values()) or 1.0
    return round(total / weight_sum, 4), {k: round(v, 4) for k, v in parts.items()}


def compute_labels(conn: sqlite3.Connection, config: dict) -> int:
    """Derive the composite label for every paper with an observation at the label horizon."""
    cfg = config.get("outcomes", {}).get("composite", {})
    horizon = int(cfg.get("horizon_days", 30))
    threshold = float(cfg.get("hit_threshold", 0.30))
    rows = conn.execute(
        "SELECT * FROM paper_outcomes WHERE horizon_days = ?", (horizon,)
    ).fetchall()
    now = dates.utcnow().isoformat(timespec="seconds")
    for row in rows:
        value, parts = composite(row, config)
        db.upsert(
            conn,
            "outcome_labels",
            {
                "arxiv_id": row["arxiv_id"],
                "horizon_days": horizon,
                "composite": value,
                "hit": 1 if value >= threshold else 0,
                "parts": db.dumps(parts),
                "computed_at": now,
            },
            keys=["arxiv_id"],
        )
    conn.commit()
    LOG.info("labelled %d papers at t+%d", len(rows), horizon)
    return len(rows)


def precision_at_k(
    conn: sqlite3.Connection,
    *,
    k: int = 10,
    bucket: str | None = None,
    scorer_version: str | None = None,
) -> list[Calibration]:
    """Per run: how many of the top k turned into hits, against that day's base rate."""
    params: list[object] = []
    clause = ""
    if scorer_version:
        clause += " AND s.scorer_version = ?"
        params.append(scorer_version)

    run_dates = [
        r[0] for r in conn.execute(
            f"SELECT DISTINCT s.run_date FROM paper_scores s WHERE 1=1{clause} ORDER BY s.run_date",
            params,
        )
    ]
    results: list[Calibration] = []
    for run_date in run_dates:
        rows = conn.execute(
            "SELECT s.arxiv_id, s.score, s.claim_score, s.bucket, l.hit"
            " FROM paper_scores s LEFT JOIN outcome_labels l USING (arxiv_id)"
            " WHERE s.run_date = ?",
            (run_date,),
        ).fetchall()
        labelled = [r for r in rows if r["hit"] is not None]
        if not labelled:
            continue
        base_rate = sum(r["hit"] for r in labelled) / len(labelled)

        if bucket:
            pool = [r for r in labelled if r["bucket"] == bucket]
            key = "claim_score" if bucket == "high_variance" else "score"
        else:
            pool = labelled
            key = "score"
        pool = sorted(pool, key=lambda r: r[key] or 0.0, reverse=True)[:k]
        if not pool:
            continue
        hits = sum(r["hit"] for r in pool)
        precision = hits / len(pool)
        results.append(
            Calibration(
                run_date=run_date,
                bucket=bucket or "all",
                k=len(pool),
                hits=hits,
                labelled=len(labelled),
                base_rate=round(base_rate, 4),
                precision=round(precision, 4),
                lift=round(precision / base_rate, 3) if base_rate else 0.0,
            )
        )
    return results


def summarise(conn: sqlite3.Connection, *, k: int = 10) -> dict:
    """Headline numbers for the digest and, later, the public prediction log."""
    out: dict = {"k": k, "buckets": {}}
    for bucket in (None, "confidence", "high_variance"):
        rows = precision_at_k(conn, k=k, bucket=bucket)
        name = bucket or "all"
        if not rows:
            out["buckets"][name] = {"runs": 0}
            continue
        out["buckets"][name] = {
            "runs": len(rows),
            "precision": round(sum(r.precision for r in rows) / len(rows), 4),
            "base_rate": round(sum(r.base_rate for r in rows) / len(rows), 4),
            "lift": round(sum(r.lift for r in rows) / len(rows), 3),
            "papers_labelled": sum(r.labelled for r in rows),
        }
    row = conn.execute("SELECT count(*) FROM outcome_labels").fetchone()
    out["labelled_total"] = row[0] if row else 0
    return out


def hits_and_misses(conn: sqlite3.Connection, limit: int = 5) -> dict[str, list[sqlite3.Row]]:
    """Shortlisted papers that landed, and high scorers that went nowhere."""
    hits = conn.execute(
        "SELECT s.arxiv_id, p.title, s.score, s.bucket, l.composite FROM paper_scores s"
        " JOIN outcome_labels l USING (arxiv_id) JOIN papers p USING (arxiv_id)"
        " WHERE s.shortlisted = 1 AND l.hit = 1 ORDER BY l.composite DESC LIMIT ?",
        (limit,),
    ).fetchall()
    misses = conn.execute(
        "SELECT s.arxiv_id, p.title, s.score, s.bucket, l.composite FROM paper_scores s"
        " JOIN outcome_labels l USING (arxiv_id) JOIN papers p USING (arxiv_id)"
        " WHERE s.shortlisted = 1 AND l.hit = 0 ORDER BY s.score DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return {"hits": list(hits), "misses": list(misses)}


def missed_winners(conn: sqlite3.Connection, limit: int = 5) -> list[sqlite3.Row]:
    """Papers that did well and never made a shortlist. The expensive errors."""
    return list(
        conn.execute(
            "SELECT s.arxiv_id, p.title, s.score, l.composite FROM paper_scores s"
            " JOIN outcome_labels l USING (arxiv_id) JOIN papers p USING (arxiv_id)"
            " WHERE s.shortlisted = 0 AND l.hit = 1 ORDER BY l.composite DESC LIMIT ?",
            (limit,),
        )
    )
