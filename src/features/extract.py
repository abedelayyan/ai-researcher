"""Assemble the day-zero feature row for each paper.

Runs on a connection that cannot see outcome tables, so a mistake here fails loudly
instead of quietly poisoning the prediction log.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

from ..llm.client import Client
from ..store import db
from ..util import dates
from ..util import logging as log
from ..util.http import Fetcher
from . import structural
from .author_prior import AuthorPriorService
from .capability import CapabilityScore, heuristic_scores, score_papers, triage

LOG = log.get("features.extract")

FEATURES_VERSION = "v1"


def paper_to_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    for key in ("categories", "authors", "links", "affiliations"):
        data[key] = db.loads(data.get(key), [])
    return data


def pending(conn: sqlite3.Connection, papers: Sequence[sqlite3.Row], *, force: bool = False) -> list[dict]:
    """Papers with no feature row yet. Re-running a day costs nothing."""
    if force:
        return [paper_to_dict(row) for row in papers]
    done = {r[0] for r in conn.execute(
        "SELECT arxiv_id FROM paper_features WHERE features_version = ?", (FEATURES_VERSION,)
    )}
    return [paper_to_dict(row) for row in papers if row["arxiv_id"] not in done]


def structural_features(conn: sqlite3.Connection, paper: dict) -> dict:
    abstract = paper.get("abstract", "")
    comments = paper.get("comments")
    code = structural.find_code_link(abstract, comments, paper.get("links", []))
    claims = structural.parse_benchmark_claims(abstract)
    categories = paper.get("categories", []) or []
    return {
        "has_code_link": 1 if code.present else 0,
        "code_url": code.url,
        "code_link_source": code.source,
        "claims_numbers": 1 if claims.has_numbers else 0,
        "benchmark_claims": db.dumps(claims.claims),
        "max_point_gain": claims.max_point_gain,
        "max_relative_gain": claims.max_relative_gain,
        "benchmark_note": claims.note,
        "cross_listed": 1 if len(categories) > 1 else 0,
        "category_count": max(1, len(categories)),
        "claims_system": 1 if structural.claims_system(abstract) else 0,
        "application_domain": structural.application_domain(paper.get("title", ""), abstract),
        "novel_terms": db.dumps(
            structural.novel_terms(conn, paper.get("title", ""), abstract, paper["announced_date"])
        ),
        "abstract_length": len(abstract or ""),
    }


def capability_columns(score: CapabilityScore) -> dict:
    columns: dict[str, object] = {}
    for axis, value in score.scores.items():
        columns[axis] = value
        columns[f"{axis}_why"] = score.reasons.get(axis, "")
    columns["capability_total"] = score.total
    columns["compute_band"] = score.compute_band
    columns["compute_band_why"] = score.compute_band_why
    columns["capability_source"] = score.source
    columns["prompt_version"] = score.prompt_version
    return columns


def extract_day(
    conn: sqlite3.Connection,
    papers: Sequence[sqlite3.Row],
    *,
    client: Client,
    fetcher: Fetcher,
    config: dict,
    cutoff_date: str = "",
    force: bool = False,
) -> dict:
    """Extract features for a set of papers and write them.

    cutoff_date is empty for a live run. The backtest passes the publication date so
    author priors are computed as they stood then.
    """
    todo = pending(conn, papers, force=force)
    if not todo:
        LOG.info("no papers needing extraction")
        return {"extracted": 0, "skipped": len(papers)}

    LOG.info("extracting features for %d papers", len(todo))
    priors = AuthorPriorService(conn, fetcher, config)

    # Structural features are free, so they come first and decide who gets a model call.
    structural_rows = {paper["arxiv_id"]: structural_features(conn, paper) for paper in todo}
    cap = int(config.get("capability_delta", {}).get("max_model_papers", 0) or 0)
    ordered = triage(todo, structural_rows)
    for_model = ordered[:cap] if cap and len(ordered) > cap else ordered
    if len(for_model) < len(todo):
        LOG.info("model budget covers %d of %d papers, the rest fall back to rules",
                 len(for_model), len(todo))
    capability = score_papers(client, for_model, config, concurrency=2)
    for paper in todo:
        if paper["arxiv_id"] not in capability:
            capability[paper["arxiv_id"]] = heuristic_scores(
                f"{paper.get('title', '')} {paper.get('abstract', '')}"
            )

    written = 0
    for paper in todo:
        row: dict[str, object] = {"arxiv_id": paper["arxiv_id"]}
        row.update(structural_rows[paper["arxiv_id"]])

        prior = priors.for_paper(paper.get("authors", []), cutoff_date)
        row["author_prior"] = prior.prior
        row["author_prior_coverage"] = 1 if prior.coverage else 0
        row["author_prior_detail"] = json.dumps(prior.detail)

        score = capability.get(paper["arxiv_id"])
        if score is not None:
            row.update(capability_columns(score))
            # The abstract sometimes states compute the model missed. Take the larger.
            if score.compute_band == "unknown":
                band, why = structural.estimate_compute_band(
                    paper.get("abstract", ""), paper.get("comments")
                )
                row["compute_band"] = band
                row["compute_band_why"] = why

        row["features_version"] = FEATURES_VERSION
        row["extracted_at"] = dates.utcnow().isoformat(timespec="seconds")
        db.upsert(conn, "paper_features", row, keys=["arxiv_id"])
        written += 1

    conn.commit()
    return {"extracted": written, "skipped": len(papers) - written}


def features_for_date(conn: sqlite3.Connection, announced_date: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT p.*, f.* FROM papers p JOIN paper_features f USING (arxiv_id)"
            " WHERE p.announced_date = ? ORDER BY p.arxiv_id",
            (announced_date,),
        )
    )
