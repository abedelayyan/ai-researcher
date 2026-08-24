"""Persist ingested papers, idempotently."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from ..store import db
from ..util import logging as log
from ..util.text import candidate_terms
from .models import Paper

LOG = log.get("ingest.store")


def save_papers(conn: sqlite3.Connection, papers: Sequence[Paper]) -> dict[str, int]:
    """Insert new papers, refresh metadata on ones already seen.

    Returns counts. Re-running the same window is a no-op beyond metadata refresh,
    which is what lets a missed run be recovered by simply widening the window.
    """
    known = {row[0] for row in conn.execute("SELECT arxiv_id FROM papers")}
    new_papers = [p for p in papers if p.arxiv_id not in known]
    for paper in papers:
        db.upsert(conn, "papers", paper.to_row(), keys=["arxiv_id"])
    _record_terms(conn, new_papers)
    conn.commit()
    LOG.info("saved %d papers (%d new)", len(papers), len(new_papers))
    return {"seen": len(papers), "new": len(new_papers)}


def _record_terms(conn: sqlite3.Connection, papers: Sequence[Paper]) -> None:
    """Grow the corpus vocabulary. Only new papers count, so counts stay honest."""
    for paper in papers:
        for term in candidate_terms(f"{paper.title}. {paper.abstract}"):
            conn.execute(
                "INSERT INTO corpus_terms (term, first_seen_date, paper_count)"
                " VALUES (?, ?, 1) ON CONFLICT(term) DO UPDATE SET"
                " paper_count = paper_count + 1",
                (term, paper.announced_date),
            )


def papers_for_date(conn: sqlite3.Connection, announced_date: str) -> list[sqlite3.Row]:
    return list(
        conn.execute("SELECT * FROM papers WHERE announced_date = ? ORDER BY arxiv_id", (announced_date,))
    )


def papers_between(conn: sqlite3.Connection, start_date: str, end_date: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM papers WHERE announced_date >= ? AND announced_date <= ?"
            " ORDER BY announced_date, arxiv_id",
            (start_date, end_date),
        )
    )
