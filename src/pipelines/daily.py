"""The daily run: ingest, extract, score, log the prediction, render the digest.

Idempotent by construction. Re-running the same day re-uses stored features and
overwrites the digest. A missed day is recovered by the widened lookback window rather
than by anyone noticing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..features.extract import extract_day
from ..ingest import lab_blogs, rss
from ..ingest.arxiv import ArxivClient
from ..ingest.models import Paper
from ..ingest.store import save_papers
from ..llm.client import Client
from ..outcomes import calibration
from ..render import digest as render_digest
from ..render.summaries import summarise
from ..score import rank
from ..store import db
from ..util import dates
from ..util import logging as log
from ..util.config import db_path
from ..util.config import load as load_config
from ..util.http import Fetcher

LOG = log.get("pipeline.daily")


@dataclass
class DailyResult:
    run_id: int
    run_date: str
    ingested: dict
    extracted: dict
    scored: int
    shortlisted: int
    paths: dict
    spend: dict


def run(
    *,
    config: dict | None = None,
    run_date: str | None = None,
    lookback_hours: float | None = None,
    limit: int | None = None,
    offline: bool = False,
    skip_ingest: bool = False,
) -> DailyResult:
    config = config or load_config()
    ingest_cfg = config.get("ingest", {})
    run_date = run_date or dates.today_str()
    path = db_path(config)

    conn = db.connect(path)
    previous = db.last_successful_run(conn, "daily")
    start, end = dates.lookback_window(
        previous["started_at"] if previous else None,
        default_hours=float(lookback_hours or ingest_cfg.get("lookback_hours", 30)),
        max_hours=float(ingest_cfg.get("max_lookback_hours", 168)),
    )
    run_id = db.start_run(conn, "daily", run_date, window_from=start.isoformat(), window_to=end.isoformat())
    LOG.info("daily run %d covering %s to %s", run_id, start.isoformat(), end.isoformat())

    retry = ingest_cfg.get("retry", {})
    fetcher = Fetcher(
        attempts=int(retry.get("attempts", 5)),
        backoff_seconds=float(retry.get("backoff_seconds", 4.0)),
        backoff_multiplier=float(retry.get("backoff_multiplier", 2.0)),
        offline=offline,
    )
    client = Client(config=config)
    status = "ok"

    try:
        papers: list[Paper] = []
        ingest_errors: list[str] = []
        if not skip_ingest:
            papers, ingest_errors = _ingest(fetcher, config, start, end)
            if limit:
                papers = papers[:limit]
            ingested = save_papers(conn, papers)
            posts = lab_blogs.fetch(fetcher, ingest_cfg.get("lab_blogs", []))
            lab_blogs.save(conn, posts)
        else:
            ingested = {"seen": 0, "new": 0}

        window_ids = _window_paper_ids(conn, papers, start, end)
        rows = _papers_by_id(conn, window_ids)

        # Day-zero work runs on a connection that cannot reach the outcome tables.
        scoped = db.open_feature_scoped(path)
        try:
            extracted = extract_day(scoped, rows, client=client, fetcher=fetcher, config=config)
            buckets = rank.rank_ids(scoped, window_ids, run_date, config)
        finally:
            db.close(scoped)

        shortlist = buckets["confidence"] + buckets["high_variance"]
        summaries = summarise(client, [p.row for p in shortlist])
        for paper in shortlist:
            rank.save_summary(conn, paper.arxiv_id, run_date, summaries.get(paper.arxiv_id, ""), config)

        context = render_digest.build_context(
            date=run_date,
            buckets=buckets,
            summaries=summaries,
            scored_count=len(window_ids),
            config=config,
            calibration=calibration.summarise(conn),
            lab_posts=render_digest.lab_posts_for(conn, run_date),
            spend=client.spend_summary(),
            run_id=run_id,
        )
        paths = render_digest.write(context, config)

        if ingest_errors or not window_ids:
            status = "partial"
        db.record_llm_calls(conn, run_id, client.call_rows())
        spend = client.spend_summary()
        db.finish_run(
            conn, run_id, status,
            {
                "ingested": ingested, "extracted": extracted, "scored": len(window_ids),
                "shortlisted": len(shortlist), "spend": spend,
                "ingest_errors": ingest_errors,
            },
            notes="; ".join(ingest_errors)[:500] or None,
        )
        return DailyResult(
            run_id=run_id, run_date=run_date, ingested=ingested, extracted=extracted,
            scored=len(window_ids), shortlisted=len(shortlist), paths=paths, spend=spend,
        )
    except Exception as exc:  # noqa: BLE001 - record the failure before re-raising
        db.finish_run(conn, run_id, "failed", {}, notes=str(exc)[:500])
        raise
    finally:
        fetcher.close()
        db.close(conn)


def _ingest(fetcher: Fetcher, config: dict, start, end) -> tuple[list[Paper], list[str]]:
    """arXiv API first, category RSS if it returns nothing.

    Returns the papers and any window the API could not read in full, which the caller
    turns into a partial run status.
    """
    client = ArxivClient(fetcher, config)
    papers = client.fetch_window(start, end)
    if papers:
        return papers, client.errors
    LOG.warning("arXiv API returned nothing, falling back to category RSS")
    fallback = rss.fetch_categories(fetcher, config.get("ingest", {}).get("categories", []))
    return fallback, client.errors + ([] if fallback else ["arXiv and RSS both empty"])


def _papers_by_id(conn, arxiv_ids: Sequence[str]) -> list:
    if not arxiv_ids:
        return []
    placeholders = ", ".join("?" for _ in arxiv_ids)
    return conn.execute(
        f"SELECT * FROM papers WHERE arxiv_id IN ({placeholders})", list(arxiv_ids)
    ).fetchall()


def _window_paper_ids(conn, papers: Sequence[Paper], start, end) -> list[str]:
    """Papers to score this run: everything ingested now, plus anything from the window
    that never got a feature row because an earlier run died halfway."""
    ids = {paper.arxiv_id for paper in papers}
    rows = conn.execute(
        "SELECT p.arxiv_id FROM papers p LEFT JOIN paper_features f USING (arxiv_id)"
        " WHERE f.arxiv_id IS NULL AND p.announced_date >= ?",
        (dates.date_str(start),),
    ).fetchall()
    ids.update(row[0] for row in rows)
    return sorted(ids)
