"""Walk back over scored papers and record what happened.

Runs on its own schedule. Each paper is revisited at t+7, t+30 and t+90. A missed run
does not lose an observation: anything overdue is still collected on the next run and
the true age is stored alongside the intended horizon.

Every source is third party and will break at some point. A dead source degrades the
observation rather than killing the run, and `sources_ok` records which ones answered.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from ..store import db
from ..util import dates
from ..util import logging as log
from ..util.http import Fetcher
from . import citations, github, hf, hn

LOG = log.get("outcomes.collect")


def due_observations(
    conn: sqlite3.Connection,
    today: str,
    horizons: Sequence[int],
    *,
    limit: int | None = None,
) -> list[tuple[sqlite3.Row, int]]:
    """Papers whose horizon has arrived and which have no observation for it yet."""
    due: list[tuple[sqlite3.Row, int]] = []
    for horizon in sorted(horizons):
        rows = conn.execute(
            "SELECT p.arxiv_id, p.title, p.announced_date, f.code_url"
            " FROM papers p"
            " LEFT JOIN paper_features f USING (arxiv_id)"
            " LEFT JOIN paper_outcomes o"
            "   ON o.arxiv_id = p.arxiv_id AND o.horizon_days = ?"
            " WHERE o.arxiv_id IS NULL"
            "   AND date(p.announced_date, ?) <= date(?)"
            " ORDER BY p.announced_date DESC",
            (horizon, f"+{horizon} days", today),
        ).fetchall()
        due.extend((row, horizon) for row in rows)
    if limit is not None:
        due = due[:limit]
    return due


def observe(fetcher: Fetcher, arxiv_id: str, title: str, code_url: str | None) -> dict:
    """One observation across every source, with per-source failure recorded."""
    record: dict = {}
    ok: dict[str, bool] = {}

    reception = hf.paper_reception(fetcher, arxiv_id)
    ok["hf"] = bool(reception.pop("ok", False))
    record.update(reception)

    repo = github.collect(fetcher, arxiv_id, code_url)
    ok["github"] = bool(repo.pop("ok", False))
    record.update(repo)

    openalex = citations.openalex_citations(fetcher, arxiv_id)
    ok["openalex"] = bool(openalex.pop("ok", False))
    record.update(openalex)

    s2 = citations.semantic_scholar_citations(fetcher, arxiv_id)
    ok["semantic_scholar"] = bool(s2.pop("ok", False))
    record.update(s2)

    forum = hn.mentions(fetcher, arxiv_id, title)
    ok["hn"] = bool(forum.pop("ok", False))
    record.update(forum)

    record["sources_ok"] = db.dumps(ok)
    return record


def collect_due(
    conn: sqlite3.Connection,
    fetcher: Fetcher,
    config: dict,
    *,
    today: str | None = None,
    limit: int | None = None,
) -> dict:
    cfg = config.get("outcomes", {})
    today = today or dates.today_str()
    horizons = cfg.get("horizons_days", [7, 30, 90])
    due = due_observations(conn, today, horizons, limit=limit)
    if not due:
        LOG.info("no observations due")
        return {"observed": 0, "due": 0}

    LOG.info("%d observations due", len(due))
    written = 0
    for row, horizon in due:
        record = observe(fetcher, row["arxiv_id"], row["title"], row["code_url"])
        record.update(
            {
                "arxiv_id": row["arxiv_id"],
                "horizon_days": horizon,
                "observed_at": today,
                "raw": db.dumps({"age_days": dates.days_between(row["announced_date"], today)}),
            }
        )
        db.upsert(conn, "paper_outcomes", record, keys=["arxiv_id", "horizon_days"])
        written += 1
        if written % 25 == 0:
            conn.commit()
    conn.commit()
    return {"observed": written, "due": len(due)}
