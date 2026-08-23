"""Lab blogs over RSS. Low volume, sometimes ahead of arXiv.

Stored for context in the digest. Blog posts are not scored, because the day-zero
rubric is built around paper claims.
"""

from __future__ import annotations

import sqlite3
from typing import Sequence

import feedparser

from ..store import db
from ..util import dates
from ..util import logging as log
from ..util.http import Fetcher, SourceError

LOG = log.get("ingest.labs")


def fetch(fetcher: Fetcher, feeds: Sequence[dict]) -> list[dict]:
    posts: list[dict] = []
    for feed_cfg in feeds:
        name, url = feed_cfg.get("name", "?"), feed_cfg.get("url")
        if not url:
            continue
        try:
            body = fetcher.get_text(url, use_cache=False)
        except SourceError as exc:
            LOG.warning("%s blog feed unavailable: %s", name, exc)
            continue
        parsed = feedparser.parse(body)
        for entry in parsed.entries[:20]:
            link = entry.get("link")
            if not link:
                continue
            published = dates.parse_iso(entry.get("published") or entry.get("updated"))
            posts.append(
                {
                    "url": link,
                    "lab": name,
                    "title": " ".join((entry.get("title") or "").split()),
                    "summary": " ".join((entry.get("summary") or "").split())[:600],
                    "published": published.isoformat() if published else None,
                    "ingested_at": dates.utcnow().isoformat(timespec="seconds"),
                }
            )
    return posts


def save(conn: sqlite3.Connection, posts: Sequence[dict]) -> int:
    for post in posts:
        db.upsert(conn, "lab_posts", dict(post), keys=["url"])
    conn.commit()
    return len(posts)
