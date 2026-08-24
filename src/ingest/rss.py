"""arXiv category RSS, used when the Atom API is refusing requests.

The RSS feed carries the current day's announcements only and gives less metadata than
the API, so it is a fallback rather than the primary path.
"""

from __future__ import annotations

from collections.abc import Sequence

import feedparser

from ..util import dates
from ..util import logging as log
from ..util.http import Fetcher, SourceError
from .models import Paper, split_arxiv_id

LOG = log.get("ingest.rss")

FEED_URL = "https://rss.arxiv.org/rss/{category}"


def parse_feed(body: str, category: str) -> list[Paper]:
    feed = feedparser.parse(body)
    papers: list[Paper] = []
    for entry in feed.entries:
        arxiv_id, version = split_arxiv_id(entry.get("id") or entry.get("link", ""))
        if not arxiv_id:
            continue
        description = entry.get("summary", "")
        # The feed prefixes the abstract with "arXiv:ID Announce Type: new Abstract: ".
        abstract = description.split("Abstract:", 1)[-1].strip()
        published = dates.parse_iso(entry.get("published"))
        authors = [a.strip() for a in (entry.get("author") or "").split(",") if a.strip()]
        terms = [tag.get("term") for tag in entry.get("tags", []) if tag.get("term")]
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                version=version,
                title=" ".join((entry.get("title") or "").split()),
                abstract=" ".join(abstract.split()),
                primary_category=terms[0] if terms else category,
                categories=terms or [category],
                authors=authors,
                submitted_at=published.isoformat() if published else None,
                announced_date=dates.date_str(published) if published else dates.today_str(),
                abs_url=entry.get("link"),
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                source="arxiv_rss",
            )
        )
    return papers


def fetch_categories(fetcher: Fetcher, categories: Sequence[str]) -> list[Paper]:
    collected: dict[str, Paper] = {}
    for category in categories:
        try:
            body = fetcher.get_text(FEED_URL.format(category=category), use_cache=False)
        except SourceError as exc:
            LOG.error("RSS for %s failed: %s", category, exc)
            continue
        for paper in parse_feed(body, category):
            collected.setdefault(paper.arxiv_id, paper)
    return list(collected.values())
