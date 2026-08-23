"""arXiv ingest.

Polls the new-submission listing directly rather than any curated feed. That is the
whole latency advantage: curation lands a day later, and a day is the competitive
position.

One request per three seconds on a single connection, exponential backoff on 429, day
sized windows because pagination through `start` breaks on large result sets.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Sequence

import feedparser

from ..util import dates
from ..util import logging as log
from ..util.http import Fetcher, SourceError
from .models import Paper, split_arxiv_id

LOG = log.get("ingest.arxiv")

API_URL = "https://export.arxiv.org/api/query"


def build_query(categories: Sequence[str], start: datetime, end: datetime) -> str:
    cats = " OR ".join(f"cat:{c}" for c in categories)
    window = f"submittedDate:[{dates.arxiv_stamp(start)} TO {dates.arxiv_stamp(end)}]"
    return f"({cats}) AND {window}"


def parse_atom(body: str, *, source: str = "arxiv_api") -> list[Paper]:
    """Turn an Atom response into paper records. Pure, so it is testable on fixtures."""
    feed = feedparser.parse(body)
    papers: list[Paper] = []
    for entry in feed.entries:
        arxiv_id, version = split_arxiv_id(entry.get("id", ""))
        if not arxiv_id:
            continue
        submitted = entry.get("published") or entry.get("updated")
        submitted_dt = dates.parse_iso(submitted)
        links = []
        pdf_url = None
        abs_url = entry.get("link")
        for link in entry.get("links", []):
            href = link.get("href")
            if not href:
                continue
            if link.get("type") == "application/pdf" or link.get("title") == "pdf":
                pdf_url = href
            elif link.get("rel") == "related":
                links.append(href)
        categories = [tag.get("term") for tag in entry.get("tags", []) if tag.get("term")]
        primary = entry.get("arxiv_primary_category", {}).get("term") if entry.get(
            "arxiv_primary_category"
        ) else (categories[0] if categories else None)
        authors = [a.get("name", "").strip() for a in entry.get("authors", []) if a.get("name")]
        affiliations = [
            a.get("arxiv_affiliation", "").strip()
            for a in entry.get("authors", [])
            if a.get("arxiv_affiliation")
        ]
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                version=version,
                title=" ".join((entry.get("title") or "").split()),
                abstract=" ".join((entry.get("summary") or "").split()),
                comments=entry.get("arxiv_comment"),
                journal_ref=entry.get("arxiv_journal_ref"),
                doi=entry.get("arxiv_doi"),
                primary_category=primary,
                categories=categories,
                authors=authors,
                affiliations=affiliations,
                submitted_at=submitted_dt.isoformat() if submitted_dt else None,
                updated_at=(dates.parse_iso(entry.get("updated")).isoformat()
                            if dates.parse_iso(entry.get("updated")) else None),
                announced_date=dates.date_str(submitted_dt) if submitted_dt else dates.today_str(),
                abs_url=abs_url,
                pdf_url=pdf_url,
                links=links,
                source=source,
            )
        )
    return papers


class ArxivClient:
    def __init__(self, fetcher: Fetcher, config: dict) -> None:
        self.fetcher = fetcher
        ingest = config.get("ingest", {})
        self.categories: list[str] = list(ingest.get("categories", []))
        self.page_size: int = int(ingest.get("max_results_per_page", 200))
        self.max_pages: int = int(ingest.get("max_pages_per_category", 8))

    def fetch_window(self, start: datetime, end: datetime) -> list[Paper]:
        """All papers in the five categories submitted between two moments.

        Windows longer than a day are split, because deep pagination is where the API
        starts truncating.
        """
        collected: dict[str, Paper] = {}
        for chunk_start, chunk_end in _split_days(start, end):
            for paper in self._fetch_chunk(chunk_start, chunk_end):
                collected.setdefault(paper.arxiv_id, paper)
        LOG.info("arXiv returned %d distinct papers", len(collected))
        return list(collected.values())

    def _fetch_chunk(self, start: datetime, end: datetime) -> Iterable[Paper]:
        query = build_query(self.categories, start, end)
        offset = 0
        for page in range(self.max_pages):
            params = {
                "search_query": query,
                "start": offset,
                "max_results": self.page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            try:
                # Cache off: the same window re-queried later should see late arrivals.
                body = self.fetcher.get_text(API_URL, params, use_cache=False)
            except SourceError as exc:
                LOG.error("arXiv page %d failed, keeping what we have: %s", page, exc)
                return
            papers = parse_atom(body)
            if not papers:
                return
            yield from papers
            if len(papers) < self.page_size:
                return
            offset += self.page_size


def _split_days(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Cut a window at midnight boundaries so no single query pages too deep."""
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        midnight = cursor.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        nxt = min(midnight, end)
        if nxt <= cursor:
            nxt = end
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks or [(start, end)]
