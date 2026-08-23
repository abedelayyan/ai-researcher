"""Hacker News attention through the Algolia API."""

from __future__ import annotations

from ..util import logging as log
from ..util.http import Fetcher, SourceError

LOG = log.get("outcomes.hn")

SEARCH_URL = "https://hn.algolia.com/api/v1/search"


def mentions(fetcher: Fetcher, arxiv_id: str, title: str | None = None) -> dict:
    """Any story pointing at the paper, by ID first and title second."""
    best: dict = {"hn_hits": 0, "hn_points": 0, "hn_comments": 0, "hn_url": None, "ok": True}
    for query in filter(None, [arxiv_id, (title or "").strip()[:80] or None]):
        try:
            payload = fetcher.get_json(SEARCH_URL, {"query": query, "tags": "story", "hitsPerPage": 10})
        except SourceError as exc:
            LOG.info("HN search failed for %r: %s", query, exc)
            best["ok"] = False
            continue
        hits = payload.get("hits") or []
        matching = [
            hit for hit in hits
            if arxiv_id in (hit.get("url") or "")
            or _title_matches(title, hit.get("title"))
        ]
        if not matching:
            continue
        top = max(matching, key=lambda hit: hit.get("points") or 0)
        if (top.get("points") or 0) >= best["hn_points"]:
            best.update(
                {
                    "hn_hits": len(matching),
                    "hn_points": int(top.get("points") or 0),
                    "hn_comments": int(top.get("num_comments") or 0),
                    "hn_url": f"https://news.ycombinator.com/item?id={top.get('objectID')}",
                }
            )
    return best


def _title_matches(paper_title: str | None, hit_title: str | None) -> bool:
    if not paper_title or not hit_title:
        return False
    a = "".join(ch for ch in paper_title.lower() if ch.isalnum() or ch == " ").strip()
    b = "".join(ch for ch in hit_title.lower() if ch.isalnum() or ch == " ").strip()
    return bool(a) and (a in b or b in a)
