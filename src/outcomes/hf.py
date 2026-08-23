"""Hugging Face Daily Papers, as an outcome signal only.

Whether the curators picked a paper up, and how many upvotes it got, says what the
crowd thought. Useful for grading the day-zero score, useless as an input to it, since
curation lands a day after the arXiv announcement this system reads.
"""

from __future__ import annotations

from ..util import logging as log
from ..util.http import Fetcher, SourceError

LOG = log.get("outcomes.hf")

PAPER_URL = "https://huggingface.co/api/papers/{arxiv_id}"
DAILY_URL = "https://huggingface.co/api/daily_papers"


def paper_reception(fetcher: Fetcher, arxiv_id: str) -> dict:
    """Upvotes and listing status for one paper. A miss means it was never listed."""
    try:
        payload = fetcher.get_json(PAPER_URL.format(arxiv_id=arxiv_id))
    except SourceError as exc:
        if "404" in str(exc):
            return {"hf_listed": 0, "hf_upvotes": 0, "ok": True}
        LOG.warning("HF lookup failed for %s: %s", arxiv_id, exc)
        return {"hf_listed": None, "hf_upvotes": None, "ok": False}
    upvotes = payload.get("upvotes")
    return {
        "hf_listed": 1,
        "hf_upvotes": int(upvotes) if isinstance(upvotes, (int, float)) else 0,
        "ok": True,
    }


def daily_listing(fetcher: Fetcher, date: str) -> list[dict]:
    """The curated list for one day. Kept for backfilling listing status in bulk."""
    try:
        payload = fetcher.get_json(DAILY_URL, {"date": date})
    except SourceError as exc:
        LOG.warning("HF daily listing failed for %s: %s", date, exc)
        return []
    return payload if isinstance(payload, list) else []
