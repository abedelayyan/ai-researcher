"""Citation counts from OpenAlex and Semantic Scholar."""

from __future__ import annotations

import os

from ..util import logging as log
from ..util.config import contact_email
from ..util.config import load as load_config
from ..util.http import Fetcher, SourceError

LOG = log.get("outcomes.citations")

OPENALEX_WORK = "https://api.openalex.org/works/doi:10.48550/arXiv.{arxiv_id}"
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"


def openalex_citations(fetcher: Fetcher, arxiv_id: str) -> dict:
    params = {}
    email = contact_email(load_config())
    if email:
        params["mailto"] = email
    try:
        payload = fetcher.get_json(OPENALEX_WORK.format(arxiv_id=arxiv_id), params)
    except SourceError as exc:
        if "404" in str(exc):
            return {"citations_openalex": 0, "ok": True}
        LOG.info("OpenAlex citation lookup failed for %s: %s", arxiv_id, exc)
        return {"ok": False}
    return {"citations_openalex": int(payload.get("cited_by_count", 0) or 0), "ok": True}


def semantic_scholar_citations(fetcher: Fetcher, arxiv_id: str) -> dict:
    headers = {}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        headers["x-api-key"] = key
    try:
        payload = fetcher.get_json(
            S2_PAPER.format(arxiv_id=arxiv_id),
            {"fields": "citationCount,influentialCitationCount"},
            headers=headers,
        )
    except SourceError as exc:
        if "404" in str(exc):
            return {"citations_s2": 0, "ok": True}
        LOG.info("Semantic Scholar lookup failed for %s: %s", arxiv_id, exc)
        return {"ok": False}
    return {"citations_s2": int(payload.get("citationCount", 0) or 0), "ok": True}
