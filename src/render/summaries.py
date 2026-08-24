"""Plain-language notes for the shortlist only.

Summarisation is a commodity and is not the product. These exist so the shortlist can
be read quickly, which is why the whole corpus is never summarised.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..llm import prompts as prompt_lib
from ..llm.client import Client
from ..util import logging as log
from ..util.text import sentences, truncate

LOG = log.get("render.summaries")

PURPOSE = "summary"


def _fallback(user_prompt: str) -> str:
    """No model reachable. Quote the opening of the abstract and say so.

    arXiv metadata is CC0, so a short extract is fine to display. It is marked as an
    extract because a machine-written summary and the authors' own words are not the
    same thing.
    """
    _, _, abstract = user_prompt.partition("Abstract:")
    first = sentences(abstract.strip())
    opening = first[0] if first else abstract.strip()
    return f"From the abstract: {truncate(opening, 240)}"


def register_fallback(client: Client) -> None:
    client.heuristic.register(PURPOSE, _fallback)


def summarise(client: Client, papers: Sequence[dict], *, concurrency: int = 2) -> dict[str, str]:
    if not papers:
        return {}
    register_fallback(client)
    prompt = prompt_lib.load("prompts/summary.md")
    requests = [
        prompt.render(title=paper.get("title", ""), abstract=paper.get("abstract", ""))
        for paper in papers
    ]
    results = client.complete_many(PURPOSE, requests, tier="cheap", concurrency=concurrency)
    out: dict[str, str] = {}
    for paper, result in zip(papers, results, strict=True):
        text = (result.text or "").strip()
        if not result.ok or not text:
            text = _fallback(f"Abstract: {paper.get('abstract', '')}")
        out[paper["arxiv_id"]] = " ".join(text.split())
    return out
