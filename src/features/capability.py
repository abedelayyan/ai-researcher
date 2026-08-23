"""Capability delta: the four-axis rubric at the centre of the day-zero score.

Scored by the cheap model against prompts/capability_delta.md. The rubric asks what the
work makes possible, not whether the method is clever. A paper that moves none of the
four axes is almost never a business seed however elegant it is.

When no model is reachable the rules in :func:`heuristic_scores` stand in. They are
weaker and deliberately capped below the top of each axis, and every row records which
source produced it so a run scored by rules can be told apart later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..llm import prompts as prompt_lib
from ..llm.client import Client, LLMResult, extract_json
from ..util import logging as log
from .structural import estimate_compute_band

LOG = log.get("features.capability")

AXES = ("cost_curve", "constraint_removal", "usability_threshold", "modality_opening")
BANDS = ("consumer", "single_node", "small_cluster", "frontier", "unknown")
PURPOSE = "capability_delta"


@dataclass
class CapabilityScore:
    scores: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    compute_band: str = "unknown"
    compute_band_why: str = ""
    source: str = "heuristic:rules"
    prompt_version: str = "v1"

    @property
    def total(self) -> int:
        return sum(self.scores.get(axis, 0) for axis in AXES)


def _clamp(value: object, ceiling: int = 3) -> int:
    try:
        number = int(float(value))  # models return "2" and 2.0 alike
    except (TypeError, ValueError):
        return 0
    return max(0, min(ceiling, number))


def parse_response(text: str, *, source: str, prompt_version: str) -> CapabilityScore:
    data = extract_json(text)
    scores = {axis: _clamp(data.get(axis)) for axis in AXES}
    reasons = {axis: str(data.get(f"{axis}_why", "") or "").strip()[:200] for axis in AXES}
    band = str(data.get("compute_band", "unknown") or "unknown").strip().lower()
    if band not in BANDS:
        band = "unknown"
    return CapabilityScore(
        scores=scores,
        reasons=reasons,
        compute_band=band,
        compute_band_why=str(data.get("compute_band_why", "") or "").strip()[:200],
        source=source,
        prompt_version=prompt_version,
    )


# --- rules fallback ---------------------------------------------------------------

_COST_RE = re.compile(
    r"\b(cheaper|cost|compute[- ]efficient|fewer (gpus|parameters|labels|annotations)|"
    r"less (data|compute|memory|supervision)|faster (training|inference)|speed[- ]?up|"
    r"quantis|quantiz|distill|prun|sparse|cache|throughput|\d+\s*[x×]\s*(faster|cheaper))\b",
    re.IGNORECASE)
_CONSTRAINT_RE = re.compile(
    r"\b(without (requiring|any|the need|labelled|labeled|retraining|access)|"
    r"no longer requires|removes the need|training[- ]free|annotation[- ]free|"
    r"privacy[- ]preserving|on[- ]device|offline|unbounded|arbitrary length|"
    r"first (method|system|approach) to)\b", re.IGNORECASE)
_USABILITY_RE = re.compile(
    r"\b(real[- ]time|interactive latency|production|deployable|reliab|"
    r"reduces hallucinat|success rate of \d|passes? \d+% of|human[- ]level|"
    r"outperforms (gpt-4|human|expert))\b", re.IGNORECASE)
_MODALITY_RE = re.compile(
    r"\b(multimodal|multi[- ]modal|video|audio|speech|3d|point cloud|tactile|"
    r"low[- ]resource languages?|new benchmark for|cross[- ]lingual|protein|molecul|"
    r"embodied|edge deployment)\b", re.IGNORECASE)


def heuristic_scores(text: str) -> CapabilityScore:
    """Rules used only when no model is reachable. Capped at 2 on every axis."""
    def graded(pattern: re.Pattern[str]) -> int:
        hits = len(pattern.findall(text or ""))
        return 0 if hits == 0 else (1 if hits < 3 else 2)

    scores = {
        "cost_curve": graded(_COST_RE),
        "constraint_removal": graded(_CONSTRAINT_RE),
        "usability_threshold": graded(_USABILITY_RE),
        "modality_opening": graded(_MODALITY_RE),
    }
    band, why = estimate_compute_band(text)
    return CapabilityScore(
        scores=scores,
        reasons={axis: "keyword rules, no model available" for axis in AXES},
        compute_band=band,
        compute_band_why=why,
        source="heuristic:rules",
        prompt_version="rules",
    )


def _heuristic_handler(user_prompt: str) -> str:
    """Registered on the client so the heuristic provider can answer this purpose."""
    score = heuristic_scores(user_prompt)
    payload = {axis: score.scores[axis] for axis in AXES}
    payload.update({f"{axis}_why": score.reasons[axis] for axis in AXES})
    payload["compute_band"] = score.compute_band
    payload["compute_band_why"] = score.compute_band_why
    import json

    return json.dumps(payload)


def register_fallback(client: Client) -> None:
    client.heuristic.register(PURPOSE, _heuristic_handler)


# --- scoring ----------------------------------------------------------------------


def build_request(paper: dict, prompt: prompt_lib.Prompt) -> tuple[str, str]:
    return prompt.render(
        title=paper.get("title", ""),
        abstract=paper.get("abstract", ""),
        categories=", ".join(paper.get("categories", []) or []),
        comments=paper.get("comments") or "none",
    )


def score_papers(
    client: Client,
    papers: Sequence[dict],
    config: dict,
    *,
    concurrency: int = 2,
) -> dict[str, CapabilityScore]:
    """Score a day's papers in one batch. Keyed by arXiv ID."""
    if not papers:
        return {}
    register_fallback(client)
    cfg = config.get("capability_delta", {})
    prompt = prompt_lib.load(cfg.get("prompt", "prompts/capability_delta.md"))
    version = cfg.get("prompt_version", prompt.version)

    requests = [build_request(paper, prompt) for paper in papers]
    results: list[LLMResult] = client.complete_many(
        PURPOSE, requests, tier="cheap", concurrency=concurrency
    )

    out: dict[str, CapabilityScore] = {}
    for paper, result in zip(papers, results):
        arxiv_id = paper["arxiv_id"]
        if not result.ok or not result.text:
            LOG.warning("no capability score for %s (%s), falling back to rules", arxiv_id, result.error)
            out[arxiv_id] = heuristic_scores(f"{paper.get('title','')} {paper.get('abstract','')}")
            continue
        try:
            out[arxiv_id] = parse_response(result.text, source=result.source, prompt_version=version)
        except (ValueError, KeyError) as exc:
            LOG.warning("unparseable capability response for %s: %s", arxiv_id, exc)
            out[arxiv_id] = heuristic_scores(f"{paper.get('title','')} {paper.get('abstract','')}")
    return out
