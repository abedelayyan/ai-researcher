"""Deterministic features, all readable straight off the paper on the day it appears.

No model calls here, so this stays free, fast and identical on every replay, which
matters for the backtest.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..util.text import candidate_terms

# --- code release ----------------------------------------------------------------

_CODE_HOSTS = (
    "github.com", "gitlab.com", "bitbucket.org", "codeberg.org",
    "huggingface.co", "github.io", "gitee.com",
)
_URL_RE = re.compile(r"https?://[^\s<>\"\)\],;]+", re.IGNORECASE)
_CODE_PROMISE_RE = re.compile(
    r"(code|models?|weights|dataset|implementation)[^.]{0,60}"
    r"(is|are|will be|has been)?\s*(publicly\s+)?(available|released|open[- ]sourced)",
    re.IGNORECASE,
)


@dataclass
class CodeLink:
    url: str | None = None
    source: str | None = None   # abstract | comments | links | promise

    @property
    def present(self) -> bool:
        return self.url is not None or self.source == "promise"


def find_code_link(abstract: str, comments: str | None, links: Sequence[str]) -> CodeLink:
    """A repository link at publication time signals intent for the work to be used.

    Star counts are a post-publication signal and are never touched here.
    """
    for source, text in (("comments", comments or ""), ("abstract", abstract or "")):
        for url in _URL_RE.findall(text):
            if any(host in url.lower() for host in _CODE_HOSTS):
                return CodeLink(url=url.rstrip(".,);"), source=source)
    for url in links or []:
        if any(host in (url or "").lower() for host in _CODE_HOSTS):
            return CodeLink(url=url, source="links")
    if _CODE_PROMISE_RE.search(abstract or "") or _CODE_PROMISE_RE.search(comments or ""):
        # A promise without a URL. Weaker, and recorded as such.
        return CodeLink(url=None, source="promise")
    return CodeLink()


# --- benchmark claims -------------------------------------------------------------

_NUM = r"(\d+(?:\.\d+)?)"
_FROM_TO_RE = re.compile(
    rf"from\s+{_NUM}\s*%?\s*(?:to|→|->)\s+{_NUM}\s*%?", re.IGNORECASE
)
_VS_PREVIOUS_RE = re.compile(
    rf"{_NUM}\s*%?[^.]{{0,40}}?(?:versus|vs\.?|compared (?:to|with)|against|over)"
    rf"[^.]{{0,40}}?{_NUM}\s*%?",
    re.IGNORECASE,
)
_ABSOLUTE_GAIN_RE = re.compile(
    rf"(?:\+|by|of|an?)\s*{_NUM}\s*(?:%|percentage points?|points?|pp)\s*"
    rf"(?:absolute\s+)?(?:improvement|gain|increase|higher|better|boost)",
    re.IGNORECASE,
)
_GAIN_PREFIX_RE = re.compile(
    rf"(?:improv\w+|outperform\w+|surpass\w+|exceed\w+|boost\w+|gain\w*)[^.]{{0,30}}?"
    rf"(?:by\s+)?{_NUM}\s*(?:%|percentage points?|points?|pp)\b",
    re.IGNORECASE,
)
_MULTIPLE_RE = re.compile(
    rf"{_NUM}\s*(?:x|×|-fold|\s*times)\s+"
    rf"(?:faster|speed[- ]?up|speedup|cheaper|smaller|less|fewer|lower|shorter|"
    rf"reduction|more efficient|throughput|higher)",
    re.IGNORECASE,
)
_REDUCTION_RE = re.compile(
    rf"(?:reduc\w+|cut\w*|shrink\w*|lower\w*|sav\w+|fall\w*|drop\w*)"
    rf"[^.]{{0,40}}?by\s+{_NUM}\s*%", re.IGNORECASE
)


#: Words that mark a number as a metric value rather than a count of something.
_METRIC_WORDS = (
    "accuracy", "score", "f1", "bleu", "rouge", "wer", "cer", "map", "auc", "precision",
    "recall", "exact match", "success rate", "win rate", "pass@", "perplexity",
    "error rate", "iou", "psnr", "ssim", "dice", "mrr", "ndcg", "hit rate", "top-1",
    "top-5", "points", "percent", "detection rate", "solve rate", "coverage",
)

#: Rates, where a jump from 340 to 1180 is a real claim rather than two unrelated counts.
_RATE_WORDS = (
    "per second", "per minute", "per hour", "per day", "throughput", "fps",
    "frames per", "queries per", "tokens per", "samples per", "requests per",
    "images per", "latency", "runtime", "wall clock", "seconds", "minutes", "ms",
)

#: Nouns that make a number a count. "60 APIs versus 12" is not a benchmark jump.
_COUNT_NOUNS = (
    "gpu", "week", "year", "day", "hour", "paper", "model", "baseline", "api",
    "language", "dataset", "task", "seed", "epoch", "layer", "author", "participant",
    "annotator", "benchmark", "example", "token", "parameter", "step", "domain",
    "category", "expert", "human", "subject", "sample", "demonstration", "trial",
)


def _window(text: str, start: int, end: int, before: int = 80, after: int = 45) -> str:
    return text[max(0, start - before) : min(len(text), end + after)].lower()


def _has(words: tuple[str, ...], haystack: str) -> bool:
    return any(word in haystack for word in words)


def _is_metric_pair(text: str, match: re.Match[str], before: float, after: float) -> bool:
    """Guard against reading two ordinary counts as a benchmark jump.

    Without this, "the study covers 60 APIs versus 12 in prior benchmarks" scores as a
    48 point gain and inflates the paper straight into the shortlist.
    """
    matched = match.group(0).lower()
    if "%" in matched:
        return True
    if _has(_COUNT_NOUNS, matched):
        return False
    context = _window(text, match.start(), match.end())
    if _has(_METRIC_WORDS, context):
        return True
    # Two decimals in a row read as scores. Two round integers usually do not.
    return before != int(before) and after != int(after)


@dataclass
class BenchmarkClaims:
    claims: list[dict] = field(default_factory=list)
    max_point_gain: float | None = None
    max_relative_gain: float | None = None
    note: str = ""

    @property
    def has_numbers(self) -> bool:
        return bool(self.claims)


def parse_benchmark_claims(text: str) -> BenchmarkClaims:
    """Pull claimed jumps out of an abstract.

    A 0.4 point gain and a 30 point gain are different species of result, and the
    absence of any number is itself informative.
    """
    text = text or ""
    claims: list[dict] = []

    for match in _FROM_TO_RE.finditer(text):
        before, after = float(match.group(1)), float(match.group(2))
        if after > before and before >= 0:
            # Percentage-scale numbers give a points gain. Anything larger is a rate,
            # so "from 340 to 1180 tokens per second" is 3.5x rather than 840 points.
            if after <= 100:
                if _is_metric_pair(text, match, before, after):
                    claims.append({"kind": "from_to", "before": before, "after": after,
                                   "points": round(after - before, 3), "text": match.group(0)})
            elif _has(_RATE_WORDS, _window(text, match.start(), match.end())):
                claims.append({"kind": "rate_gain", "before": before, "after": after,
                               "multiple": round(after / before, 3) if before else None,
                               "text": match.group(0)})
        elif before > after and after >= 0 and _looks_like_cost(text, match.start()):
            claims.append({"kind": "cost_drop", "before": before, "after": after,
                           "multiple": round(before / after, 3) if after else None,
                           "text": match.group(0)})

    for match in _VS_PREVIOUS_RE.finditer(text):
        ours, theirs = float(match.group(1)), float(match.group(2))
        if ours > theirs and ours <= 100 and theirs >= 0 and _is_metric_pair(text, match, theirs, ours):
            claims.append({"kind": "vs_previous", "before": theirs, "after": ours,
                           "points": round(ours - theirs, 3), "text": match.group(0)})

    for pattern in (_ABSOLUTE_GAIN_RE, _GAIN_PREFIX_RE):
        for match in pattern.finditer(text):
            claims.append({"kind": "absolute_gain", "points": float(match.group(1)),
                           "text": match.group(0)})

    for match in _MULTIPLE_RE.finditer(text):
        claims.append({"kind": "multiple", "multiple": float(match.group(1)),
                       "text": match.group(0)})

    for match in _REDUCTION_RE.finditer(text):
        pct = float(match.group(1))
        if 0 < pct < 100:
            claims.append({"kind": "reduction", "percent": pct,
                           "multiple": round(100 / (100 - pct), 3), "text": match.group(0)})

    claims = _dedupe(claims)
    points = [c["points"] for c in claims if c.get("points") is not None]
    multiples = [c["multiple"] for c in claims if c.get("multiple") is not None]
    note = "" if claims else "no numeric claim in the abstract"
    return BenchmarkClaims(
        claims=claims,
        max_point_gain=max(points) if points else None,
        max_relative_gain=max(multiples) if multiples else None,
        note=note,
    )


def _looks_like_cost(text: str, position: int) -> bool:
    window = text[max(0, position - 80) : position].lower()
    return any(word in window for word in
               ("cost", "latency", "memory", "time", "parameters", "flops", "energy", "error", "loss"))


def _dedupe(claims: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for claim in claims:
        key = f"{claim['kind']}:{claim.get('points')}:{claim.get('multiple')}:{claim.get('percent')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(claim)
    return out[:12]


# --- compute band -----------------------------------------------------------------

_BAND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("consumer", re.compile(
        r"\b(single (consumer|commodity|rtx|gaming)?\s*gpu|one gpu|consumer[- ]grade|"
        r"rtx\s?\d{3,4}|gtx\s?\d{3,4}|\b(3090|4090|5090|a6000)\b|laptop|edge device|"
        r"mobile device|on[- ]device|raspberry pi|cpu[- ]only|24\s?gb)\b", re.IGNORECASE)),
    ("single_node", re.compile(
        r"\b((eight|8|4|four|2|two)\s?[x×]?\s?(a100|h100|v100|a6000|l40s?|gpus?)|"
        r"single node|one node|8\s?gpus?)\b", re.IGNORECASE)),
    ("small_cluster", re.compile(
        r"\b((16|24|32|64|128)\s?[x×]?\s?(a100|h100|v100|gpus?)|small cluster|"
        r"multi[- ]node)\b", re.IGNORECASE)),
    ("frontier", re.compile(
        r"\b((256|512|1024|2048|4096)\s?[x×]?\s?(a100|h100|v100|tpu|gpus?)|"
        r"tpu\s?v[45]\s?pod|thousands of gpus|\d{3,}\s?gpu[- ]?(days|hours|years)|"
        r"pre[- ]?train(ed|ing)? (a|our) (\d+\s?b|large) (parameter )?model from scratch|"
        r"\b\d{2,}\s?trillion tokens)\b", re.IGNORECASE)),
]
_SCALE_HINT_RE = re.compile(r"\b(\d{2,3})\s?[bB]\b(?!\w)")


def estimate_compute_band(abstract: str, comments: str | None = None) -> tuple[str, str]:
    """Bands: consumer, single_node, small_cluster, frontier, unknown.

    Work that needs a frontier cluster is a lab result rather than a startup seed, so
    this filter removes a lot of noise before anything expensive runs.
    """
    text = f"{abstract or ''} {comments or ''}"
    hits = [(band, pattern.search(text)) for band, pattern in _BAND_PATTERNS]
    matched = [(band, match) for band, match in hits if match]
    if matched:
        # The largest band mentioned wins: needing 1024 GPUs somewhere beats a claim
        # that inference fits on one card.
        order = {"consumer": 0, "single_node": 1, "small_cluster": 2, "frontier": 3}
        band, match = max(matched, key=lambda pair: order[pair[0]])
        return band, f"matched {match.group(0).strip()!r}"
    scale = _SCALE_HINT_RE.search(text)
    if scale and int(scale.group(1)) >= 30 and re.search(r"train|pretrain|pre-train", text, re.I):
        return "frontier", f"training at {scale.group(0)} parameter scale"
    return "unknown", "no compute detail in the abstract"


# --- cheap structural signals ------------------------------------------------------

_SYSTEM_RE = re.compile(
    r"\b(we (build|present|deploy|ship|release) (a|an|our) (system|pipeline|platform|"
    r"framework|agent|service|tool)|end[- ]to[- ]end system|production|in deployment|"
    r"real[- ]time system|open[- ]source (library|toolkit|framework))\b", re.IGNORECASE)

_DOMAINS: dict[str, tuple[str, ...]] = {
    "healthcare": ("clinical", "medical", "patient", "radiolog", "diagnos", "ehr", "healthcare"),
    "biology": ("protein", "genomic", "molecul", "drug discovery", "cell ", "rna", "chemistry"),
    "legal": ("legal", "contract", "litigation", "compliance", "regulator"),
    "finance": ("financial", "trading", "credit", "insurance", "fraud", "accounting"),
    "robotics": ("robot", "manipulation", "locomotion", "embodied", "grasp"),
    "software": ("code generation", "software engineering", "bug", "repository", "unit test",
                 "programming", "swe-bench"),
    "security": ("malware", "vulnerabilit", "intrusion", "adversarial attack", "phishing"),
    "education": ("student", "tutoring", "classroom", "curriculum", "learner"),
    "driving": ("autonomous driving", "self-driving", "lane", "traffic", "lidar"),
    "manufacturing": ("manufactur", "industrial", "supply chain", "defect detection", "logistics"),
    "agriculture": ("crop", "agricultur", "farm", "yield prediction"),
    "science": ("materials", "physics simulation", "climate", "weather forecast", "astronom"),
}


def claims_system(abstract: str) -> bool:
    return bool(_SYSTEM_RE.search(abstract or ""))


def application_domain(title: str, abstract: str) -> str | None:
    text = f"{title or ''} {abstract or ''}".lower()
    best: tuple[int, str] | None = None
    for domain, keywords in _DOMAINS.items():
        hits = sum(text.count(word) for word in keywords)
        if hits and (best is None or hits > best[0]):
            best = (hits, domain)
    return best[1] if best else None


def novel_terms(
    conn: sqlite3.Connection,
    title: str,
    abstract: str,
    announced_date: str,
    *,
    max_paper_count: int = 3,
) -> list[str]:
    """Terms appearing for the first time in the corpus on this paper's day.

    A named method nobody has used before is weak evidence on its own and useful in
    combination with a large claim.
    """
    terms = candidate_terms(f"{title}. {abstract}")
    if not terms:
        return []
    placeholders = ", ".join("?" for _ in terms)
    rows = conn.execute(
        f"SELECT term, first_seen_date, paper_count FROM corpus_terms WHERE term IN ({placeholders})",
        terms,
    ).fetchall()
    known = {row[0]: (row[1], row[2]) for row in rows}
    fresh = []
    for term in terms:
        first_seen, count = known.get(term, (announced_date, 1))
        if first_seen >= announced_date and count <= max_paper_count:
            fresh.append(term)
    return fresh[:8]
