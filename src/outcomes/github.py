"""Repository adoption after publication.

The presence of a code link at t=0 is a day-zero feature and lives in src/features/.
Stars are the lagging half of the same story and belong here.
"""

from __future__ import annotations

import os
import re

from ..util import logging as log
from ..util.http import Fetcher, SourceError

LOG = log.get("outcomes.github")

REPO_RE = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)", re.IGNORECASE)
REPO_API = "https://api.github.com/repos/{owner}/{repo}"
SEARCH_API = "https://api.github.com/search/repositories"


def _headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token and token != "proxy-injected":
        headers["Authorization"] = f"Bearer {token}"
    return headers


def repo_from_url(url: str | None) -> tuple[str, str] | None:
    match = REPO_RE.search(url or "")
    if not match:
        return None
    return match.group(1), match.group(2).removesuffix(".git")


def repo_stats(fetcher: Fetcher, owner: str, repo: str) -> dict:
    try:
        payload = fetcher.get_json(
            REPO_API.format(owner=owner, repo=repo), headers=_headers()
        )
    except SourceError as exc:
        LOG.warning("repo lookup failed for %s/%s: %s", owner, repo, exc)
        return {"ok": False}
    return {
        "github_repo": payload.get("full_name"),
        "github_stars": int(payload.get("stargazers_count", 0)),
        "github_forks": int(payload.get("forks_count", 0)),
        "ok": True,
    }


def find_repo(fetcher: Fetcher, arxiv_id: str) -> dict:
    """Search for a repository that names the paper, for work with no link at t=0."""
    try:
        payload = fetcher.get_json(
            SEARCH_API, {"q": f"{arxiv_id} in:readme,description", "per_page": 3},
            headers=_headers(),
        )
    except SourceError as exc:
        LOG.info("repo search failed for %s: %s", arxiv_id, exc)
        return {"ok": False}
    items = payload.get("items") or []
    if not items:
        return {"github_repo": None, "github_stars": 0, "github_forks": 0, "ok": True}
    best = max(items, key=lambda item: item.get("stargazers_count", 0))
    return {
        "github_repo": best.get("full_name"),
        "github_stars": int(best.get("stargazers_count", 0)),
        "github_forks": int(best.get("forks_count", 0)),
        "ok": True,
    }


def collect(fetcher: Fetcher, arxiv_id: str, code_url: str | None) -> dict:
    parsed = repo_from_url(code_url)
    if parsed:
        stats = repo_stats(fetcher, *parsed)
        if stats.get("ok"):
            return stats
    return find_repo(fetcher, arxiv_id)
