"""HTTP with the manners each source asks for.

Per-host rate limiting, exponential backoff on 429 and 5xx, a descriptive User-Agent,
and an optional on-disk response cache so re-running the pipeline during development
costs nothing and hits nobody's API twice.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from . import logging as log
from .config import REPO_ROOT, user_agent

LOG = log.get("http")

#: Minimum seconds between requests to a host. arXiv's rule is the strict one.
HOST_INTERVALS: dict[str, float] = {
    "export.arxiv.org": 3.0,
    "arxiv.org": 3.0,
    "api.openalex.org": 0.15,
    "api.semanticscholar.org": 1.1,
    "huggingface.co": 0.3,
    "api.github.com": 0.8,
    "hn.algolia.com": 0.3,
    "yc-oss.github.io": 0.3,
}
DEFAULT_INTERVAL = 0.5


class SourceError(RuntimeError):
    """A third-party source failed. Callers degrade the run rather than dying."""


class Fetcher:
    def __init__(
        self,
        *,
        cache_dir: str | Path | None = REPO_ROOT / "data" / "cache",
        cache_ttl_seconds: float | None = 86_400.0,
        timeout: float = 40.0,
        attempts: int = 5,
        backoff_seconds: float = 4.0,
        backoff_multiplier: float = 2.0,
        offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl_seconds
        self.attempts = attempts
        self.backoff_seconds = backoff_seconds
        self.backoff_multiplier = backoff_multiplier
        self.offline = offline
        self._last_call: dict[str, float] = {}
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": user_agent(), "Accept-Encoding": "gzip"},
        )

    # -- cache ------------------------------------------------------------------

    def _cache_path(self, url: str, params: Mapping[str, Any] | None) -> Path | None:
        if not self.cache_dir:
            return None
        key = url + "?" + json.dumps(dict(params or {}), sort_keys=True)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        host = httpx.URL(url).host or "unknown"
        return self.cache_dir / host / f"{digest}.cache"

    def _read_cache(self, path: Path | None) -> str | None:
        if not path or not path.exists():
            return None
        if self.cache_ttl is not None and time.time() - path.stat().st_mtime > self.cache_ttl:
            return None
        return path.read_text(encoding="utf-8")

    def _write_cache(self, path: Path | None, body: str) -> None:
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    # -- fetching ---------------------------------------------------------------

    def _wait_for_host(self, url: str) -> None:
        host = httpx.URL(url).host or ""
        interval = HOST_INTERVALS.get(host, DEFAULT_INTERVAL)
        last = self._last_call.get(host)
        if last is not None:
            gap = interval - (time.monotonic() - last)
            if gap > 0:
                time.sleep(gap)
        self._last_call[host] = time.monotonic()

    def get_text(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        use_cache: bool = True,
    ) -> str:
        cache_path = self._cache_path(url, params) if use_cache else None
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached
        if self.offline:
            raise SourceError(f"offline and no cached response for {url}")

        delay = self.backoff_seconds
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            self._wait_for_host(url)
            try:
                response = self._client.get(url, params=dict(params or {}), headers=dict(headers or {}))
            except httpx.HTTPError as exc:  # network level
                last_error = exc
                LOG.warning("%s failed (%s), attempt %d/%d", url, exc, attempt, self.attempts)
            else:
                if response.status_code == 200:
                    body = response.text
                    self._write_cache(cache_path, body)
                    return body
                if response.status_code in (429, 500, 502, 503, 504):
                    retry_after = response.headers.get("retry-after")
                    if retry_after and retry_after.isdigit():
                        delay = max(delay, float(retry_after))
                    last_error = SourceError(f"{response.status_code} from {url}")
                    LOG.warning(
                        "%s returned %d, attempt %d/%d", url, response.status_code, attempt, self.attempts
                    )
                else:
                    raise SourceError(f"{response.status_code} from {url}: {response.text[:200]}")
            if attempt < self.attempts:
                time.sleep(delay)
                delay *= self.backoff_multiplier
        raise SourceError(f"gave up on {url}: {last_error}")

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        use_cache: bool = True,
    ) -> Any:
        body = self.get_text(url, params, headers=headers, use_cache=use_cache)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise SourceError(f"bad JSON from {url}: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
