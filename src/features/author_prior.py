"""Author priors from OpenAlex.

The question is not how famous an author is. It is whether their previous work went on
to accumulate citations quickly, which is a rough proxy for whether the field picks up
what they publish.

Two properties matter here:

* **Date bounding.** Citation counts are summed from OpenAlex's per-year breakdown up
  to a cutoff, never from the current total. Without that the backtest would score a
  2024 paper using citations earned in 2026, which is exactly the leak this project
  exists to avoid.
* **Null, not zero.** An author with no usable record gets a null prior and the paper
  routes to the high-variance bucket. Scoring them zero would bury unknown authors,
  and unknown authors making large claims are the interesting case.
"""

from __future__ import annotations

import sqlite3
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from ..store import db
from ..util import dates
from ..util import logging as log
from ..util.config import contact_email
from ..util.http import Fetcher, SourceError

LOG = log.get("features.author_prior")

WORKS_URL = "https://api.openalex.org/works"
SELECT_FIELDS = "id,publication_year,publication_date,cited_by_count,counts_by_year"


@dataclass
class AuthorPrior:
    author_key: str
    display_name: str
    prior: float | None
    coverage: bool
    works_count: int = 0
    citations_total: int = 0
    citations_per_year: float = 0.0
    first_year: int | None = None


@dataclass
class PaperPrior:
    prior: float | None
    coverage: bool
    detail: dict


def normalise_name(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", name or "")
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return " ".join(stripped.lower().replace(".", " ").split())


def select_authors(authors: Sequence[str], limit: int) -> list[str]:
    """First and last author carry most of the signal, so keep both ends."""
    if len(authors) <= limit:
        return list(authors)
    head = authors[: max(1, limit - 1)]
    return list(head) + [authors[-1]]


class AuthorPriorService:
    def __init__(self, conn: sqlite3.Connection, fetcher: Fetcher, config: dict) -> None:
        self.conn = conn
        self.fetcher = fetcher
        cfg = config.get("author_prior", {})
        self.refresh_days = int(cfg.get("refresh_days", 30))
        self.works_per_author = int(cfg.get("works_per_author", 50))
        self.lookback_years = int(cfg.get("lookback_years", 6))
        self.max_authors = int(cfg.get("max_authors_per_paper", 6))
        self.saturation = float(cfg.get("saturation_citations_per_year", 40.0))
        self.min_works = int(cfg.get("min_works_for_coverage", 3))
        self.email = contact_email(config)
        self.offline = False

    # -- public ----------------------------------------------------------------

    def for_paper(self, authors: Sequence[str], cutoff_date: str | None = None) -> PaperPrior:
        """Prior for a paper, from the strongest author on it.

        cutoff_date is '' for a live run and the publication date for a replay.
        """
        cutoff = cutoff_date or ""
        picked = select_authors(list(authors), self.max_authors)
        priors = [self.for_author(name, cutoff) for name in picked]
        known = [p for p in priors if p.prior is not None]
        if not known:
            return PaperPrior(prior=None, coverage=False,
                              detail={"authors_checked": len(picked), "with_record": 0})
        best = max(known, key=lambda p: p.prior or 0.0)
        mean = sum(p.prior or 0.0 for p in known) / len(known)
        return PaperPrior(
            prior=best.prior,
            coverage=any(p.coverage for p in known),
            detail={
                "authors_checked": len(picked),
                "with_record": len(known),
                "best_author": best.display_name,
                "best_prior": round(best.prior or 0.0, 4),
                "mean_prior": round(mean, 4),
                "best_citations_per_year": round(best.citations_per_year, 2),
                "cutoff": cutoff or "live",
            },
        )

    def for_author(self, name: str, cutoff_date: str = "") -> AuthorPrior:
        key = normalise_name(name)
        cached = self._read_cache(key, cutoff_date)
        if cached is not None:
            return cached
        try:
            computed = self._compute(name, key, cutoff_date)
        except SourceError as exc:
            LOG.warning("OpenAlex unavailable for %s: %s", name, exc)
            return AuthorPrior(author_key=key, display_name=name, prior=None, coverage=False)
        self._write_cache(computed, cutoff_date)
        return computed

    # -- internals -------------------------------------------------------------

    def _read_cache(self, key: str, cutoff_date: str) -> AuthorPrior | None:
        row = self.conn.execute(
            "SELECT * FROM author_priors WHERE author_key = ? AND cutoff_date = ?",
            (key, cutoff_date),
        ).fetchone()
        if row is None:
            return None
        # A dated backtest prior never goes stale. A live one is refreshed monthly.
        if not cutoff_date:
            computed = dates.parse_iso(row["computed_at"])
            if computed and (dates.utcnow() - computed).days > self.refresh_days:
                return None
        return AuthorPrior(
            author_key=row["author_key"],
            display_name=row["display_name"] or row["author_key"],
            prior=row["prior"],
            coverage=bool(row["coverage"]),
            works_count=row["works_count"] or 0,
            citations_total=row["citations_total"] or 0,
            citations_per_year=row["citations_per_year"] or 0.0,
            first_year=row["first_year"],
        )

    def _write_cache(self, prior: AuthorPrior, cutoff_date: str) -> None:
        db.upsert(
            self.conn,
            "author_priors",
            {
                "author_key": prior.author_key,
                "cutoff_date": cutoff_date,
                "display_name": prior.display_name,
                "works_count": prior.works_count,
                "citations_total": prior.citations_total,
                "citations_per_year": prior.citations_per_year,
                "first_year": prior.first_year,
                "prior": prior.prior,
                "coverage": 1 if prior.coverage else 0,
                "detail": db.dumps({}),
                "computed_at": dates.utcnow().isoformat(timespec="seconds"),
            },
            keys=["author_key", "cutoff_date"],
        )
        self.conn.commit()

    def _compute(self, name: str, key: str, cutoff_date: str) -> AuthorPrior:
        cutoff = dates.to_date(cutoff_date) if cutoff_date else dates.utcnow().date()
        earliest = cutoff.replace(year=cutoff.year - self.lookback_years)
        params = {
            "filter": (
                f"raw_author_name.search:{name},"
                f"from_publication_date:{earliest.isoformat()},"
                f"to_publication_date:{cutoff.isoformat()}"
            ),
            "select": SELECT_FIELDS,
            "per-page": min(self.works_per_author, 100),
            "sort": "publication_date:desc",
        }
        if self.email:
            params["mailto"] = self.email
        payload = self.fetcher.get_json(WORKS_URL, params)
        works = payload.get("results", []) or []
        return summarise_works(name, key, works, cutoff.year, saturation=self.saturation,
                               min_works=self.min_works)


def citations_by(work: dict, cutoff_year: int) -> int:
    """Citations this work had accumulated by the end of the cutoff year.

    OpenAlex's cited_by_count is the count today, which for a replay is the future.
    """
    counts = work.get("counts_by_year") or []
    if not counts:
        return 0
    return sum(int(entry.get("cited_by_count", 0)) for entry in counts
               if int(entry.get("year", 0)) <= cutoff_year)


def summarise_works(
    name: str,
    key: str,
    works: Sequence[dict],
    cutoff_year: int,
    *,
    saturation: float,
    min_works: int,
) -> AuthorPrior:
    """Turn an author's work list into a 0..1 prior on citation velocity."""
    velocities: list[float] = []
    total = 0
    first_year: int | None = None
    for work in works:
        year = work.get("publication_year")
        if not year:
            continue
        first_year = year if first_year is None else min(first_year, int(year))
        citations = citations_by(work, cutoff_year)
        total += citations
        age = max(0.5, (cutoff_year - int(year)) + 0.5)
        velocities.append(citations / age)

    if not velocities:
        return AuthorPrior(author_key=key, display_name=name, prior=None, coverage=False)

    velocities.sort(reverse=True)
    # Top three works, so one strong result is not diluted by a long tail.
    headline = sum(velocities[:3]) / min(3, len(velocities))
    prior = min(1.0, headline / saturation) if saturation > 0 else 0.0
    coverage = len(velocities) >= min_works
    if not coverage and headline <= 0:
        # Thin record with nothing behind it. Unknown, not bad.
        return AuthorPrior(author_key=key, display_name=name, prior=None, coverage=False,
                           works_count=len(velocities), first_year=first_year)
    return AuthorPrior(
        author_key=key,
        display_name=name,
        prior=round(prior, 4),
        coverage=coverage,
        works_count=len(velocities),
        citations_total=total,
        citations_per_year=round(headline, 3),
        first_year=first_year,
    )
