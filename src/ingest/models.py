"""The paper record as it exists at t=0."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v(\d+))?")


def split_arxiv_id(raw: str) -> tuple[str, int | None]:
    """``http://arxiv.org/abs/2408.01234v2`` becomes ``("2408.01234", 2)``."""
    match = ARXIV_ID_RE.search(raw or "")
    if not match:
        return raw.strip(), None
    return match.group(1), int(match.group(2)) if match.group(2) else None


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    announced_date: str
    source: str
    version: int | None = None
    comments: str | None = None
    journal_ref: str | None = None
    doi: str | None = None
    primary_category: str | None = None
    categories: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    submitted_at: str | None = None
    updated_at: str | None = None
    abs_url: str | None = None
    pdf_url: str | None = None
    links: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "version": self.version,
            "title": self.title,
            "abstract": self.abstract,
            "comments": self.comments,
            "journal_ref": self.journal_ref,
            "doi": self.doi,
            "primary_category": self.primary_category,
            "categories": json.dumps(self.categories),
            "authors": json.dumps(self.authors, ensure_ascii=False),
            "author_count": len(self.authors),
            "affiliations": json.dumps(self.affiliations, ensure_ascii=False),
            "submitted_at": self.submitted_at,
            "updated_at": self.updated_at,
            "announced_date": self.announced_date,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
            "links": json.dumps(self.links),
            "source": self.source,
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
