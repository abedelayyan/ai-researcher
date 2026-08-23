"""Turn a scored day into a digest, in markdown and static HTML."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Sequence

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..score.rank import ScoredPaper
from ..store import db
from ..util import logging as log
from ..util.config import REPO_ROOT, load as load_config

LOG = log.get("render.digest")

TEMPLATE_DIR = Path(__file__).parent / "templates"

AXIS_LABELS = {
    "cost_curve": "cost",
    "constraint_removal": "constraint",
    "usability_threshold": "usability",
    "modality_opening": "modality",
}


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _authors_text(authors: Sequence[str]) -> str:
    names = list(authors)
    if not names:
        return "authors unlisted"
    if len(names) <= 3:
        return ", ".join(names)
    remaining = len(names) - 3
    return f"{', '.join(names[:3])} and {remaining} other" + ("s" if remaining > 1 else "")


def _claim_text(row: dict) -> str:
    points, multiple = row.get("max_point_gain"), row.get("max_relative_gain")
    claims = db.loads(row.get("benchmark_claims"), [])
    if points and multiple:
        return f"Claims {points:g} points over prior art and {multiple:g}x on a cost axis."
    if points:
        return f"Claims {points:g} points over the stated prior art."
    if multiple:
        return f"Claims {multiple:g}x on a cost or speed axis."
    if claims:
        return "Numbers claimed, none parsed as a jump over prior art."
    return "No numeric claim in the abstract."


def _code_text(row: dict) -> str:
    if row.get("code_url"):
        return row["code_url"]
    if row.get("code_link_source") == "promise":
        return "promised, no link yet"
    return "none at publication"


def _prior_text(row: dict) -> str:
    prior = row.get("author_prior")
    if prior is None:
        return "unknown"
    return f"{prior:.2f}" + ("" if row.get("author_prior_coverage") else " (thin record)")


def _compute_text(row: dict) -> str:
    band = (row.get("compute_band") or "unknown").replace("_", " ")
    why = (row.get("compute_band_why") or "").strip()
    return f"Compute band: {band}" + (f" ({why})" if why else "")


def paper_view(paper: ScoredPaper, summary: str) -> dict:
    row = paper.row
    axes = [(AXIS_LABELS[a], row.get(a) or 0) for a in AXIS_LABELS]
    reasons = [
        (AXIS_LABELS[axis], (row.get(f"{axis}_why") or "").strip())
        for axis in AXIS_LABELS
        if (row.get(axis) or 0) > 0 and (row.get(f"{axis}_why") or "").strip()
    ]
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "summary": summary,
        "score": paper.score,
        "claim_score": paper.claim_score,
        "capability_total": row.get("capability_total") or 0,
        "axis_line": ", ".join(f"{name} {value}" for name, value in axes),
        "axis_reasons": reasons,
        "compute_band": (row.get("compute_band") or "unknown").replace("_", " "),
        "compute_band_why": row.get("compute_band_why") or "",
        "compute_text": _compute_text(row),
        "claim_text": _claim_text(row),
        "code_text": _code_text(row),
        "prior_text": _prior_text(row),
        "authors_text": _authors_text(db.loads(row.get("authors"), [])),
        "categories_text": ", ".join(db.loads(row.get("categories"), [])[:4]) or row.get("primary_category") or "",
        "abs_url": row.get("abs_url") or f"https://arxiv.org/abs/{paper.arxiv_id}",
    }


def build_context(
    *,
    date: str,
    buckets: dict[str, list[ScoredPaper]],
    summaries: dict[str, str],
    scored_count: int,
    config: dict,
    calibration: dict,
    lab_posts: Sequence[dict] = (),
    spend: dict | None = None,
    run_id: int | str = "-",
) -> dict:
    views = {
        name: [paper_view(p, summaries.get(p.arxiv_id, "")) for p in papers]
        for name, papers in buckets.items()
    }
    shortlisted = [p for papers in buckets.values() for p in papers]
    degraded = any(
        (p.row.get("capability_source") or "").startswith("heuristic") for p in shortlisted
    )
    return {
        "date": date,
        "buckets": views,
        "counts": {"scored": scored_count, "shortlisted": len(shortlisted)},
        "categories": config.get("ingest", {}).get("categories", []),
        "calibration": calibration,
        "label_horizon": config.get("outcomes", {}).get("composite", {}).get("horizon_days", 30),
        "lab_posts": list(lab_posts),
        "spend": spend or {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "run_id": run_id,
        "degraded": degraded,
    }


def render_markdown(context: dict) -> str:
    text = _environment().get_template("digest.md.j2").render(**context)
    # Collapse the blank-line runs Jinja leaves behind.
    lines, out = text.splitlines(), []
    blanks = 0
    for line in lines:
        blanks = blanks + 1 if not line.strip() else 0
        if blanks < 3:
            out.append(line.rstrip())
    return "\n".join(out).strip() + "\n"


def render_html(context: dict, recent_dates: Sequence[str] = ()) -> str:
    return _environment().get_template("page.html.j2").render(
        recent_dates=list(recent_dates), **context
    )


def write(context: dict, config: dict | None = None, *, recent_dates: Sequence[str] = ()) -> dict[str, Path]:
    """Write the markdown digest and refresh the static site."""
    config = config or load_config()
    render_cfg = config.get("render", {})
    digest_dir = REPO_ROOT / render_cfg.get("digest_dir", "data/digest")
    site_dir = REPO_ROOT / render_cfg.get("site_dir", "site")
    digest_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "digest").mkdir(parents=True, exist_ok=True)

    md_path = digest_dir / f"{context['date']}.md"
    md_path.write_text(render_markdown(context), encoding="utf-8")

    html = render_html(context, recent_dates)
    page_path = site_dir / "digest" / f"{context['date']}.html"
    page_path.write_text(html, encoding="utf-8")
    index_path = site_dir / "index.html"
    shutil.copyfile(page_path, index_path)

    try:
        shown = md_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = md_path
    LOG.info("wrote %s", shown)
    return {"markdown": md_path, "html": page_path, "index": index_path}


def recent_digest_dates(config: dict | None = None, limit: int = 10) -> list[str]:
    config = config or load_config()
    digest_dir = REPO_ROOT / config.get("render", {}).get("digest_dir", "data/digest")
    if not digest_dir.exists():
        return []
    dates = sorted((p.stem for p in digest_dir.glob("*.md")), reverse=True)
    return dates[:limit]


def lab_posts_for(conn: sqlite3.Connection, date: str, limit: int = 5) -> list[dict]:
    rows = conn.execute(
        "SELECT lab, title, url FROM lab_posts WHERE substr(coalesce(published, ingested_at), 1, 10) >= date(?, '-2 days')"
        " ORDER BY published DESC LIMIT ?",
        (date, limit),
    ).fetchall()
    return [dict(row) for row in rows]
