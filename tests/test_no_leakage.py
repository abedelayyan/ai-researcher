"""The rule this project lives or dies by.

Nothing that computes a day-zero score may reach information that only exists after
publication. This test checks it three ways: by module imports, by the SQL those
modules contain, and at runtime through the feature-scoped connection.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from src.store import db

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

#: Packages that run at t=0. None of them may see an outcome.
DAY_ZERO_PACKAGES = ("features", "score", "ingest")

#: Names that only exist once the crowd has reacted.
FORBIDDEN_TOKENS = (
    "paper_outcomes",
    "outcome_labels",
    "hf_upvotes",
    "github_stars",
    "citations_openalex",
    "citations_s2",
    "hn_points",
    "stargazers_count",
)


def day_zero_modules() -> list[Path]:
    files: list[Path] = []
    for package in DAY_ZERO_PACKAGES:
        files.extend(sorted((SRC / package).rglob("*.py")))
    assert files, "expected day-zero modules to exist"
    return files


def imported_modules(tree: ast.AST, path: Path) -> set[str]:
    """Every module name a file imports, with relative imports resolved to absolute."""
    package = list(path.relative_to(REPO_ROOT).parts[:-1])  # e.g. ["src", "features"]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            if prefix:
                names.add(prefix)
                names.update(f"{prefix}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize("path", day_zero_modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_day_zero_module_does_not_import_outcomes(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for name in imported_modules(tree, path):
        parts = name.split(".")
        assert "outcomes" not in parts, f"{path} imports outcome code via {name}"
        assert "backtest" not in parts, f"{path} imports the replay harness via {name}"


@pytest.mark.parametrize("path", day_zero_modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_day_zero_module_never_names_an_outcome_field(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    # Strip docstrings and comments so prose about the rule is not mistaken for a breach.
    tree = ast.parse(source)
    code_strings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    live_strings = [s for s in code_strings if s not in docstrings]
    haystack = "\n".join(live_strings).lower()
    for token in FORBIDDEN_TOKENS:
        assert token not in haystack, f"{path} references outcome field {token!r}"


def _seed_paper(conn, arxiv_id: str) -> None:
    conn.execute(
        "INSERT INTO papers (arxiv_id, title, abstract, announced_date, source, ingested_at)"
        " VALUES (?, 't', 'a', '2026-01-01', 'test', '2026-01-01T00:00:00+00:00')",
        (arxiv_id,),
    )


def test_feature_connection_cannot_read_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "leak.db"
    with db.connect(path) as full:
        _seed_paper(full, "9999.00001")
        full.execute(
            "INSERT INTO paper_outcomes (arxiv_id, horizon_days, observed_at, hf_upvotes)"
            " VALUES ('9999.00001', 7, '2026-01-08', 42)"
        )
        full.commit()

    scoped = db.open_feature_scoped(path)
    with pytest.raises(sqlite3.DatabaseError):
        scoped.execute("SELECT hf_upvotes FROM paper_outcomes").fetchall()
    with pytest.raises(sqlite3.DatabaseError):
        scoped.execute("SELECT * FROM outcome_labels").fetchall()
    with pytest.raises(sqlite3.DatabaseError):
        scoped.execute("DELETE FROM paper_outcomes")
    # The day-zero tables stay readable.
    assert scoped.execute("SELECT count(*) FROM papers").fetchone()[0] == 1


def test_scoring_runs_on_a_scoped_connection(tmp_path: Path) -> None:
    """The scorer must accept the locked-down connection, not merely tolerate it."""
    from src.score import rank

    path = tmp_path / "scored.db"
    with db.connect(path) as full:
        _seed_paper(full, "9999.00002")
        full.execute(
            "INSERT INTO paper_outcomes (arxiv_id, horizon_days, observed_at, hf_upvotes)"
            " VALUES ('9999.00002', 7, '2026-01-08', 99)"
        )
        full.commit()
    scoped = db.open_feature_scoped(path)
    assert rank.rank_day(scoped, "2026-01-08") == {"confidence": [], "high_variance": []}
