from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.util.config import load as load_config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_atom() -> str:
    return (FIXTURES / "arxiv_sample.xml").read_text(encoding="utf-8")


@pytest.fixture
def config(tmp_path: Path) -> dict:
    """The real config, pointed at a throwaway database and forced offline."""
    cfg = copy.deepcopy(load_config())
    cfg["storage"]["db_path"] = str(tmp_path / "test.db")
    cfg["render"]["digest_dir"] = str(tmp_path / "digest")
    cfg["render"]["site_dir"] = str(tmp_path / "site")
    cfg["llm"]["fallback_order"] = ["heuristic"]
    cfg["llm"]["spend_log"] = str(tmp_path / "spend.jsonl")
    return cfg
