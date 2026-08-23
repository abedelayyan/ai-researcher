"""Load config.yaml once and hand out sections."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


@lru_cache(maxsize=8)
def load(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("SIGNAL_ZERO_CONFIG") or DEFAULT_CONFIG)
    with open(cfg_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get(dotted: str, default: Any = None, *, config: dict | None = None) -> Any:
    """Fetch a nested key, e.g. ``get("score.weights.author_prior")``."""
    node: Any = config if config is not None else load()
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def db_path(config: dict | None = None) -> Path:
    return REPO_ROOT / get("storage.db_path", "data/signal.db", config=config)


def user_agent(config: dict | None = None) -> str:
    return get("project.contact", "signal-zero", config=config)


def contact_email(config: dict | None = None) -> str | None:
    """The mailto address inside the contact string, for OpenAlex's polite pool."""
    contact = user_agent(config)
    if "mailto:" not in contact:
        return None
    return contact.split("mailto:", 1)[1].strip(" )>;")
