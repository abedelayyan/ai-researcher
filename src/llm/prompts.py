"""Prompt templates live in prompts/ as versioned files, never inline in Python.

A template is a markdown file with a `## System` section and a `## User` section.
Placeholders are `{{name}}`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..util.config import REPO_ROOT

_SECTION = re.compile(r"^##\s+(System|User)\s*$", re.IGNORECASE | re.MULTILINE)
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str
    user: str

    def render(self, **values: object) -> tuple[str, str]:
        def substitute(text: str) -> str:
            return _PLACEHOLDER.sub(lambda m: str(values.get(m.group(1), "")), text)

        return substitute(self.system), substitute(self.user)


@lru_cache(maxsize=32)
def load(path: str) -> Prompt:
    full = Path(path)
    if not full.is_absolute():
        full = REPO_ROOT / path
    text = full.read_text(encoding="utf-8")

    header = text.splitlines()[0].lstrip("# ").strip() if text.strip() else full.stem
    name, _, version = header.partition(" ")
    version = version.split()[0] if version else "v0"

    parts = _SECTION.split(text)
    sections = {}
    for index in range(1, len(parts) - 1, 2):
        sections[parts[index].lower()] = parts[index + 1].strip()
    if "system" not in sections or "user" not in sections:
        raise ValueError(f"{full} needs both a '## System' and a '## User' section")
    return Prompt(name=name or full.stem, version=version, system=sections["system"], user=sections["user"])
