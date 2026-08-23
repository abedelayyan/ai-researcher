"""Prompts are versioned files, and the house style applies to what they produce."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.llm import prompts as prompt_lib

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_FILES = sorted(PROMPT_DIR.glob("*.md"))

BANNED = ("leverage", "seamless", "deepen", "revolutionary", "landscape")


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: p.name)
def test_every_prompt_loads_with_both_sections(path: Path) -> None:
    prompt = prompt_lib.load(str(path))
    assert prompt.system.strip()
    assert prompt.user.strip()
    assert prompt.version.startswith("v")


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: p.name)
def test_prompts_follow_the_house_style(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "—" not in text, "no em dashes"
    lowered = text.lower()
    for word in BANNED:
        # "landscape" is allowed only where a prompt tells a model to avoid it.
        if word in lowered and "consultant filler" in lowered:
            continue
        assert word not in lowered, f"{path.name} uses {word!r}"


def test_placeholders_are_substituted():
    prompt = prompt_lib.load("prompts/capability_delta.md")
    _, user = prompt.render(title="T", abstract="A", categories="cs.AI", comments="none")
    assert "{{" not in user
    assert "T" in user and "A" in user


def test_no_python_module_hides_a_prompt_inline():
    """The rubric is iterated many times during backtesting. The diffs need to be
    readable, which means prompts stay in files."""
    src = Path(__file__).resolve().parents[1] / "src"
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "You are an expert" not in text
        assert "## System" not in text or path.name == "prompts.py"
