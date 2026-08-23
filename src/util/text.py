"""Text helpers shared by ingest and feature extraction."""

from __future__ import annotations

import re

#: Method names in this field look like FlashAttention, LoRA, RT-2, Mamba-2.
_CAMEL = re.compile(r"\b[A-Z][a-z]+(?:[A-Z]+[a-z0-9]*)+\b")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{2,6}\b")
_HYPHENATED = re.compile(r"\b[A-Za-z][A-Za-z0-9]+-[A-Za-z0-9]+\b")

#: Words that look like terms but carry no signal.
_STOP_TERMS = {
    "the", "and", "for", "with", "state-of-the-art", "sota", "gpu", "cpu", "llm", "llms",
    "nlp", "cnn", "rnn", "gan", "vae", "mlp", "api", "http", "https", "arxiv", "github",
    "ai", "ml", "rl", "sgd", "lstm", "bert", "gpt", "code", "data", "pre-trained",
    "fine-tuned", "open-source", "real-world", "large-scale", "end-to-end", "multi-modal",
    "self-supervised", "zero-shot", "few-shot", "in-context", "high-quality", "long-term",
}


def candidate_terms(text: str, *, limit: int = 40) -> list[str]:
    """Pull method-name shaped tokens out of a title or abstract.

    Deliberately narrow. A full vocabulary would be mostly ordinary English, and the
    signal we want is the first appearance of a named method.
    """
    found: list[str] = []
    seen: set[str] = set()
    for pattern in (_CAMEL, _ACRONYM, _HYPHENATED):
        for match in pattern.findall(text or ""):
            term = match.strip("-").lower()
            if len(term) < 3 or term in _STOP_TERMS or term in seen:
                continue
            if term.replace("-", "").isdigit():
                continue
            seen.add(term)
            found.append(term)
            if len(found) >= limit:
                return found
    return found


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"
