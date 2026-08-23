"""One logging setup for every entry point."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def setup(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, (level or os.environ.get("LOG_LEVEL", "INFO")).upper(), 20),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(name)
