"""Tiered model access with graceful degradation.

Cheap tier does feature extraction and summaries for 30 to 50 papers a day. Strong tier
is reserved for weekly cluster synthesis and the critic.

Providers are tried in the order set by ``llm.fallback_order`` and the first one holding
credentials wins. ``heuristic`` sits at the end of that list: it needs no key and no
network, returns rule-based answers, and keeps a run alive when every API is refusing.
A run that fell back to heuristics says so in the digest, so the prediction log never
silently changes meaning.

Prices move monthly. The table below is mid-2026 list pricing and exists to make spend
visible, not to be authoritative. Re-check before trusting a budget.
"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import httpx

from ..util import dates
from ..util import logging as log
from ..util.config import REPO_ROOT, load as load_config

LOG = log.get("llm")

# USD per million tokens, (input, output).
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-4.1-mini": (0.40, 1.60),
}


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    ok: bool = True
    error: str | None = None
    purpose: str = ""

    @property
    def source(self) -> str:
        return f"{self.provider}:{self.model}"


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def price_for(model: str, input_tokens: int, output_tokens: int) -> float:
    for key, (per_in, per_out) in PRICES.items():
        if key in model:
            return (input_tokens * per_in + output_tokens * per_out) / 1_000_000
    return 0.0


def extract_json(text: str) -> dict:
    """Models wrap JSON in prose and fences more often than they should."""
    if not text:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in response")
        candidate = text[start : end + 1]
    return json.loads(candidate)


# --- providers ------------------------------------------------------------------


class Provider:
    name = "base"

    def available(self) -> bool:
        return False

    def complete(self, system: str, user: str, *, model: str, max_tokens: int,
                 temperature: float, purpose: str) -> LLMResult:
        raise NotImplementedError


class OpenAICompatibleProvider(Provider):
    """Covers GitHub Models and the OpenAI API, which share a request shape."""

    def __init__(self, name: str, base_url: str, env_keys: Sequence[str], timeout: float) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.env_keys = list(env_keys)
        self.timeout = timeout

    def _token(self) -> str | None:
        for key in self.env_keys:
            value = os.environ.get(key)
            if value and value != "proxy-injected":
                return value
        return None

    def available(self) -> bool:
        return self._token() is not None

    def complete(self, system: str, user: str, *, model: str, max_tokens: int,
                 temperature: float, purpose: str) -> LLMResult:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage", {}) or {}
        text = (body["choices"][0]["message"]["content"] or "").strip()
        in_tok = int(usage.get("prompt_tokens") or estimate_tokens(system + user))
        out_tok = int(usage.get("completion_tokens") or estimate_tokens(text))
        return LLMResult(
            text=text,
            provider=self.name,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=0.0 if self.name == "github_models" else price_for(model, in_tok, out_tok),
            purpose=purpose,
        )


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str, *, model: str, max_tokens: int,
                 temperature: float, purpose: str) -> LLMResult:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # The system block is identical across every paper in a run, so caching it
            # is where most of the saving on a daily batch comes from.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        text = "".join(block.get("text", "") for block in body.get("content", [])).strip()
        usage = body.get("usage", {}) or {}
        in_tok = int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0))
        return LLMResult(
            text=text,
            provider=self.name,
            model=model,
            input_tokens=in_tok or estimate_tokens(system + user),
            output_tokens=out_tok or estimate_tokens(text),
            cost_usd=price_for(model, in_tok, out_tok),
            purpose=purpose,
        )


class HeuristicProvider(Provider):
    """No key, no network, deterministic. The last line of defence."""

    name = "heuristic"

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[str], str]] = {}

    def register(self, purpose: str, handler: Callable[[str], str]) -> None:
        self._handlers[purpose] = handler

    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, *, model: str, max_tokens: int,
                 temperature: float, purpose: str) -> LLMResult:
        handler = self._handlers.get(purpose)
        if handler is None:
            raise RuntimeError(f"no heuristic fallback registered for purpose {purpose!r}")
        return LLMResult(
            text=handler(user),
            provider=self.name,
            model="rules",
            input_tokens=estimate_tokens(user),
            output_tokens=0,
            cost_usd=0.0,
            purpose=purpose,
        )


# --- client ---------------------------------------------------------------------


@dataclass
class Client:
    config: dict = field(default_factory=load_config)
    providers: list[Provider] = field(default_factory=list)
    calls: list[LLMResult] = field(default_factory=list)
    heuristic: HeuristicProvider = field(default_factory=HeuristicProvider)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.providers:
            return
        llm_cfg = self.config.get("llm", {})
        timeout = float(llm_cfg.get("request_timeout_seconds", 90))
        catalogue: dict[str, Provider] = {
            "github_models": OpenAICompatibleProvider(
                "github_models",
                "https://models.github.ai/inference",
                ["GITHUB_MODELS_TOKEN", "MODELS_TOKEN", "GITHUB_TOKEN"],
                timeout,
            ),
            "openai": OpenAICompatibleProvider(
                "openai", "https://api.openai.com/v1", ["OPENAI_API_KEY"], timeout
            ),
            "anthropic": AnthropicProvider(timeout),
            "heuristic": self.heuristic,
        }
        order = llm_cfg.get("fallback_order", ["heuristic"])
        self.providers = [catalogue[name] for name in order if name in catalogue]

    # -- single call ------------------------------------------------------------

    def complete(self, purpose: str, system: str, user: str, *, tier: str = "cheap") -> LLMResult:
        tier_cfg = (self.config.get("llm", {}).get("tiers", {}) or {}).get(tier, {})
        preferred = tier_cfg.get("provider")
        max_tokens = int(tier_cfg.get("max_output_tokens", 900))
        temperature = float(tier_cfg.get("temperature", 0.0))
        retries = int(self.config.get("llm", {}).get("max_retries", 3))

        ordered = sorted(self.providers, key=lambda p: 0 if p.name == preferred else 1)
        last_error: Exception | None = None
        for provider in ordered:
            if not provider.available():
                continue
            model = tier_cfg.get("model", "") if provider.name == preferred else _default_model(provider.name, tier_cfg)
            for attempt in range(1, retries + 1):
                try:
                    result = provider.complete(
                        system, user, model=model, max_tokens=max_tokens,
                        temperature=temperature, purpose=purpose,
                    )
                except Exception as exc:  # noqa: BLE001 - any provider failure falls through
                    last_error = exc
                    LOG.warning("%s failed on %s (attempt %d): %s", provider.name, purpose, attempt, exc)
                    if not _worth_retrying(exc):
                        break
                else:
                    self._record(result)
                    return result
        failure = LLMResult(
            text="", provider="none", model="", ok=False,
            error=str(last_error) if last_error else "no provider available", purpose=purpose,
        )
        self._record(failure)
        return failure

    # -- batched calls ----------------------------------------------------------

    def complete_many(
        self,
        purpose: str,
        requests: Sequence[tuple[str, str]],
        *,
        tier: str = "cheap",
        concurrency: int = 2,
    ) -> list[LLMResult]:
        """Run a day's worth of calls together.

        Concurrency is deliberately low. The free GitHub Models tier allows two
        concurrent requests, and nothing here is latency sensitive.
        """
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            return list(pool.map(lambda pair: self.complete(purpose, pair[0], pair[1], tier=tier), requests))

    # -- spend ------------------------------------------------------------------

    def _record(self, result: LLMResult) -> None:
        with self._lock:
            self.calls.append(result)
        path = self.config.get("llm", {}).get("spend_log")
        if not path:
            return
        target = Path(path)
        if not target.is_absolute():
            target = REPO_ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": dates.utcnow().isoformat(timespec="seconds"),
            "purpose": result.purpose,
            "provider": result.provider,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": round(result.cost_usd, 6),
            "ok": result.ok,
            "error": result.error,
        }
        with self._lock, open(target, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def spend_summary(self) -> dict:
        return {
            "calls": len(self.calls),
            "failed": sum(1 for c in self.calls if not c.ok),
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "cost_usd": round(sum(c.cost_usd for c in self.calls), 4),
            "providers": sorted({c.source for c in self.calls if c.ok}),
        }

    def call_rows(self) -> list[dict]:
        return [
            {
                "purpose": c.purpose, "provider": c.provider, "model": c.model,
                "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                "cost_usd": c.cost_usd, "ok": c.ok, "error": c.error,
            }
            for c in self.calls
        ]


def _default_model(provider: str, tier_cfg: dict) -> str:
    """When falling back, pick something sane rather than the preferred tier's name."""
    defaults = {
        "github_models": "openai/gpt-4o-mini",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
        "heuristic": "rules",
    }
    return defaults.get(provider, tier_cfg.get("model", ""))


def _worth_retrying(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 409, 425, 429, 500, 502, 503, 504)
    return isinstance(exc, httpx.HTTPError)
