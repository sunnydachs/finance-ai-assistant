"""Thin wrapper around the Anthropic SDK (Messages API) with usage logging.

All LLM traffic in the project funnels through here so that token usage and
costs are recorded in one place (logs/usage.jsonl) regardless of whether the
caller is the CLI app, the eval runner, or the judge.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from anthropic import Anthropic

from . import config

logger = logging.getLogger(__name__)

_client: Anthropic | None = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        if not config.API_AUTH_TOKEN:
            raise RuntimeError(
                "ANTHROPIC_AUTH_TOKEN is not set. Copy .env.example to .env and fill it in."
            )
        _client = Anthropic(
            base_url=config.API_BASE_URL,
            auth_token=config.API_AUTH_TOKEN,
            max_retries=3,
            timeout=180.0,
            default_headers={"X-Title": "finance-ai-assistant"},
        )
    return _client


def call_model(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    purpose: str,
    max_tokens: int = config.APP_MAX_TOKENS,
    temperature: float = config.APP_TEMPERATURE,
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    """Call the Messages API and log usage. Returns the raw SDK Message.

    `purpose` labels the call in the usage log (app / judge / comparison ...).
    """
    client = get_client()
    started = time.monotonic()
    # SDK 1.x dropped `temperature` from create(); sampling params go through
    # extra_body (the gateway passes them through to the model).
    kwargs: dict[str, Any] = dict(
        model=model,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        extra_body={"temperature": temperature},
    )
    if tools:
        kwargs["tools"] = tools
    message = client.messages.create(**kwargs)
    # Some gateway models occasionally return an empty/None content list;
    # retry once rather than crashing mid-eval.
    if not message.content:
        message = client.messages.create(**kwargs)
    if not message.content:
        raise RuntimeError(f"model returned empty content twice (model={model})")
    elapsed = time.monotonic() - started

    # Some gateway models don't report usage — log zeros rather than crash.
    usage = getattr(message, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) or 0
    output_tokens = getattr(usage, "output_tokens", None) or 0

    _log_usage(
        model=model,
        purpose=purpose,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_s=round(elapsed, 2),
        stop_reason=message.stop_reason,
    )
    return message


def _log_usage(
    *, model: str, purpose: str, input_tokens: int, output_tokens: int,
    elapsed_s: float, stop_reason: str,
) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "purpose": purpose,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": elapsed_s,
        "stop_reason": stop_reason,
        "cost_usd": round(
            config.model_cost_usd(model, input_tokens, output_tokens), 6
        ),
    }
    try:
        config.USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with config.USAGE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Usage logging must never break the actual call.
        logger.warning("could not write usage log", exc_info=True)


def text_of(message: Any) -> str:
    """Concatenate all text blocks of a Message into one string."""
    return "".join(
        block.text for block in (message.content or []) if block.type == "text"
    )
