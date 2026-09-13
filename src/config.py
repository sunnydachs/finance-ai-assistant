"""Project configuration: env loading, model IDs, paths, pricing.

All secrets come from environment variables (loaded from .env, which is
git-ignored). No key material is ever written to source files.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- API access -------------------------------------------------------------
# The Anthropic SDK is pointed at an Anthropic-Messages-compatible gateway
# (OpenRouter). Switching to the first-party Anthropic API is a .env-only
# change: swap BASE_URL/AUTH_TOKEN/MODEL_* values.
API_BASE_URL: str = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
API_AUTH_TOKEN: str = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")

APP_MODEL: str = os.environ.get("APP_MODEL", "inclusionai/ling-3.0-flash-fin:free")
JUDGE_MODEL: str = os.environ.get("JUDGE_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus"
FAQ_PATH = CORPUS_DIR / "faq.jsonl"
NOTES_DIR = CORPUS_DIR / "notes"
LOGS_DIR = PROJECT_ROOT / "logs"
USAGE_LOG_PATH = LOGS_DIR / "usage.jsonl"
TOOL_LOG_PATH = LOGS_DIR / "tool_calls.jsonl"
EVALS_DIR = PROJECT_ROOT / "evals"
GOLDEN_PATH = EVALS_DIR / "golden.jsonl"
EVAL_RUNS_DIR = EVALS_DIR / "runs"
EVAL_REPORTS_DIR = EVALS_DIR / "reports"

# --- Generation parameters ---------------------------------------------------
APP_TEMPERATURE = 0.2          # low but not 0: factual answers, less repetition
APP_MAX_TOKENS = 1200
JUDGE_TEMPERATURE = 0.0        # determinism first for scoring
JUDGE_MAX_TOKENS = 700
RETRIEVAL_TOP_K = 4            # docs (or chunks) handed to the model

# --- Cost accounting (USD per 1M tokens) -------------------------------------
# Free-tier gateway models cost 0; the table keeps cost reporting honest if
# models are swapped later via .env.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "inclusionai/ling-3.0-flash-fin:free": (0.0, 0.0),
    "inclusionai/ling-3.0-flash-vl:free": (0.0, 0.0),
    "nex-agi/nex-n2.5-pro:free": (0.0, 0.0),
    "nex-agi/nex-n2.5-mini:free": (0.0, 0.0),
    "nvidia/nemotron-3-ultra-550b-a55b:free": (0.0, 0.0),
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
}


def model_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = MODEL_PRICING.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
