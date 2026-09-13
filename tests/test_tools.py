"""Tests for the loan calculator tool (offline, deterministic)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools import TOOL_SCHEMA, calculate_monthly_payment, execute_tool  # noqa: E402


def test_known_payment_value():
    # Verified independently: 30M yen, 2.1%/yr, 35y -> 100,925 yen/month.
    assert calculate_monthly_payment(30_000_000, 0.021, 35) == 100_925


def test_zero_interest_is_linear():
    assert calculate_monthly_payment(12_000_000, 0.0, 10) == 100_000


def test_one_year_loan():
    assert calculate_monthly_payment(1_200_000, 0.12, 1) == round(
        1_200_000 * 0.01 / (1 - (1.01) ** (-12))
    )


def test_input_validation():
    for args in [(0, 0.01, 10), (-100, 0.01, 10), (1_000_000, -0.01, 10), (1_000_000, 0.01, 0)]:
        try:
            calculate_monthly_payment(*args)
            raise AssertionError(f"expected ValueError for {args}")
        except ValueError:
            pass


def test_execute_tool_success_payload():
    payload = execute_tool(
        "calculate_monthly_payment",
        json.dumps({"principal": 30_000_000, "annual_rate": 0.021, "years": 35}),
    )
    assert payload == {"monthly_payment_yen": 100_925}


def test_execute_tool_returns_error_object_not_exception():
    payload = execute_tool("calculate_monthly_payment", json.dumps({"principal": "x"}))
    assert "error" in payload
    assert execute_tool("no_such_tool", "{}")["error"].startswith("unknown tool")


def test_tool_schema_shape():
    assert TOOL_SCHEMA["name"] == "calculate_monthly_payment"
    assert set(TOOL_SCHEMA["input_schema"]["required"]) == {"principal", "annual_rate", "years"}
