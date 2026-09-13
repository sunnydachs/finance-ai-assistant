"""Loan payment calculator tool (元利均等 / equal total payment).

Pure functions + the Anthropic tool schema. Deterministic and unit-tested;
the agent calls it when the model emits a tool_use block.
"""
from __future__ import annotations

import json


def calculate_monthly_payment(principal: float, annual_rate: float, years: int) -> int:
    """Monthly payment for equal-principal-and-interest (元利均等) repayment.

    annual_rate is a decimal (0.021 = 2.1%). Returns yen per month, rounded
    to the nearest yen.
    """
    if principal <= 0:
        raise ValueError("principal must be positive")
    if annual_rate < 0:
        raise ValueError("annual_rate must be >= 0")
    if years <= 0:
        raise ValueError("years must be >= 1")
    n = years * 12
    r = annual_rate / 12
    if r == 0:
        payment = principal / n
    else:
        payment = principal * r / (1 - (1 + r) ** (-n))
    return round(payment)


TOOL_SCHEMA = {
    "name": "calculate_monthly_payment",
    "description": (
        "等額返済（元利均等）の住宅ローン・カードローンの月返済額を計算する。"
        "principal は借入元金（円）、annual_rate は年利率（0.021 のように小数で）、"
        "years は返済年数（整数）。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "principal": {"type": "number", "description": "借入元金（円）"},
            "annual_rate": {"type": "number", "description": "年利率（例: 0.021）"},
            "years": {"type": "integer", "description": "返済年数"},
        },
        "required": ["principal", "annual_rate", "years"],
    },
}


def execute_tool(name: str, raw_input: str) -> dict:
    """Run a tool call by name; returns the tool_result content payload.

    Errors are returned to the model as a JSON error object so the agent loop
    can recover instead of crashing the request.
    """
    if name != TOOL_SCHEMA["name"]:
        return {"error": f"unknown tool: {name}"}
    try:
        args = json.loads(raw_input)
        payment = calculate_monthly_payment(
            principal=float(args["principal"]),
            annual_rate=float(args["annual_rate"]),
            years=int(args["years"]),
        )
        return {"monthly_payment_yen": payment}
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
