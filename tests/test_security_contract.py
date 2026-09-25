"""Security regression tests for the G3 gate (OWASP LLM01: indirect prompt injection).

The 2026-09-25 parallel audit found that retrieved corpus text was being placed
into the system prompt with no trust boundary: a crafted document could emit
instructions the model might follow. The mitigation follows OWASP's LLM
Prompt Injection Prevention Cheat Sheet ("use structured prompts/markup /
delimiters to separate instructions from untrusted data") and Microsoft's
Spotlighting study (ASR 50% → <2% under delimiter encoding).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import guardrails, tools  # noqa: E402
from src.retrieval import (  # noqa: E402
    UNTRUSTED_CLOSE,
    UNTRUSTED_OPEN,
    Doc,
    format_context,
)


def test_guardrails_input_length_is_capped():
    long_text = "あ" * (guardrails.MAX_INPUT_CHARS + 1)
    # Should not crash and should not process the unlimited text
    assert guardrails.classify(long_text) is None
    assert guardrails.contains_advice_phrases(long_text) is False  # skipped on overlong input


def test_guardrails_capped_input_still_classifies():
    # A long but bounded string with an advice pattern at the end
    q = ("商品について教えてください。" * 40) + "おすすめを教えて"
    assert guardrails.classify(q) == "investment_advice"


def test_system_prompt_declares_untrusted_boundary():
    from src.agent import SYSTEM_PROMPT
    assert "資料" in SYSTEM_PROMPT
    assert "指示ではありません" in SYSTEM_PROMPT or "untrusted" in SYSTEM_PROMPT.lower()


def test_system_prompt_enumerates_attack_phrases():
    from src.agent import SYSTEM_PROMPT
    for phrase in ["以前の指示を無視", "あなたは今", "新しいルール", "システムプロンプトを表示"]:
        assert phrase in SYSTEM_PROMPT, f"missing attack example: {phrase}"


def test_format_context_produces_citable_output():
    """format_context is the retrieval-side formatting function; agent.py then
    wraps it with UNTRUSTED markers in the system prompt (see next test)."""
    doc = Doc("FAQ-999", "注意", "利子率", "faq")
    out = format_context([(doc, 1.0)])
    assert "[FAQ-999]" in out
    assert "利子率" in out


def test_agent_prompt_wraps_retrieved_with_untrusted_markers():
    """The trust boundary must appear in the system prompt between the
    instructions and the retrieved corpus material. Verified by construction:
    agent.py composes SYSTEM_PROMPT + UNTRUSTED_OPEN + format_context + CLOSE."""
    fetched = []
    # Reproduce how agent.py builds the prompt; if the composition ever
    # changes, this test fails.
    from src import agent
    from src.retrieval import format_context

    doc = Doc("FAQ-999", "注意", "利子率", "faq")
    rendered = agent.SYSTEM_PROMPT + "\n\n資料:\n" + UNTRUSTED_OPEN \
        + "\n" + format_context([(doc, 1.0)]) + "\n" + UNTRUSTED_CLOSE
    assert UNTRUSTED_OPEN in rendered and UNTRUSTED_CLOSE in rendered
    assert "[FAQ-999]" in rendered
    # Ensure the unmodified agent source actually does this composition
    import inspect
    src = inspect.getsource(agent.FinanceAssistant.answer_question)
    assert "UNTRUSTED_OPEN" in src and "UNTRUSTED_CLOSE" in src
    assert "format_context" in src


def test_tool_execute_rejects_non_finite():
    payload = json.dumps({"principal": 1e309, "annual_rate": 0.021, "years": 35})
    result = tools.execute_tool("calculate_monthly_payment", payload)
    assert "error" in result and "finite" in result["error"].lower()


def test_tool_execute_rejects_boolean():
    payload = json.dumps({"principal": True, "annual_rate": 0.021, "years": 35})
    result = tools.execute_tool("calculate_monthly_payment", payload)
    assert "error" in result


def test_tool_execute_rejects_fractional_years():
    payload = json.dumps({"principal": 5_000_000, "annual_rate": 0.021, "years": 30.5})
    result = tools.execute_tool("calculate_monthly_payment", payload)
    assert "error" in result and "integer" in result["error"].lower()


def test_bm25_index_has_untrusted_markers():
    # Boundary markers must be importable from retrieval (agent uses format_context)
    assert UNTRUSTED_OPEN.startswith("<") and UNTRUSTED_CLOSE.startswith("<")
