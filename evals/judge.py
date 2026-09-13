"""LLM-as-judge for the "answer" category.

Judge design (see DECISIONS.md):
  - A different, larger model than the answering model to avoid
    self-preference bias.
  - Rubric anchored on expected_points (the facts a good answer must contain).
  - Strict JSON output. Parsing is defensive (fence stripping, brace slicing,
    one repair retry); unparseable output is recorded as a scoring failure
    instead of silently counted as a pass.
"""
from __future__ import annotations

import json
import logging
import re

from src import config, llm

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """あなたは金融商品FAQアシスタントの回答を評価するジャッジです。以下のルーブリックに従い、回答を1〜5点で採点してください。

ルーブリック:
- 5: 期待される要素(expected_points)をすべて正確に含み、事実誤認がない。
- 4: 期待される要素のほとんどを含むが、軽微な漏れや表現の甘さがある。
- 3: 期待される要素の一部のみを含む、または説明が曖昧。
- 2: 期待される要素がほとんど含まれない、または重要な誤りがある。
- 1: 事実と異なる、資料にない内容を捏造している、または質問に答えていない。

出力は必ず次のJSONのみ（前後に説明・コードフェンスを付けない）:
{"score": <1から5の整数>, "reason": "<50文字以内の日本語の理由>"}"""

_RETRY_SYSTEM = JUDGE_SYSTEM + "\n\n前回の出力はJSONとして解析できませんでした。JSONのみを出力してください。"


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    # Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "score" not in data:
        return None
    try:
        score = int(data["score"])
    except (TypeError, ValueError):
        return None
    if not 1 <= score <= 5:
        return None
    return {"score": score, "reason": str(data.get("reason", ""))[:200]}


def judge_answer(question: str, answer: str, expected_points: list[str]) -> dict:
    """Score one answer. Returns {score, reason, parse_ok}.

    score is None when the judge output could not be parsed after a retry.
    """
    user_msg = (
        f"質問: {question}\n\n"
        f"回答: {answer}\n\n"
        f"expected_points（良い回答が含むべき要素）: {json.dumps(expected_points, ensure_ascii=False)}"
    )
    for attempt, system in enumerate([JUDGE_SYSTEM, _RETRY_SYSTEM]):
        try:
            message = llm.call_model(
                model=config.JUDGE_MODEL,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
                purpose="judge",
                max_tokens=config.JUDGE_MAX_TOKENS,
                temperature=config.JUDGE_TEMPERATURE,
            )
        except Exception as e:  # gateway/model flakiness must not kill the run
            logger.warning("judge call failed (attempt %d): %s", attempt + 1, e)
            continue
        parsed = _extract_json(llm.text_of(message))
        if parsed:
            return {"score": parsed["score"], "reason": parsed["reason"], "parse_ok": True}
        logger.warning("judge output unparseable (attempt %d)", attempt + 1)
    return {"score": None, "reason": "judge output unparseable", "parse_ok": False}
