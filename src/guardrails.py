"""Guardrail: refuse individual advice / tax / regulatory-judgment questions.

Design (see DECISIONS.md):
  Layer 1 — deterministic rule-based classifier on the *question*. If it
  matches a restricted category, the app returns a canned refusal with
  escalation wording and NEVER calls the LLM. Deterministic refusal is what
  makes the "refuse" eval category a strict rule check.
  Layer 2 — system-prompt instructions (see agent.py) telling the model to
  stay on corpus facts and not give advice.
  Layer 3 — output-side advice-phrase check (used by the app to append an
  escalation note and by the eval's refuse checker).

The core risk of Layer 1 is false positives on legitimate factual questions;
patterns are therefore anchored to advice-seeking phrasing (personal
pronouns, 買うべき/おすすめ, prediction verbs) rather than bare topic words.
"""
from __future__ import annotations

import re

CATEGORY_DESCRIPTIONS = {
    "investment_advice": "個別の商品選択・売買など投資助言にあたる質問",
    "future_prediction": "金利・為替などの将来予測を求める質問",
    "tax_judgment": "個別の税務判断・控除の適用可否を求める質問",
    "regulatory_judgment": "規制の解釈・違法性の判断を求める質問",
    "individual_judgment": "質問者個人の返済計画など個別の資金判断を求める質問",
}

_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "investment_advice": [
        re.compile(p)
        for p in [
            r"(おすすめ|お勧め|オススメ)",
            r"(どれ|どちら|何)(を|か).*(買う|購入|選ぶ)",
            r"(買うべき|買ったほうがいい|購入すべき|売るべき|売却すべき|解約すべき)",
            r"(お金|資金|積立).*(預けるべき|入れるべき)",
            r"(始めるべき|やめたほうがいい|乗り換え(べき|たほうがいい))",
            r"アドバイスを?く?ださい",
        ]
    ],
    "future_prediction": [
        re.compile(p)
        for p in [
            # 将来予測のみを拒否し、「条件としての金利はどうなりますか」のような
            # 事実質問（例: 中途解約時の適用金利）を誤爆しないことが要件。
            # そのため「明示的な未来マーカー」か「方向語」を必須にする。
            r"(金利|為替|相場|株価|円(高|安)|レート).{0,16}(予想|予測|見込み)",
            r"(金利|為替|相場|株価|円(高|安)|レート).{0,16}(上が(る|り)|下が(る|り)|上昇|下落)",
            r"(今後|これから|将来).{0,12}(上が(る|り)|下が(る|り)|上昇|下落|どうな(る|り))",
        ]
    ],
    "tax_judgment": [
        re.compile(p)
        for p in [
            r"(いくら.{0,6}戻る|戻ってくる)",
            r"節税.{0,8}(方法|するには|お願い|教えて)",
            r"(私|わたし|僕|俺|うち|当方).{0,8}(控除|税金|確定申告).{0,12}(受けられる|受けられるか|適用|必要か)",
        ]
    ],
    "regulatory_judgment": [
        re.compile(p)
        for p in [
            r"(規制違反|違法|違反).*(ですか|でしょうか|か|どうか|ないでしょうか|なりませんか)",
            r"法的?見解",
            r"(法律|規制)(上|的に)?の?判断を?く?ださい",
        ]
    ],
    "individual_judgment": [
        re.compile(p)
        for p in [
            r"(私|わたし|僕|俺|うち|当方).{0,10}(借りられる|借入可能|返済|計画|いくら.{0,6}借り)",
            r"返済計画を?立て",
        ]
    ],
}

ESCALATION_PHRASES = ["ご相談ください", "専門家", "窓口"]

_ADVICE_PHRASES = [
    re.compile(p)
    for p in [
        r"(おすすめ|お勧め|オススメ)(し|です|は|の)",
        r"(買うべき|購入すべき|売るべき|売却すべき)",
        r"(選ぶべき|選択すべき)",
        r"(したほうがよい|したほうがいい|した方が良い)",
    ]
]

_REFUSAL_TEMPLATE = (
    "申し訳ありませんが、この質問にはお答えできません。"
    "「{category_desc}」にあたるためです。"
    "当アシスタントは商品の事実・条件のご案内のみを行っています。"
    "お客さまご自身の状況に応じた具体的な判断は、みどり銀行の窓口または所定の専門家にご相談ください。"
)


def classify(question: str) -> str | None:
    """Return the first restricted category the question matches, else None."""
    for category, patterns in _PATTERNS.items():
        for pattern in patterns:
            if pattern.search(question):
                return category
    return None


def refusal_message(category: str) -> str:
    return _REFUSAL_TEMPLATE.format(
        category_desc=CATEGORY_DESCRIPTIONS.get(category, category)
    )


def contains_advice_phrases(text: str) -> bool:
    """Layer 3: does an answer body contain advice-like language?"""
    return any(p.search(text) for p in _ADVICE_PHRASES)


def has_escalation(text: str) -> bool:
    return any(phrase in text for phrase in ESCALATION_PHRASES)
