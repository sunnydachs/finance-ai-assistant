"""One-off model comparison battery for model selection (recorded in DECISIONS.md).

Tests per model:
  A. Japanese RAG-style QA with citation instructions
  B. Tool use (loan payment calculator) via Anthropic Messages schema
  C. Strict JSON output (LLM-as-judge style)
"""
import json
import os
import sys

from anthropic import Anthropic

MODELS = [
    "inclusionai/ling-3.0-flash-fin:free",
    "inclusionai/ling-3.0-flash-vl:free",
    "nex-agi/nex-n2.5-pro:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]

client = Anthropic(
    base_url=os.environ["ANTHROPIC_BASE_URL"],
    auth_token=os.environ["ANTHROPIC_AUTH_TOKEN"],
    max_retries=1,
    timeout=150.0,
)

DOCS = [
    {"id": "FAQ-012", "text": "みどり銀行の「ほっと住宅ローン」は変動金利型の場合、借入時の基準金利は年2.1%（2026年9月時点）。10年固定は年2.4%。事務手数料は借入金額の2.2%（税込）。"},
    {"id": "FAQ-013", "text": "「ほっと住宅ローン」の繰上返済は1回100万円以上から。手数料はネットバンキングからの申し込みなら無料、店頭では8,800円（税込）かかる。"},
]
QUESTION = "繰上返済の手数料はいくらですか？店頭でやった場合も教えてください。"

def test_rag(model):
    sys_prompt = "あなたは銀行のFAQアシスタントです。以下の資料のみに基づいて日本語で回答し、文末に出典IDを [FAQ-xxx] 形式で付けてください。資料にないことは「わかりません」と答えてください。"
    docs_text = "\n\n".join(f"[{d['id']}] {d['text']}" for d in DOCS)
    msg = client.messages.create(
        model=model, max_tokens=600, system=sys_prompt,
        messages=[{"role": "user", "content": f"資料:\n{docs_text}\n\n質問: {QUESTION}"}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text[:500]

LOAN_TOOL = {
    "name": "calculate_monthly_payment",
    "description": "等額返済(元利均等)の住宅ローン月返済額を計算する",
    "input_schema": {
        "type": "object",
        "properties": {
            "principal": {"type": "number", "description": "借入元金(円)"},
            "annual_rate": {"type": "number", "description": "年利率(例: 0.021)"},
            "years": {"type": "integer", "description": "返済年数"},
        },
        "required": ["principal", "annual_rate", "years"],
    },
}

def test_tool(model):
    msg = client.messages.create(
        model=model, max_tokens=800,
        system="利用者の質問に答えるために、必要なら提供されたツールを使ってください。",
        messages=[{"role": "user", "content": "3000万円を年利2.1%で35年借りた場合の月々の返済額を計算して。"}],
        tools=[LOAN_TOOL],
    )
    tool_blocks = [b for b in msg.content if b.type == "tool_use"]
    if not tool_blocks:
        return f"NO_TOOL_USE: {json.dumps([b.type for b in msg.content])}, text={(''.join(b.text for b in msg.content if b.type=='text')[:200])}"
    args = tool_blocks[0].input
    return f"TOOL_OK name={tool_blocks[0].name} args={json.dumps(args, ensure_ascii=False)}"

def test_json(model):
    sys_prompt = "あなたは評価(judge)です。回答を1-5点で採点し、必ず次のJSONのみを出力してください: {\"score\": <1-5>, \"reason\": \"<短い理由>\"} 前後に他のテキストを付けないでください。"
    msg = client.messages.create(
        model=model, max_tokens=400, system=sys_prompt,
        messages=[{"role": "user", "content": "質問: 繰上返済の手数料は？\n回答: ネットバンキングからの申し込みなら無料、店頭では8,880円かかります。[FAQ-013]"}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    try:
        json.loads(text)
        verdict = "JSON_PARSED"
    except Exception:
        verdict = f"JSON_FAIL (raw head: {text[:120]!r})"
    return verdict

for model in MODELS:
    print(f"===== {model} =====")
    for name, fn in [("A-rag", test_rag), ("B-tool", test_tool), ("C-json", test_json)]:
        try:
            print(f"  [{name}] {fn(model)}")
        except Exception as e:
            print(f"  [{name}] ERROR: {type(e).__name__}: {str(e)[:200]}")
    print()
    sys.stdout.flush()
