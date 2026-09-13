"""CLI entry point.

Usage:
    python main.py "質問"                 # RAG answer with citations
    python main.py "質問" --raw           # no corpus (baseline comparison)
    python main.py "質問" --show-context  # also print retrieved docs + scores
    python main.py "質問" --no-tool       # disable the calculator tool
    python main.py "質問" --json          # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from src import config
from src.agent import get_assistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="みどり銀行 FAQ アシスタント（架空。RAG + tool use + ガードレール付き）"
    )
    parser.add_argument("question", help="質問（日本語）")
    parser.add_argument("--raw", action="store_true", help="コーパス検索なし（素のモデル）")
    parser.add_argument("--no-tool", action="store_true", help="計算ツールを無効化")
    parser.add_argument("--show-context", action="store_true", help="検索した資料とスコアを表示")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    assistant = get_assistant()
    result = assistant.answer_question(
        args.question, use_rag=not args.raw, use_tool=not args.no_tool
    )

    if args.json:
        payload = {
            "question": result.question,
            "answer": result.answer,
            "citations": result.citations,
            "refused": result.refused,
            "guardrail_category": result.guardrail_category,
            "used_tool": result.used_tool,
            "retrieved": [
                {"id": d.id, "score": round(d.score, 3)} for d in result.retrieved
            ],
            "raw_mode": result.raw_mode,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.show_context and result.retrieved:
        print("=== 検索した資料 (id: score) ===")
        for doc in result.retrieved:
            print(f"  [{doc.id}] {doc.title} : {doc.score:.3f}")
        print()

    print(result.answer)
    if result.citations and not result.refused:
        print("\n出典: " + ", ".join(f"[{c}]" for c in result.citations))
    if result.used_tool:
        print("(ツール使用: 月返済額計算)")
    if result.refused:
        print(f"\n※ ガードレール作動 ({result.guardrail_category}): 上記は定型の拒否応答です（LLMを呼び出していません）")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
