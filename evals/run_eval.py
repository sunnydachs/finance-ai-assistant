"""Run the golden-set evaluation, save a run record, diff vs previous run,
and generate a Markdown report.

Usage:
    python evals/run_eval.py                     # full run (fresh answers)
    python evals/run_eval.py --tag baseline      # label the run
    python evals/run_eval.py --limit 3           # smoke test
    python evals/run_eval.py --reuse-answers evals/runs/<ts>.json   # re-judge only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import judge, rules  # noqa: E402
from src import config  # noqa: E402
from src.agent import AgentResult, get_assistant  # noqa: E402

CALL_PACE_SECONDS = 3.2  # stay under free-tier 20 req/min
ANSWER_RETRIES = 3       # free-tier providers throttle transiently; retry with backoff
ANSWER_BACKOFF_SECONDS = (30.0, 60.0, 90.0)


def _answer_with_retry(assistant, question: str) -> "AgentResult":
    last_error: Exception | None = None
    for attempt, backoff in enumerate((0.0,) + ANSWER_BACKOFF_SECONDS):
        if backoff:
            print(f"    retrying after {backoff:.0f}s ({last_error})", flush=True)
            time.sleep(backoff)
        try:
            return assistant.answer_question(question)
        except Exception as e:
            last_error = e
    raise last_error  # type: ignore[misc]


def load_golden() -> list[dict]:
    items = []
    with config.GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def corpus_fingerprint() -> str:
    h = hashlib.sha256()
    h.update(config.FAQ_PATH.read_bytes())
    for p in sorted(config.NOTES_DIR.glob("*.md")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def config_snapshot() -> dict:
    from src.retrieval import BM25Index  # for the default question weight

    return {
        "app_model": config.APP_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "retrieval_top_k": config.RETRIEVAL_TOP_K,
        "notes_chunk_mode": "sections" if _chunked() else "none",
        "corpus_sha256_12": corpus_fingerprint(),
        "system_prompt_sha256_12": hashlib.sha256(
            __import__("src.agent", fromlist=["SYSTEM_PROMPT"]).SYSTEM_PROMPT.encode()
        ).hexdigest()[:12],
    }


def _chunked() -> bool:
    import os

    return os.environ.get("RAG_CHUNK_NOTES", "none") == "sections"


def run_answers(golden: list[dict], throttle: bool) -> list[dict]:
    assistant = get_assistant()
    records = []
    for i, item in enumerate(golden):
        try:
            result = _answer_with_retry(assistant, item["question"])
        except Exception as e:  # keep the run alive on a persistently failing item
            print(f"  [{i + 1}/{len(golden)}] {item['id']}: ERROR {e}", flush=True)
            records.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "expected_behavior": item["expected_behavior"],
                    "expected_points": item["expected_points"],
                    "source_ref": item["source_ref"],
                    "answer": f"[run error: {type(e).__name__}: {e}]",
                    "citations": [],
                    "refused": False,
                    "guardrail_category": None,
                    "used_tool": False,
                    "retrieved": [],
                    "run_error": True,
                }
            )
            continue
        record = {
            "id": item["id"],
            "question": item["question"],
            "expected_behavior": item["expected_behavior"],
            "expected_points": item["expected_points"],
            "source_ref": item["source_ref"],
            "answer": result.answer,
            "citations": result.citations,
            "refused": result.refused,
            "guardrail_category": result.guardrail_category,
            "used_tool": result.used_tool,
            "retrieved": [{"id": d.id, "score": round(d.score, 2)} for d in result.retrieved],
        }
        records.append(record)
        print(
            f"  [{i + 1}/{len(golden)}] {item['id']}: "
            f"{'REFUSED' if result.refused else ('tool' if result.used_tool else 'answered')}",
            flush=True,
        )
        if throttle:
            time.sleep(CALL_PACE_SECONDS)
    return records


def evaluate(records: list[dict], throttle: bool) -> list[dict]:
    for i, rec in enumerate(records):
        behavior = rec["expected_behavior"]
        if behavior == "refuse":
            passed, reason = rules.check_refuse(rec["answer"])
            rec.update({"rule_pass": passed, "rule_reason": reason, "judge": None})
        elif behavior == "cite":
            retrieved_ids = [d["id"] for d in rec.get("retrieved", [])] or None
            passed, reason, _found = rules.check_cite(rec["answer"], rec["source_ref"], retrieved_ids)
            rec.update({"rule_pass": passed, "rule_reason": reason, "judge": None})
        else:  # answer
            result = judge.judge_answer(
                rec["question"], rec["answer"], rec["expected_points"]
            )
            rec.update({"rule_pass": None, "rule_reason": None, "judge": result})
        if throttle and behavior == "answer":  # only LLM calls need pacing
            time.sleep(CALL_PACE_SECONDS)
        print(f"  judged [{i + 1}/{len(records)}] {rec['id']}", flush=True)
    return records


def score_run(records: list[dict]) -> dict:
    answer_records = [
        r for r in records if r["expected_behavior"] == "answer"
    ]
    # A provider failure (retry-exhausted rate limit etc.) is a run failure,
    # not a model-quality reading. Count them separately and exclude them
    # from the judge-score mean so "provider was down" is never smuggled
    # into "model was worse".
    run_errors = sum(1 for r in answer_records if r.get("run_error"))
    judge_calls_failed = sum(
        1 for r in answer_records if not r.get("run_error") and r.get("judge") is None
    )
    answer_scores = [
        r["judge"]["score"] for r in answer_records
        if not r.get("run_error") and r.get("judge") and r["judge"]["score"]
    ]
    answer_parse_failures = sum(
        1 for r in answer_records
        if not r.get("run_error") and r.get("judge") and not r["judge"]["parse_ok"]
    )
    refuse = [r for r in records if r["expected_behavior"] == "refuse"]
    cite = [r for r in records if r["expected_behavior"] == "cite"]
    return {
        "answer_expected_n": len(answer_records),
        "answer_run_errors": run_errors,
        "judge_call_errors": judge_calls_failed,
        "answer_mean": round(sum(answer_scores) / len(answer_scores), 3) if answer_scores else None,
        "answer_n_scored": len(answer_scores),
        "answer_pct_at_least_4": round(
            100 * sum(1 for s in answer_scores if s >= 4) / len(answer_scores), 1
        ) if answer_scores else None,
        "judge_parse_failures": answer_parse_failures,
        "refuse_pass_rate": round(
            100 * sum(1 for r in refuse if r["rule_pass"]) / len(refuse), 1
        ) if refuse else None,
        "cite_pass_rate": round(
            100 * sum(1 for r in cite if r["rule_pass"]) / len(cite), 1
        ) if cite else None,
    }


def find_previous_run(runs_dir: Path, current_name: str) -> Path | None:
    candidates = sorted(
        p for p in runs_dir.glob("*.json")
        if p.name != current_name and not p.name.endswith(".answers.json")
    )
    return candidates[-1] if candidates else None


def build_diff(records: list[dict], prev_records: dict[str, dict] | None) -> list[dict]:
    if not prev_records:
        return []
    diffs = []
    for rec in records:
        prev = prev_records.get(rec["id"])
        if not prev:
            diffs.append({"id": rec["id"], "change": "new"})
            continue
        if rec["expected_behavior"] == "answer":
            now = rec["judge"]["score"] if rec["judge"] else None
            before = prev["judge"]["score"] if prev["judge"] else None
        else:
            now, before = rec["rule_pass"], prev["rule_pass"]
        if now != before:
            diffs.append({"id": rec["id"], "before": before, "after": now, "change": "changed"})
    return diffs


def write_report(
    run_dir: Path, run_id: str, tag: str, summary: dict, records: list[dict],
    diffs: list[dict], prev_run_id: str | None, snapshot: dict,
) -> Path:
    report_dir = run_dir.parent / "reports"
    # run_dir is the runs/ directory passed by the caller, so the report
    # always lands in the sibling reports/ directory — never derived from
    # cwd, which would land reports in different places per call site.
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"report_{run_id}.md"

    lines: list[str] = []
    lines.append(f"# Eval report — {run_id}" + (f" ({tag})" if tag else ""))
    lines.append("")
    lines.append(f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"- previous run: {prev_run_id or 'none (first run)'}")
    lines.append(f"- app model: `{snapshot['app_model']}`  |  judge: `{snapshot['judge_model']}`")
    lines.append(
        f"- retrieval top-k: {snapshot['retrieval_top_k']}  |  notes chunk mode: `{snapshot['notes_chunk_mode']}`"
    )
    lines.append(f"- corpus fingerprint: `{snapshot['corpus_sha256_12']}`  |  prompt: `{snapshot['system_prompt_sha256_12']}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| answer mean (1–5) | {summary['answer_mean']} (n={summary['answer_n_scored']}) |")
    lines.append(f"| answer % scoring ≥4 | {summary['answer_pct_at_least_4']}% |")
    lines.append(f"| refuse pass rate (rule) | {summary['refuse_pass_rate']}% |")
    lines.append(f"| cite pass rate (rule) | {summary['cite_pass_rate']}% |")
    lines.append(f"| judge parse failures | {summary['judge_parse_failures']} |")
    lines.append("")

    lines.append("## Regression vs previous run")
    lines.append("")
    if not diffs:
        lines.append(
            "_No changes vs previous run._" if prev_run_id
            else "_No comparison (first run)._"
        )
    else:
        lines.append("| item | before | after |")
        lines.append("|---|---|---|")
        for d in diffs:
            if d["change"] == "new":
                lines.append(f"| {d['id']} | — | new |")
            else:
                lines.append(f"| {d['id']} | {d['before']} | {d['after']} |")
    lines.append("")

    lines.append("## Per-item results")
    lines.append("")
    lines.append("| id | behavior | result | note |")
    lines.append("|---|---|---|---|")
    for rec in records:
        if rec["expected_behavior"] == "answer":
            result = rec["judge"]["score"] if rec["judge"] else "?"
            note = (rec["judge"] or {}).get("reason", "")
        else:
            result = "pass" if rec["rule_pass"] else "FAIL"
            note = rec["rule_reason"] or ""
        lines.append(f"| {rec['id']} | {rec['expected_behavior']} | {result} | {note} |")
    lines.append("")

    weak = [
        rec for rec in records
        if rec["expected_behavior"] == "answer" and rec["judge"] and (rec["judge"]["score"] or 0) <= 3
    ]
    if weak:
        lines.append("## Weak answers (score ≤ 3)")
        lines.append("")
        for rec in weak:
            lines.append(f"### {rec['id']} — score {rec['judge']['score']}")
            lines.append(f"- Q: {rec['question']}")
            lines.append(f"- judge: {rec['judge']['reason']}")
            lines.append(f"- retrieved: {[r['id'] for r in rec['retrieved']]}")
            lines.append(f"- answer: {rec['answer'][:400]}")
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden-set evaluation")
    parser.add_argument("--tag", default="", help="label for this run (e.g. baseline, chunked)")
    parser.add_argument("--limit", type=int, default=0, help="only run first N items (smoke test)")
    parser.add_argument("--reuse-answers", type=Path, default=None,
                        help="path to a previous run JSON; reuse its answers, re-judge only")
    parser.add_argument("--no-throttle", action="store_true", help="no pacing delay between API calls")
    args = parser.parse_args(argv)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    golden = load_golden()
    if args.limit:
        golden = golden[: args.limit]

    print(f"run {run_id}: {len(golden)} items, app={config.APP_MODEL}, judge={config.JUDGE_MODEL}")

    if args.reuse_answers:
        prev = json.loads(args.reuse_answers.read_text(encoding="utf-8"))
        # A reused answer set must come from a run with the same config;
        # otherwise the judge scores answers against a corpus/system prompt
        # the model never saw, and the combined run is silently invalid.
        current_snap = config_snapshot()
        prev_snap = prev.get("config", {})
        if prev_snap and prev_snap != current_snap:
            raise SystemExit(
                f"config mismatch: {args.reuse_answers} was generated under "
                f"{prev_snap}, current config is {current_snap}. "
                "Re-running with --reuse-answers across config changes "
                "mixes two incompatible runs."
            )
        prev_by_id = {r["id"]: r for r in prev["records"]}
        records = [dict(prev_by_id[item["id"]]) for item in golden]
        print("reusing answers from", args.reuse_answers)
    else:
        print("generating answers ...")
        records = run_answers(golden, throttle=not args.no_throttle)

    print("evaluating ...")
    snapshot = config_snapshot()
    runs_dir = config.EVAL_RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoint answers before judging so a judge-phase crash does not
    # discard a completed answer-generation pass.
    checkpoint_path = runs_dir / f"{run_id}.answers.json"
    checkpoint_path.write_text(
        json.dumps(
            {"run_id": run_id, "tag": args.tag + "-answers", "config": snapshot,
             "summary": None, "records": records},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    records = evaluate(records, throttle=not args.no_throttle)

    summary = score_run(records)

    run_path = runs_dir / f"{run_id}.json"
    run_path.write_text(
        json.dumps(
            {"run_id": run_id, "tag": args.tag, "config": snapshot, "summary": summary,
             "records": records},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    prev_path = find_previous_run(runs_dir, run_path.name)
    prev_records = None
    prev_run_id = None
    if prev_path:
        prev_run_id = prev_path.stem
        prev_records = {r["id"]: r for r in json.loads(prev_path.read_text(encoding="utf-8"))["records"]}
    diffs = build_diff(records, prev_records)

    report_path = write_report(run_path.parent, run_id, args.tag, summary, records, diffs,
                               prev_run_id, snapshot)

    print(json.dumps(summary, ensure_ascii=False))
    print(f"run saved:   {run_path}")
    print(f"report:      {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
