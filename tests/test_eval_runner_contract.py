"""Regression tests for the eval runner's contract bugs and metric honesty.

These pin the G4-gate bugs found by the 2026-09-25 parallel audit:
  - write_report must not crash with NameError on first/no-change runs
  - run_error records must not be smuggled into the judge-score mean
  - check_cite must reject citations that were not retrieved
  - write_report's run_dir parameter must actually be used (dead param)
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.run_eval import score_run, write_report, find_previous_run, build_diff  # noqa: E402
from evals import rules  # noqa: E402


_SNAPSHOT = {
    "app_model": "m", "judge_model": "j", "retrieval_top_k": 4,
    "notes_chunk_mode": "none", "corpus_sha256_12": "abc",
    "system_prompt_sha256_12": "def",
}
_SUMMARY = {
    "answer_mean": 5.0, "answer_expected_n": 1, "answer_n_scored": 1,
    "answer_pct_at_least_4": 100.0, "answer_run_errors": 0,
    "judge_call_errors": 0, "judge_parse_failures": 0,
    "refuse_pass_rate": 100.0, "cite_pass_rate": 100.0,
}
_RECORD = {
    "id": "ans-01", "question": "q", "expected_behavior": "answer",
    "expected_points": [], "source_ref": "FAQ-001",
    "answer": "a[FAQ-001]", "citations": ["FAQ-001"], "refused": False,
    "guardrail_category": None, "used_tool": False,
    "retrieved": [{"id": "FAQ-001", "score": 1.0}],
    "judge": {"score": 5, "reason": "ok", "parse_ok": True},
    "rule_pass": None, "rule_reason": None,
}


def test_write_report_first_run_does_not_crash():
    with tempfile.TemporaryDirectory() as d:
        p = write_report(Path(d), "rid1", "tag", _SUMMARY, [_RECORD], [], None, _SNAPSHOT)
        assert p.exists()
        assert "No comparison (first run)" in p.read_text(encoding="utf-8")


def test_write_report_no_change_vs_prev_does_not_crash():
    with tempfile.TemporaryDirectory() as d:
        p = write_report(Path(d), "rid2", "tag", _SUMMARY, [_RECORD], [], "previd", _SNAPSHOT)
        assert p.exists()
        assert "No changes vs previous run" in p.read_text(encoding="utf-8")


def test_score_run_excludes_run_errors_from_mean():
    ok = {**_RECORD}
    err = {**_RECORD, "id": "ans-02", "run_error": True,
           "judge": {"score": 1, "reason": "err", "parse_ok": True}}
    summary = score_run([ok, err])
    assert summary["answer_expected_n"] == 2
    assert summary["answer_n_scored"] == 1  # only the non-error record
    assert summary["answer_mean"] == 5.0     # the 1-point error record must not pollute


def test_score_run_counts_run_errors_separately():
    err = {**_RECORD, "run_error": True,
           "judge": {"score": 1, "reason": "err", "parse_ok": True}}
    summary = score_run([err])
    assert summary["answer_run_errors"] == 1


def test_check_cite_accepts_expected_retrieved():
    passed, reason, found = rules.check_cite(
        "…。[FAQ-001]", "FAQ-001", retrieved_ids=["FAQ-001", "FAQ-002"]
    )
    assert passed, reason


def test_check_cite_rejects_unretrieved_citation():
    """An answer citing both the retrieved (expected) and an unretrieved
    source must fail: presence of the expected citation is not grounding."""
    passed, reason, found = rules.check_cite(
        "…。[FAQ-001] と [FAQ-099]", "FAQ-001", retrieved_ids=["FAQ-001"]
    )
    assert not passed
    assert "FAQ-099" in reason


def test_check_cite_empty_retrieved_rejects_all_citations():
    """With nothing retrieved, any citation is ungrounded by definition."""
    passed, reason, found = rules.check_cite(
        "…。[FAQ-001]", "FAQ-001", retrieved_ids=[]
    )
    assert not passed


def test_check_cite_parent_note_counted_as_grounded():
    passed, reason, found = rules.check_cite(
        "…。[NOTE-002-A]", "NOTE-002-A", retrieved_ids=["NOTE-002"]
    )
    assert passed, reason


def test_write_report_uses_run_dir_argument():
    """run_dir is the runs/ dir; the report must land in the sibling reports/."""
    with tempfile.TemporaryDirectory() as root:
        runs_dir = Path(root) / "runs"
        runs_dir.mkdir()
        p = write_report(runs_dir, "rid1", "tag", _SUMMARY, [_RECORD], [], None, _SNAPSHOT)
        assert p.parent == runs_dir.parent / "reports"
        assert p.exists()
