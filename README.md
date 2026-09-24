# Finance AI Assistant — RAG + Evals for a Fictional Bank

A command-line question-answering assistant over a fictional Japanese bank's
FAQ corpus ("みどり銀行" / Midori Bank), built as a demonstration of how to
ship a small, honest, *measured* LLM application:

- **RAG** over a self-written corpus (40 FAQ items + 2 guideline study notes),
  using a zero-dependency character n-gram BM25 retrieval baseline
- **One tool** (`calculate_monthly_payment`) invoked through native tool use
- **Guardrails** that refuse investment advice, tax judgment, and regulatory
  interpretation questions and escalate to a human, deterministically
- **An evaluation suite** (30 golden questions: 20 answer / 5 refuse / 5 cite)
  scored by rules + an LLM-as-judge, with run-to-run regression reports

> **Disclaimer** — Midori Bank, its products, and all rates in `corpus/` are
> entirely fictional and exist only to make the RAG problem realistic. Nothing
> here is financial advice or a real bank's terms. The guideline "notes" are
> personal study summaries, not official regulatory text.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in your API token
python main.py "住宅ローンの繰上返済の手数料を教えてください"
```

```text
住宅ローンの繰上返済の手数料は、店頭窓口で申し込む場合、8,800円（税込）です。
なお、インターネットバンキングからのお申し込みの場合は手数料は無料となります。

出典: [FAQ-013]
```

Other modes:

```bash
python main.py "質問" --raw            # no corpus — baseline comparison
python main.py "質問" --show-context   # show retrieved docs + BM25 scores
python main.py "質問" --json           # machine-readable output
python main.py "3000万円を年利2.1%で35年間借りた場合の月々の返済額は？"  # triggers the tool
python main.py "損をしない投資信託のおすすめを教えてください"            # refused by guardrail
```

Run the evaluation suite:

```bash
python evals/run_eval.py --tag my-run     # writes evals/runs/<ts>.json + Markdown report
```

---

## Architecture

```text
                    ┌────────────────────────────────────────────────┐
 question ────────► │ Layer 1  guardrails.classify()                 │
                    │ rule-based regex classifier → canned refusal   │
                    │ + escalation, NO LLM call                      │
                    └───────────────┬────────────────────────────────┘
                                    │ (not restricted)
                    ┌───────────────▼────────────────────────────────┐
                    │ Retrieval: char 2/3-gram BM25 (zero deps)      │
                    │ corpus: 40 FAQ items + 2 guideline notes       │
                    └───────────────┬────────────────────────────────┘
                    ┌───────────────▼────────────────────────────────┐
                    │ Layer 2  LLM (tool-use loop, ≤3 rounds)        │
                    │ system prompt: answer only from 資料,           │
                    │ cite [FAQ-xxx]/[NOTE-xxx], never advise        │
                    │ tool: calculate_monthly_payment (元利均等)      │
                    └───────────────┬────────────────────────────────┘
                    ┌───────────────▼────────────────────────────────┐
                    │ Layer 3  output check                          │
                    │ advice-like phrasing → append escalation note  │
                    └────────────────────────────────────────────────┘

Eval:   golden.jsonl (30) ──► answers ──► rules (refuse/cite) + LLM-as-judge
        ──► runs/<ts>.json ──► Markdown report + diff vs previous run
```

## Design decisions (and why)

Full log with alternatives in [DECISIONS.md](DECISIONS.md). Highlights:

**RAG vs fine-tuning vs prompting.** This project needs *lookup of
fast-changing, verifiable facts* (rates, fees, procedures) over a small,
versioned corpus — the textbook case for retrieval, not weight learning:

| factor | prompting alone | fine-tuning | RAG (chosen) |
|---|---|---|---|
| data volume | n/a | needs thousands of QA pairs | works with ~50 docs |
| update frequency | re-prompt, still hallucinates | retrain per change | edit a JSONL line |
| verifiability | none | none | citation per answer |
| latency/cost | lowest | high upfront | small retrieval overhead |
| governance (financial firm) | weak audit trail | opaque | corpus is reviewable |

Fine-tuning was rejected in one sentence: we cannot justify training on a
50-document corpus, and fine-tuned facts still go stale and cannot show
sources. Prompting alone is kept as a *measured baseline* (`--raw`), not the
product.

**Search: character n-gram BM25, zero dependencies.** Japanese has no spaces,
so word retrieval requires a morphological analyzer (MeCab/janome). Instead we
index lowercased character 2/3-grams plus whole ASCII tokens (ATM, NISA), with
politeness boilerplate (教えてください, ですか, …) stripped from both queries and
documents — boilerplate carries no topical signal and demonstrably biased
ranking (see the ans-10 story below). It is boring, dependency-free, and —
measured by the cite category of the eval — good enough on a ~45-document
corpus. Embeddings get added only if the eval shows retrieval failures.
(`corpus.py` supports section-chunked notes; the change was measured and
*not* adopted at this scale — see the eval reports.)

**Deterministic refusal for restricted topics.** A refusal that needs an LLM
to verify it is not a guarantee, so Layer 1 is a regex classifier: if the
question matches a restricted category (investment advice, future
predictions, tax judgment, regulatory interpretation, personal repayment
planning), the app returns a canned refusal with escalation wording and never
calls the model. The known cost is false positives; patterns are therefore
anchored to advice-seeking phrasing and pinned by unit tests
(`tests/test_guardrails.py`), including one case a test suite caught and fixed
during development.

**A different model as judge.** The answering model and the judging model are
different (and the judge is the larger of the two) to avoid self-preference
bias — a structural choice, not a tuning knob.

## How we know it works

The eval suite is the core of this project. Every run executes
`evals/golden.jsonl` — 30 questions in three categories, each checked by a
method matched to what it can actually guarantee:

| category | n | check | why this check |
|---|---|---|---|
| `answer` | 20 | LLM-as-judge, 1–5 rubric anchored on `expected_points` | free-form Japanese text has no exact-match ground truth; the rubric ties the score to checkable facts |
| `refuse` | 5 | pure string rules: no advice phrasing **and** escalation wording present | a refusal must be verifiable without trusting another model's opinion |
| `cite` | 5 | expected source ID must appear in the answer | citation is exactly the property retrieval should guarantee; string-matchable |

Every run is saved (`evals/runs/<timestamp>.json`) with a config fingerprint
(model IDs, retrieval top-k, chunk mode, corpus hash, prompt hash), scored,
and diffed against the previous run in a Markdown report
(`evals/reports/`). The iteration protocol is: change one thing → run →
compare → analyze → record in DECISIONS.md.

**What the loop caught in practice** — three findings from the first three
runs, each fixed or decided *through measurement*:

1. **A guardrail gap (ref-03).** The first full run flagged a refusal
   failure: the personal-repayment-planning question was phrased "返済計画**も**
   立ててください", which the regex (written for "返済計画**を**立てて") did not
   match. The unit tests then caught the same wording, the pattern was
   narrowed to a stem form (`返済計画.{0,2}立て`), and the item passed on
   re-run.
2. **A retrieval miss (ans-10).** The baseline scored ans-10 ("what
   documents do I need for a mortgage review?") 1/5: the right FAQ (020)
   ranked 10th and the app answered "not in the corpus". Per-term analysis
   showed *politeness boilerplate ngrams* (教えてください, ですか, …) inflating
   documents whose embedded FAQ question ends with 教えてください, drowning the
   discriminative terms (審査, 書類). Stripping politeness boilerplate and
   punctuation from both queries and documents lifted retrieval hits on the
   golden set from 24/25 to 25/25, and ans-10 from 1/5 to 5/5 in the
   end-to-end run. A BM25F-style question-field weight was *also* tried for
   this failure and dropped: it added nothing once stripping was in place.
3. **A measured non-improvement (notes chunking).** Splitting the guideline
   notes into section chunks (a standard RAG instinct) changed nothing at
   ~45-document scale: identical scores on all three categories. Chunking
   stays *off* by default, recorded as "revisit when the corpus grows" —
   complexity only earns its keep when the eval moves.

This is the project's working definition of "it works": **stated behavior,
checked automatically, with the history of what changed kept in reports and
DECISIONS.md.**

On human review: production eval practice layers automated metrics,
LLM-as-judge, and stratified human review. This project implements the first
two rigorously and *deliberately* leaves the third as documented future work
(see Limitations) rather than pretending a handful of ad-hoc spot checks
constitute a review program.

### Results

Golden set = 30 questions (20 answer / 5 refuse / 5 cite). "retrieval hits"
is an offline check that each non-refuse item's expected source ranks in the
top-4 (25 items applicable).

| run (all on 2026-09-13) | change under test | answer mean (1–5) | refuse pass | cite pass | retrieval hits |
|---|---|---|---|---|---|
| `baseline` | — | 4.80 | 100% (5/5) | 100% (5/5) | 24/25 (`ans-10` miss) |
| `chunked-notes` | notes split into section chunks | 4.80 | 100% (5/5) | 100% (5/5) | 24/25 (unchanged) |
| `boilerplate-strip` | politeness boilerplate stripped from query+docs | **5.00** | **100% (5/5)** | **100% (5/5)** | **25/25** |

- Reports: `evals/reports/report_20260913T170214Z.md` (baseline),
  `report_20260913T172230Z.md` (chunking), `report_20260913T175536Z.md`
  (final). Raw records with per-item answers, citations, retrieval scores,
  judge reasons and config fingerprints: `evals/runs/`.
- Per-item judge scores, 20/20 = 5.0 in the final run; zero judge parse
  failures.
- Retrieval quality, measured offline over the 25 golden items that carry a
  ground-truth source (`evals/retrieval_eval.py`, report:
  `evals/reports/retrieval_metrics.md`): **MRR 0.90, recall@5 100%,
  precision@1 84%** — and precision decays to 10% at k=10, the
  recall-vs-distractor tradeoff that keeps top-k at 4. Note refs (e.g.
  `NOTE-002-A`) are credited when their parent note ranks, which matches
  whole-file note indexing.
- API spend across all runs and development: **$0.00** (free-tier gateway
  models; token usage logged per call in `logs/usage.jsonl`).

Reproduce:

```bash
python evals/run_eval.py --tag my-run   # diff appears against the latest saved run
```

## Cost

Models are served through the OpenRouter gateway (Anthropic
Messages-API-compatible), using free-tier models selected by an empirical
comparison battery ([scripts/model_comparison.py](scripts/model_comparison.py)):
`inclusionai/ling-3.0-flash-fin` for answers (finance-specialized),
`nvidia/nemotron-3-ultra-550b` as judge (larger, different family, stable
JSON). API spend for the entire project so far: **$0.00**. Token usage is
logged per call to `logs/usage.jsonl`; swapping to paid Claude models is a
`.env`-only change (`ANTHROPIC_BASE_URL`, token, model IDs) because the
`anthropic` SDK is used throughout.

## Limitations

- **The corpus is tiny and self-written.** Retrieval quality on ~45 documents
  says little about behavior at thousands of documents.
- **No stratified human review yet.** The judge itself is unvalidated — a
  proper program would measure judge–human agreement on a sample before
  trusting judge scores (planned; see below).
- **Single-turn only.** No conversation memory; the CLI answers one question
  per invocation.
- **Guardrail Layer 1 is regex.** It catches phrasing variants, not meaning;
  paraphrases outside the pattern set may reach the LLM (Layer 2 mitigates,
  but does not guarantee).
- **Free-tier models** vary in availability and instruction-following; the
  judge retries once on unparseable output and parse failures are counted in
  reports rather than hidden.
- **Run-to-run variance — read the regression tables with this in mind.**
  One intermediate run (`report_20260913T173630Z`) was partially degraded by
  upstream 429 rate-limiting: four items returned provider-error strings, so
  the final report's diff table shows "1 → 5" and "False → True" recoveries
  that reflect retry resilience, not a model-quality jump. The honest
  before/after comparison is `baseline` (4.80, ans-10 = 1/5) vs the final
  clean run (5.00, ans-10 = 5/5) — the retrieval fix, not the rate-limit
  recovery, is the improvement claim. All runs including the degraded one
  are kept in `evals/runs/` (nothing was cherry-picked), and averaging over
  repeated runs per configuration would be the more rigorous protocol —
  listed in Future improvements.
- **Fictional data**: no claim is made about real-world product accuracy —
  the *pipeline* is the deliverable.

## Future improvements

1. Embedding-based retrieval (e.g. multilingual MiniLM) compared against the
   BM25 baseline *through the eval*, not swapped on faith.
2. Judge calibration: 50-item human-labeled sample → judge–human agreement
   (Cohen's κ) before treating judge scores as ground truth.
3. Conversation memory + multi-turn tool flows.
4. Batch API for eval runs to cut wall-clock time, plus averaging over
   repeated runs per configuration to quantify run-to-run variance (single
   runs on stochastic free-tier models are noisy — see Limitations).
5. CI job running the eval on a schedule to catch model drift.

## CI & Security

Inherited from the [sunnydachs/repo-template](https://github.com/sunnydachs/repo-template)
stack (all free tiers), adapted for Python:

- **CI** (`ci.yml`): pytest on Python 3.11 / 3.12 / 3.13. The test suite is
  fully offline — no API keys are needed to build or test the repo.
- **CodeQL** (`codeql.yml`): `security-extended` query pack on Python.
- **gitleaks** (`gitleaks.yml`): secret scanning on every push/PR (plus
  GitHub-native secret scanning & push protection on the public repo).
- **Dependabot** (`dependabot.yml`): weekly pip + github-actions updates,
  grouped minor/patch, with auto-merge for patch/security updates only.
- **Stale workflow** (`stale.yml`): weekly cleanup of stale issues/PRs.

Agent-facing hard rules (zero secrets in git, no absolute paths, offline
tests) live in [AGENTS.md](AGENTS.md).

## Project layout

```text
src/        app code (agent, retrieval, guardrails, tool, llm wrapper, config)
corpus/     faq.jsonl (40 items) + notes/ (own-words guideline summaries)
evals/      golden.jsonl (30), judge, rule checks, run_eval.py, retrieval metrics, runs/, reports/
tests/      offline unit tests (26) — retrieval, guardrails, tool, eval rules
docs/       raw-vs-RAG comparison notes, demo script + captures, demo videos
.github/    CI (pytest matrix), CodeQL, gitleaks, dependabot, stale bot
DECISIONS.md  design decisions: options → choice → reasons → eval evidence
AGENTS.md   hard rules for AI agents working in this repo
```

## Demo video

- **docs/demo_video_ja.mp4** — 3m12s Japanese demo (VOICEVOX narration), produced
  in the original review session.
- **docs/demo_video_ja_short.mp4** — 1m25s Japanese demo (VOICEVOX narration,
  1280×720), rebuilt with clean frame layouts.

The English 5-minute walkthrough script is in
[docs/demo_script.md](docs/demo_script.md).
