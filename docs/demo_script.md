# Demo Video Script — 5-minute English walkthrough

> Target length: ~5 minutes. Format: screen recording with voice-over
> (reading this script is fine). Record at 1440p for readable terminal text,
> font size 16+.

## Pre-recording checklist

- [ ] Terminal prepared: `cd finance-ai-assistant && source .venv/bin/activate`
- [ ] Two terminals: one for app demos, one for the eval run (already completed
      reports open in an editor)
- [ ] Latest eval report open: `evals/reports/report_<latest>.md`
- [ ] Microphone level checked; read slowly, sentence by sentence

---

## [0:00] Hook + what this is (30s)

> "This is a question-answering assistant for a fictional Japanese bank,
> built as a small, honest demonstration of production-minded LLM
> engineering: retrieval over a curated corpus, one tool call,
> compliance-minded guardrails — and, most importantly, an evaluation suite
> that can prove behavior and catch regressions. In the next five minutes
> I'll show what it does, how it's built, and how I know it works."

## [0:30] Demo 1 — the happy path (60s)

Type:

```bash
python main.py "住宅ローンの繰上返済の手数料を教えてください。店頭で申し込む場合も。"
```

> "A natural question about prepayment fees. The answer is grounded in the
> bank's FAQ — notice the citation, FAQ-013, at the end. Every answer names
> the documents it came from, so any claim can be checked."

Run with `--show-context` on the same question.

> "Here's the retrieval step: a zero-dependency character n-gram BM25 index
> over forty FAQ items and two guideline notes. The right document ranks
> first with a strong margin. I chose keyword retrieval deliberately as a
> measured baseline — embeddings come later, only if the eval shows
> retrieval failures."

## [1:30] Demo 2 — tool use (45s)

```bash
python main.py "3000万円を年利2.1%で35年間借りた場合、月々の返済額はいくらになりますか？"
```

> "For calculation questions, the model can invoke a real tool — an
> equal-payment loan calculator — instead of doing arithmetic in its head.
> The result, 100,925 yen, is computed by a pure Python function with unit
> tests, and the tool call is logged."

## [2:15] Demo 3 — guardrails (45s)

```bash
python main.py "損をしない投資信託のおすすめを教えてください"
```

> "'Recommend me a fund that won't lose money.' The assistant refuses and
> escalates to a human — and it does this without calling the model at all.
> A refusal that needs an LLM to verify is not a guarantee, so the first
> layer is a deterministic classifier, pinned by unit tests, with the
> system prompt and an output check as second and third layers."

## [3:00] Demo 4 — the eval suite (90s)

Open `evals/golden.jsonl` (scroll a few lines), then the latest report.

> "This is the core of the project. Thirty golden questions in three
> categories. Twenty 'answer' questions scored one-to-five by an LLM judge
> against explicit expected points — using a different, larger model than
> the answering model to avoid self-preference bias. Five 'refuse' questions
> checked by pure string rules: no advice phrasing, escalation present. And
> five 'cite' questions where the expected source ID must appear in the
> answer.
>
> Every run is saved with a config fingerprint — models, top-k, corpus hash,
> prompt hash — and each report diffs against the previous run. Here's a
> real example from development: the first full run failed a refusal
> question because the regex pattern didn't match one phrasing of 'please
> draw up a repayment plan'. I fixed the pattern, re-ran, and the diff table
> shows the item going from fail to pass. That loop — change one thing,
> measure, record — is what I mean by knowing the system works."

## [4:30] Design decisions + limitations (30s)

Point at DECISIONS.md and README.

> "Every design choice — keyword search first, a finance-specialized free
> model served through the Anthropic-compatible gateway, why RAG instead of
> fine-tuning — is written down with alternatives and reasons in
> DECISIONS.md. And the README states the limits plainly: no human review
> program yet, single-turn only, and a tiny self-written corpus. The
> pipeline is the deliverable, not the fictional bank."

## [5:00] Close (10s)

> "Thanks for watching — the repository README has the full eval results
> and the 'How we know it works' section."

---

## Recording tips

- Pauses are fine; cut them in editing. Reading errors: just re-read the
  sentence and cut.
- Show, don't claim: whenever the script says "notice X", keep the X on
  screen for at least 3 seconds.
- If the eval report feels too dense on camera, zoom into the Summary and
  Regression tables only.
