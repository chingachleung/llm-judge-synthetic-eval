# llm-judge-synthetic-eval

A small, runnable pipeline for evaluating an LLM-backed (or any text-generating)
system: expand a handful of hand-written seed questions into a larger synthetic
test set, score a candidate system's answers against reference answers with a
pluggable judge, and roll the results up into a per-intent gap report.

This is an original, from-scratch demo — not production code — built to show
the same evaluation pattern (small seed set -> synthetic expansion -> tiered
judging -> per-intent gap report, rather than one blended score) used in
LLM-as-judge eval frameworks I've built professionally.

## Why this design

- **Synthetic expansion from a small seed set.** Hand-writing hundreds of eval
  cases doesn't scale. `generate_synthetic_cases` takes a handful of seed
  questions (each tagged with an intent, optionally with `{slot}` fillers) and
  multiplies them via slot expansion and WordNet synonym substitution — so a
  handful of seeds becomes a broader regression set that still exercises the
  same underlying behavior.
- **Categorical, ranked judgments — not one opaque score.** A `Judge` returns
  `correct` / `partially_correct` / `incorrect` plus a rationale, which is
  what makes a gap report possible: you can see *which* intent is weak and
  *why*, not just a single blended number.
- **Tiered judging for cost control.** `HeuristicJudge` (TF-IDF lexical
  overlap) is free and runs on every case. `TieredJudge` only escalates the
  uncertain (`partially_correct`) cases to a more expensive, more accurate
  judge — mirroring gating an expensive LLM/entailment check behind a cheap
  first-pass filter so it only runs on the fraction of cases that actually
  need it.
- **Gap reports, not pass/fail.** `build_gap_report` aggregates judgments per
  intent (accuracy, pass rate, failing examples) so the output tells you where
  to focus, the same shape as aggregating issues per intent into a report
  rather than a single top-line score.

## What's real vs. mocked

| Component | Status |
|---|---|
| Synthetic case generation (slot expansion + WordNet synonym substitution) | Real, fully functional |
| `HeuristicJudge` (TF-IDF lexical-overlap judge) | Real, runs locally, no API key needed |
| `TieredJudge` escalation logic | Real, fully functional |
| `LLMJudge` | Real integration code, **inactive without an API key** |
| The "candidate system" being evaluated in `examples/run_demo.py` | Mocked — a stand-in for whatever system you're actually evaluating |

`HeuristicJudge` is intentionally a rough proxy: lexical overlap can call a
correctly-rephrased answer "incorrect" because it shares few words with the
reference. That's realistic and is exactly the gap `LLMJudge` (real semantic
judgment) is meant to close once an API key is available — see the demo
output for a concrete example of this limitation.

## Run it

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
python examples/run_demo.py
```

No API key needed for the default run.

To try the LLM-backed judge for escalated cases instead:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
```

```python
from src import HeuristicJudge, LLMJudge, TieredJudge

judge = TieredJudge(fast_judge=HeuristicJudge(), precise_judge=LLMJudge())
```

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

## Layout

```
src/
  synthetic_generator.py   # seed -> slot expansion -> synonym substitution
  judge.py                  # HeuristicJudge (default), LLMJudge, TieredJudge
  scorer.py                  # score_cases + build_gap_report (per-intent rollup)
examples/run_demo.py         # end-to-end runnable demo with a mocked candidate system
tests/                        # pytest suite
```
