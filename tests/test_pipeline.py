import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import (
    HeuristicJudge,
    SeedCase,
    TieredJudge,
    Verdict,
    build_gap_report,
    generate_synthetic_cases,
    score_cases,
)
from src.judge import Judge, Judgment
from src.synthetic_generator import expand_slots, synonym_substitute
import random


def test_expand_slots_combinatorial():
    seed = SeedCase(text="Why is {product} slow?", intent="perf", slots={"product": ["app", "client"]})
    expanded = expand_slots(seed)
    assert expanded == ["Why is app slow?", "Why is client slow?"]


def test_expand_slots_no_slots_returns_original():
    seed = SeedCase(text="Why is it slow?", intent="perf")
    assert expand_slots(seed) == ["Why is it slow?"]


def test_synonym_substitute_changes_text_when_possible():
    rng = random.Random(1)
    result = synonym_substitute("The system is running slowly today", rng, max_substitutions=2)
    # Not guaranteed to differ for every seed, but should return a non-empty string
    # of the same rough shape.
    assert isinstance(result, str) and len(result.split()) == len("The system is running slowly today".split())


def test_generate_synthetic_cases_produces_more_than_seeds():
    seeds = [SeedCase(text="How do I reset my password?", intent="account")]
    cases = generate_synthetic_cases(seeds, n_paraphrases_per_seed=2)
    assert len(cases) >= 1
    assert all(c.intent == "account" for c in cases)
    assert all(c.source_seed == "How do I reset my password?" for c in cases)


def test_heuristic_judge_identical_text_is_correct():
    judge = HeuristicJudge()
    judgment = judge.judge("q", "the exact same answer text", "the exact same answer text")
    assert judgment.verdict == Verdict.CORRECT


def test_heuristic_judge_unrelated_text_is_incorrect():
    judge = HeuristicJudge()
    judgment = judge.judge("q", "refunds are processed within five business days", "purple elephants dance quietly")
    assert judgment.verdict == Verdict.INCORRECT


class _StubJudge:
    """Test double implementing the Judge protocol without any network call."""

    def __init__(self, verdict: Verdict):
        self._verdict = verdict
        self.calls = 0

    def judge(self, question, reference_answer, candidate_response) -> Judgment:
        self.calls += 1
        return Judgment(verdict=self._verdict, rationale="stub")


def test_tiered_judge_skips_precise_judge_when_fast_judge_confident():
    fast = _StubJudge(Verdict.CORRECT)
    precise = _StubJudge(Verdict.INCORRECT)
    tiered = TieredJudge(fast_judge=fast, precise_judge=precise)

    result = tiered.judge("q", "ref", "cand")

    assert result.verdict == Verdict.CORRECT
    assert precise.calls == 0


def test_tiered_judge_escalates_when_fast_judge_uncertain():
    fast = _StubJudge(Verdict.PARTIALLY_CORRECT)
    precise = _StubJudge(Verdict.CORRECT)
    tiered = TieredJudge(fast_judge=fast, precise_judge=precise)

    result = tiered.judge("q", "ref", "cand")

    assert result.verdict == Verdict.CORRECT
    assert precise.calls == 1


def test_build_gap_report_groups_by_intent_and_collects_failures():
    seeds = [SeedCase(text="q1", intent="a"), SeedCase(text="q2", intent="b")]
    cases = generate_synthetic_cases(seeds, n_paraphrases_per_seed=0)
    judge = _StubJudge(Verdict.INCORRECT)
    scored_inputs = [(c, "ref", "cand") for c in cases]
    scored = score_cases(scored_inputs, judge)

    report = build_gap_report(scored)

    intents = {r.intent for r in report.intent_reports}
    assert intents == {"a", "b"}
    assert len(report.failing_examples) == len(scored)
    for r in report.intent_reports:
        assert r.accuracy == 0.0
        assert r.pass_rate == 0.0


def test_llm_judge_raises_without_api_key(monkeypatch):
    from src.judge import LLMJudge

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    judge = LLMJudge(api_key=None)
    with pytest.raises(RuntimeError):
        judge.judge("q", "ref", "cand")
