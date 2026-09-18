"""Aggregation and reporting.

Runs a `Judge` over a batch of (synthetic case, candidate response) pairs and
rolls the individual judgments up into a per-intent gap report — mirroring
the "aggregate issues per intent into a gap report" pattern used for
turn-level agent eval frameworks, rather than a single blended pass/fail
number that hides which slice of behavior is actually weak.
"""
from __future__ import annotations

from dataclasses import dataclass

from .judge import Judge, Judgment, Verdict
from .synthetic_generator import SyntheticCase


@dataclass
class ScoredCase:
    case: SyntheticCase
    candidate_response: str
    judgment: Judgment


@dataclass
class IntentReport:
    intent: str
    total: int
    correct: int
    partially_correct: int
    incorrect: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def pass_rate(self) -> float:
        """Correct + partially_correct counted as "not a failure" — a looser
        bar than strict accuracy, useful for triage: incorrect is the bucket
        that needs attention first."""
        return (self.correct + self.partially_correct) / self.total if self.total else 0.0


@dataclass
class GapReport:
    intent_reports: list[IntentReport]
    failing_examples: list[ScoredCase]

    def summary_lines(self) -> list[str]:
        lines = []
        for r in sorted(self.intent_reports, key=lambda r: r.accuracy):
            lines.append(
                f"{r.intent:>20s}  n={r.total:<4d}  "
                f"accuracy={r.accuracy:.0%}  pass_rate={r.pass_rate:.0%}  "
                f"(correct={r.correct} partial={r.partially_correct} incorrect={r.incorrect})"
            )
        return lines


def score_cases(
    scored_inputs: list[tuple[SyntheticCase, str, str]],
    judge: Judge,
) -> list[ScoredCase]:
    """`scored_inputs` is a list of (synthetic_case, reference_answer,
    candidate_response) triples. Returns one `ScoredCase` per input."""
    results = []
    for case, reference_answer, candidate_response in scored_inputs:
        judgment = judge.judge(case.text, reference_answer, candidate_response)
        results.append(ScoredCase(case=case, candidate_response=candidate_response, judgment=judgment))
    return results


def build_gap_report(scored_cases: list[ScoredCase]) -> GapReport:
    by_intent: dict[str, list[ScoredCase]] = {}
    for sc in scored_cases:
        by_intent.setdefault(sc.case.intent, []).append(sc)

    intent_reports = []
    for intent, cases in by_intent.items():
        correct = sum(1 for c in cases if c.judgment.verdict == Verdict.CORRECT)
        partial = sum(1 for c in cases if c.judgment.verdict == Verdict.PARTIALLY_CORRECT)
        incorrect = sum(1 for c in cases if c.judgment.verdict == Verdict.INCORRECT)
        intent_reports.append(
            IntentReport(
                intent=intent,
                total=len(cases),
                correct=correct,
                partially_correct=partial,
                incorrect=incorrect,
            )
        )

    failing = [sc for sc in scored_cases if sc.judgment.verdict == Verdict.INCORRECT]
    return GapReport(intent_reports=intent_reports, failing_examples=failing)
