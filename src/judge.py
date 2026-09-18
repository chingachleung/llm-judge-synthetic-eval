"""LLM-as-judge scoring.

A `Judge` reads a (question, reference_answer, candidate_response) triple and
returns a categorical, ranked label — the same "categorical ranked-label
judge" pattern used for turn-level and dialog-level evaluation of agentic
systems, rather than a single opaque scalar score.

Two implementations:

- `HeuristicJudge`: TF-IDF similarity between candidate and reference, mapped
  to categorical bins. Deterministic, zero-dependency, runs by default.
- `LLMJudge`: real prompt + parsing logic for an actual LLM-as-judge call.
  Correct, complete, and NOT active by default (no bundled API key) — this
  is what you'd swap in for real semantic judgment rather than lexical
  overlap.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Verdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


@dataclass
class Judgment:
    verdict: Verdict
    rationale: str
    raw_score: float | None = None


class Judge(Protocol):
    def judge(self, question: str, reference_answer: str, candidate_response: str) -> Judgment: ...


class HeuristicJudge:
    """Lexical-overlap stand-in judge. Cheap, deterministic, no external
    calls — useful as a fast first pass before routing uncertain cases to a
    more expensive LLM judge (see `TieredJudge` below)."""

    def __init__(self, correct_floor: float = 0.55, partial_floor: float = 0.25):
        self.correct_floor = correct_floor
        self.partial_floor = partial_floor

    def judge(self, question: str, reference_answer: str, candidate_response: str) -> Judgment:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([reference_answer, candidate_response])
        score = float(cosine_similarity(matrix[0], matrix[1])[0][0])

        if score >= self.correct_floor:
            verdict = Verdict.CORRECT
        elif score >= self.partial_floor:
            verdict = Verdict.PARTIALLY_CORRECT
        else:
            verdict = Verdict.INCORRECT

        return Judgment(
            verdict=verdict,
            rationale=f"lexical-overlap similarity to reference = {score:.2f}",
            raw_score=score,
        )


class LLMJudge:
    """Real (but inactive-by-default) LLM-as-judge. Reads the question,
    reference answer, and candidate response, and asks the model for a
    categorical verdict plus a short rationale — real semantic judgment
    instead of lexical overlap."""

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def judge(self, question: str, reference_answer: str, candidate_response: str) -> Judgment:
        if not self.api_key:
            raise RuntimeError(
                "LLMJudge requires ANTHROPIC_API_KEY. Use HeuristicJudge for the no-key demo path."
            )

        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        prompt = (
            "You are grading an AI agent's response.\n"
            f"Question: {question}\n"
            f"Reference answer: {reference_answer}\n"
            f"Candidate response: {candidate_response}\n\n"
            "Grade the candidate as one of: correct, partially_correct, incorrect.\n"
            'Respond with JSON only: {"verdict": "...", "rationale": "..."}'
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        payload = json.loads(response.content[0].text)
        return Judgment(verdict=Verdict(payload["verdict"]), rationale=payload["rationale"])


class TieredJudge:
    """Cost-aware pattern: run the cheap heuristic judge first, and only pay
    for the LLM judge on cases the heuristic judge is unsure about (i.e.
    landed on `partially_correct`). Mirrors gating an expensive
    entailment/LLM check behind a cheap first-pass filter so it only runs on
    the fraction of cases that actually need it."""

    def __init__(self, fast_judge: Judge, precise_judge: Judge):
        self.fast_judge = fast_judge
        self.precise_judge = precise_judge

    def judge(self, question: str, reference_answer: str, candidate_response: str) -> Judgment:
        fast_result = self.fast_judge.judge(question, reference_answer, candidate_response)
        if fast_result.verdict != Verdict.PARTIALLY_CORRECT:
            return fast_result
        return self.precise_judge.judge(question, reference_answer, candidate_response)
