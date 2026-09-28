"""End-to-end demo: seed cases -> synthetic expansion -> judge -> gap report.

Scenario: a small FAQ-answering system for a fictional software product. We
have a handful of hand-written seed questions with known-correct reference
answers, and a "candidate system" that answers some of them well and some
of them poorly. The point isn't the fictional Q&A content — it's the
pipeline: expand a small seed set, score every case, and get a per-intent
report that tells you *where* the candidate system is weak, not just an
overall pass rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (
    HeuristicJudge,
    SeedCase,
    build_gap_report,
    generate_synthetic_cases,
    score_cases,
)

# --- Seed cases: a few real questions per intent, with a slot for product
# name so slot-expansion multiplies each seed across a couple of products. ---
SEEDS = [
    SeedCase(
        text="How do I reset my password for {product}?",
        intent="account_access",
        slots={"product": ["the mobile app", "the web dashboard"]},
    ),
    SeedCase(
        text="What is the refund policy if I cancel {product} within 30 days?",
        intent="billing",
        slots={"product": ["my subscription", "my annual plan"]},
    ),
    SeedCase(
        text="Why is {product} running slowly on my device?",
        intent="performance",
        slots={"product": ["the app", "the sync client"]},
    ),
]

# Reference answers, keyed by intent (used for every case under that intent —
# a simplification for the demo; a real eval set would pair each case with
# its own reference).
REFERENCE_ANSWERS = {
    "account_access": (
        "Go to the login screen, select 'Forgot password', and follow the "
        "emailed reset link to set a new password."
    ),
    "billing": (
        "Cancellations within 30 days of purchase are fully refunded to the "
        "original payment method within 5-10 business days."
    ),
    "performance": (
        "Slow performance is usually caused by a large local cache; clearing "
        "the cache in Settings > Storage typically resolves it."
    ),
}


def mock_candidate_system(question: str, intent: str) -> str:
    """Stands in for 'the system under test'. Deliberately answers some
    intents well and one intent poorly, so the gap report below has
    something real to surface. In a real eval this function would call
    whatever agent/RAG pipeline/model you're actually evaluating."""
    if intent == "account_access":
        return (
            "Click 'Forgot password' on the login page and follow the link "
            "sent to your email to choose a new password."
        )
    if intent == "billing":
        return (
            "Cancellations within 30 days of purchase are fully refunded to "
            "the original payment method within 5-10 business days."
        )
    # performance: deliberately vague/wrong to demonstrate the judge catching it
    return "Try restarting your device, that usually helps with most issues."


def main() -> None:
    synthetic_cases = generate_synthetic_cases(SEEDS, n_paraphrases_per_seed=2)
    print(f"Expanded {len(SEEDS)} seeds into {len(synthetic_cases)} synthetic cases.\n")

    scored_inputs = [
        (case, REFERENCE_ANSWERS[case.intent], mock_candidate_system(case.text, case.intent))
        for case in synthetic_cases
    ]

    judge = HeuristicJudge()
    scored_cases = score_cases(scored_inputs, judge)
    report = build_gap_report(scored_cases)

    print("Per-intent gap report (HeuristicJudge, TF-IDF lexical overlap):")
    for line in report.summary_lines():
        print("  " + line)

    print(f"\n{len(report.failing_examples)} case(s) judged incorrect:")
    for sc in report.failing_examples[:5]:
        print(f"  - [{sc.case.intent}] Q: {sc.case.text!r}")
        print(f"    candidate: {sc.candidate_response!r}")
        print(f"    verdict: {sc.judgment.verdict.value} ({sc.judgment.rationale})")

    print(
        "\nNote: HeuristicJudge is lexical-overlap only, so it's a rough proxy "
        "-- it can call a rephrased-but-correct answer 'incorrect'. TieredJudge "
        "would escalate uncertain (partially_correct) cases to LLMJudge for a "
        "real semantic read; that path needs ANTHROPIC_API_KEY and isn't run "
        "here. See README for how to enable it."
    )


if __name__ == "__main__":
    main()
