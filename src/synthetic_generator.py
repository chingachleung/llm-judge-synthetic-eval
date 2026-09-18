"""Synthetic test-case generation.

Given a small set of seed utterances tied to a use case (the kind of thing
you'd otherwise have to hand-write hundreds of variations of), generate
linguistic variation automatically:

1. WordNet synonym substitution — swaps content words for synonyms so the
   generated corpus isn't just literal repeats of the seed phrasing.
2. Slot-templated generation — seeds can declare a `{slot}` and a list of
   fillers (e.g. plan names, product names) to combine combinatorially.

This mirrors mining-and-expanding a smoke-test / eval set from a small seed
set rather than hand-authoring every case — the same idea as expanding an
offline eval set to catch cases a narrow hand-written set would miss.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from nltk.corpus import wordnet as wn


@dataclass
class SeedCase:
    text: str
    intent: str
    slots: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class SyntheticCase:
    text: str
    intent: str
    source_seed: str
    generation_method: str


def _synonym_candidates(word: str) -> list[str]:
    candidates: set[str] = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                candidates.add(name)
    return sorted(candidates)


def synonym_substitute(text: str, rng: random.Random, max_substitutions: int = 2) -> str:
    words = text.split()
    editable_idx = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 3]
    rng.shuffle(editable_idx)

    substitutions = 0
    for idx in editable_idx:
        if substitutions >= max_substitutions:
            break
        candidates = _synonym_candidates(words[idx])
        if candidates:
            words[idx] = rng.choice(candidates)
            substitutions += 1
    return " ".join(words)


def expand_slots(seed: SeedCase) -> list[str]:
    if not seed.slots:
        return [seed.text]
    texts = [seed.text]
    for slot, fillers in seed.slots.items():
        placeholder = "{" + slot + "}"
        texts = [t.replace(placeholder, filler) for t in texts for filler in fillers] if placeholder in seed.text else texts
    return texts


def generate_synthetic_cases(
    seeds: list[SeedCase], n_paraphrases_per_seed: int = 2, seed_rng: int = 13
) -> list[SyntheticCase]:
    rng = random.Random(seed_rng)
    out: list[SyntheticCase] = []

    for seed in seeds:
        slot_expanded = expand_slots(seed)
        for base_text in slot_expanded:
            out.append(SyntheticCase(base_text, seed.intent, seed.text, "slot_expansion"))
            for _ in range(n_paraphrases_per_seed):
                paraphrase = synonym_substitute(base_text, rng)
                if paraphrase != base_text:
                    out.append(SyntheticCase(paraphrase, seed.intent, seed.text, "synonym_substitution"))
    return out
