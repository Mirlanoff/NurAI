from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[\wа-яА-ЯёЁ]+")


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}


def retrieval_recall(expected_keywords: Iterable[str], retrieved_texts: Iterable[str]) -> float:
    """Fraction of expected keywords/phrases that appear in any retrieved chunk.

    Matching is case-insensitive substring matching so multi-word keywords are
    treated as a single phrase.
    """
    keywords = [keyword.strip().lower() for keyword in expected_keywords if keyword.strip()]
    if not keywords:
        return 1.0
    haystack = " ".join(text.lower() for text in retrieved_texts)
    if not haystack:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in haystack)
    return hits / len(keywords)


def answer_overlap(predicted: str, ground_truth: str) -> float:
    """Token-level Jaccard overlap between predicted answer and ground truth."""
    predicted_tokens = _tokens(predicted)
    truth_tokens = _tokens(ground_truth)
    if not predicted_tokens and not truth_tokens:
        return 1.0
    if not predicted_tokens or not truth_tokens:
        return 0.0
    intersection = predicted_tokens & truth_tokens
    union = predicted_tokens | truth_tokens
    return len(intersection) / len(union)
