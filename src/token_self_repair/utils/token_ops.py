"""Utility helpers for working with token streams."""

from __future__ import annotations

from typing import Iterable, List


def to_probabilities(logits: Iterable[float]) -> List[float]:
    """Convert logits to probabilities with log-sum-exp stability."""
    import math

    logits_list = list(logits)
    if not logits_list:
        return []
    max_logit = max(logits_list)
    exp_shifted = [math.exp(l - max_logit) for l in logits_list]
    denom = sum(exp_shifted)
    return [val / denom for val in exp_shifted]


def entropy(probs: Iterable[float]) -> float:
    """Compute Shannon entropy in nats."""
    import math

    total = 0.0
    for p in probs:
        if p > 0:
            total -= p * math.log(p)
    return total
