"""Implementation of the LogTokU evidence-based uncertainty estimator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from ..config import ProjectConfig
from ..types import TokenScore, UncertaintyLevel
from ..utils.token_ops import entropy, to_probabilities
from .base import UncertaintyEstimator


def _softplus(arr: Sequence[float]) -> np.ndarray:
    return np.log1p(np.exp(np.asarray(arr)))


@dataclass(slots=True)
class LogTokUEstimator(UncertaintyEstimator):
    """Approximate LogTokU estimator operating on logits from a single LLM pass."""

    config: ProjectConfig

    def score(
        self, tokens: Iterable[str], logits: Iterable[Sequence[float]]
    ) -> Iterable[TokenScore]:
        tokens_list = list(tokens)
        logits_list = list(logits)
        scores: List[TokenScore] = []

        for token, logit_vector in zip(tokens_list, logits_list):
            evidence = _softplus(logit_vector)
            alpha = evidence + 1.0
            s_total = float(np.sum(alpha))
            probabilities = to_probabilities(logit_vector)

            # Safe guard for zero denominators in entropy and uncertainty calculations.
            if not probabilities:
                probabilities = [1.0]

            sampled_prob = max(probabilities)
            token_entropy = entropy(probabilities)
            aleatoric = self._aleatoric(alpha, s_total)
            epistemic = self._epistemic(alpha, s_total)
            total_uncertainty = min(1.0, aleatoric + epistemic)
            level = self._level(total_uncertainty)

            scores.append(
                TokenScore(
                    token=token,
                    logit=float(np.max(logit_vector)),
                    probability=float(sampled_prob),
                    entropy=float(token_entropy),
                    aleatoric=float(aleatoric),
                    epistemic=float(epistemic),
                    total_uncertainty=float(total_uncertainty),
                    level=level,
                )
            )
        return scores

    def _aleatoric(self, alpha: np.ndarray, s_total: float) -> float:
        probs = alpha / s_total
        return float(np.sum(probs * (1 - probs)))

    def _epistemic(self, alpha: np.ndarray, s_total: float) -> float:
        k = float(len(alpha))
        return float(k / (s_total + 1.0))

    def _level(self, total_uncertainty: float) -> UncertaintyLevel:
        thresholds = self.config.thresholds
        if total_uncertainty < 1 - thresholds.high_confidence:
            return UncertaintyLevel.HIGH_CONFIDENCE
        if total_uncertainty < 1 - thresholds.moderate_confidence:
            return UncertaintyLevel.MODERATE
        return UncertaintyLevel.LOW
