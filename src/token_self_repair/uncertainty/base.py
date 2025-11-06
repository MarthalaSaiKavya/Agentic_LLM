"""Uncertainty estimation interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from ..types import TokenScore


class UncertaintyEstimator(ABC):
    """Base class for token-level uncertainty estimators."""

    @abstractmethod
    def score(self, tokens: Iterable[str], logits: Iterable[Sequence[float]]) -> Iterable[TokenScore]:
        """Compute uncertainty scores for each generated token."""
