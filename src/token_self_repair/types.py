"""Shared dataclasses and enumerations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class UncertaintyLevel(Enum):
    """Discrete categories for token uncertainty."""

    HIGH_CONFIDENCE = auto()
    MODERATE = auto()
    LOW = auto()


@dataclass(slots=True)
class TokenScore:
    """Container for token-level uncertainty scores."""

    token: str
    logit: float
    probability: float
    entropy: float
    aleatoric: float
    epistemic: float
    total_uncertainty: float
    level: UncertaintyLevel


@dataclass(slots=True)
class GenerationStep:
    """Represents the partial state of an LLM generation."""

    prompt: str
    generated_tokens: List[str]
    token_scores: List[TokenScore]
    final: bool = False
    repair_attempt: int = 0
    repair_notes: Optional[str] = None
