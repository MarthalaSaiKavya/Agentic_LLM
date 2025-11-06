"""Interfaces for LLM providers that expose logits during generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(slots=True)
class TokenLogit:
    """Container for a token and its logit vector."""

    token: str
    logits: Sequence[float]


class LLMClient(ABC):
    """Abstract client that streams tokens and logits."""

    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int = 256) -> Iterable[TokenLogit]:
        """Yield successive tokens alongside their logits."""


class StreamingBuffer:
    """Aggregates streamed tokens for downstream consumers."""

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.logits: List[Sequence[float]] = []

    def append(self, token_logit: TokenLogit) -> None:
        self.tokens.append(token_logit.token)
        self.logits.append(token_logit.logits)

    def snapshot(self) -> Tuple[List[str], List[Sequence[float]]]:
        return list(self.tokens), list(self.logits)
