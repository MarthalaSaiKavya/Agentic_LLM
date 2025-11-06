"""Mock LLM providers for offline experimentation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from .base import LLMClient, TokenLogit


@dataclass(slots=True)
class DeterministicMockLLM(LLMClient):
    """LLM mock that emits scripted tokens with synthetic logits.

    The logits are crafted to simulate rising or falling confidence levels.
    """

    scripted_responses: Dict[str, List[str]] = field(default_factory=dict)
    vocab_size: int = 8

    def generate(self, prompt: str, *, max_tokens: int = 256) -> Iterable[TokenLogit]:
        tokens = self.scripted_responses.get(prompt, ["I", "am", "uncertain", "."])
        for index, token in enumerate(tokens[:max_tokens]):
            logits = self._synthesize_logits(index, len(tokens))
            yield TokenLogit(token=token, logits=logits)

    def _synthesize_logits(self, position: int, total: int) -> Sequence[float]:
        center = self.vocab_size // 2
        logits = []
        for i in range(self.vocab_size):
            base = -abs(i - center)
            adjustment = math.cos((position + 1) / total * math.pi)
            logits.append(base + adjustment)
        return logits
