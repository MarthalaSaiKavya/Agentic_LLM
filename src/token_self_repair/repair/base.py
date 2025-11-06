"""Base classes for self-repair strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from ..types import GenerationStep


class SelfRepairStrategy(ABC):
    """Abstract base class for repair policies."""

    @abstractmethod
    def applies(self, step: GenerationStep) -> bool:
        """Return True if the strategy should execute on the given step."""

    @abstractmethod
    def repair(self, step: GenerationStep) -> Optional[str]:
        """Return a corrective instruction or None if no repair occurred."""


class RepairContext:
    """State container that survives across repair attempts."""

    def __init__(self) -> None:
        self.history: Iterable[GenerationStep] = []
