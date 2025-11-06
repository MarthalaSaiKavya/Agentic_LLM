"""Rule-based repair strategy inspired by constitutional AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..types import GenerationStep, UncertaintyLevel
from .base import SelfRepairStrategy


@dataclass(slots=True)
class ConstitutionalRepair(SelfRepairStrategy):
    """Applies declarative rules when uncertainty exceeds thresholds."""

    rules: Dict[UncertaintyLevel, List[str]] = field(
        default_factory=lambda: {
            UncertaintyLevel.LOW: [
                "Re-evaluate the last segment for factual accuracy.",
                "Detail the reasoning steps leading to the answer.",
            ],
            UncertaintyLevel.MODERATE: [
                "Clarify assumptions and provide alternative interpretations."
            ],
        }
    )

    def applies(self, step: GenerationStep) -> bool:
        if not step.token_scores:
            return False
        return step.token_scores[-1].level in self.rules

    def repair(self, step: GenerationStep) -> Optional[str]:
        rule_list = self.rules.get(step.token_scores[-1].level)
        if not rule_list:
            return None
        index = step.repair_attempt % len(rule_list)
        instruction = rule_list[index]
        return (
            "A self-repair was triggered:\n"
            f"- Detected uncertainty level: {step.token_scores[-1].level.name}\n"
            f"- Action: {instruction}"
        )
