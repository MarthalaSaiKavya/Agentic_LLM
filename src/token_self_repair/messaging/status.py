"""Messaging layer for uncertainty and repair events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..types import GenerationStep, TokenScore, UncertaintyLevel
from ..utils.logging import Logger


@dataclass(slots=True)
class StatusMessage:
    """Represents a message communicated to the end user."""

    level: UncertaintyLevel
    title: str
    detail: str


@dataclass(slots=True)
class StatusMessenger:
    """Publishes human-readable messages about uncertainty and repairs."""

    logger: Logger = field(default_factory=lambda: Logger("Messenger"))
    history: List[StatusMessage] = field(default_factory=list)

    def notify_token(self, score: TokenScore) -> None:
        title, detail = self._describe(score)
        message = StatusMessage(level=score.level, title=title, detail=detail)
        self.history.append(message)
        self.logger.info(f"{title} — {detail}", status=score.level.name)

    def notify_repair(self, step: GenerationStep, instruction: str) -> None:
        title = "Moderate uncertainty detected - refining answer"
        if step.token_scores[-1].level == UncertaintyLevel.LOW:
            title = "Low confidence - running corrective pass"
        message = StatusMessage(
            level=step.token_scores[-1].level,
            title=title,
            detail=instruction,
        )
        self.history.append(message)
        self.logger.info(instruction, status="REPAIR")

    def notify_completion(self, step: GenerationStep) -> None:
        token_status = step.token_scores[-1].level if step.token_scores else UncertaintyLevel.HIGH_CONFIDENCE
        title = "High confidence response" if token_status == UncertaintyLevel.HIGH_CONFIDENCE else "Response delivered with residual uncertainty"
        detail = f"Generated {len(step.generated_tokens)} tokens across {step.repair_attempt + 1} pass(es)."
        message = StatusMessage(level=token_status, title=title, detail=detail)
        self.history.append(message)
        self.logger.info(detail, status="COMPLETE")

    def notify_no_repair(self, step: GenerationStep) -> None:
        detail = "No repair strategy applicable; delivering best-effort output."
        message = StatusMessage(
            level=step.token_scores[-1].level,
            title="Uncertainty persisted - user review suggested",
            detail=detail,
        )
        self.history.append(message)
        self.logger.info(detail, status="SKIP")

    def _describe(self, score: TokenScore) -> tuple[str, str]:
        mapping = {
            UncertaintyLevel.HIGH_CONFIDENCE: (
                "High confidence response",
                f"Token '{score.token}' exhibits low uncertainty (p={score.probability:.2f}).",
            ),
            UncertaintyLevel.MODERATE: (
                "Moderate uncertainty detected - refining answer",
                f"Token '{score.token}' shows moderate uncertainty (U={score.total_uncertainty:.2f}).",
            ),
            UncertaintyLevel.LOW: (
                "Low confidence - results may vary",
                f"Token '{score.token}' is highly uncertain (U={score.total_uncertainty:.2f}).",
            ),
        }
        return mapping[score.level]
