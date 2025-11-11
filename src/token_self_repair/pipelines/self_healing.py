"""Self-Healing LLM pipeline integration with uncertainty-driven repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import ProjectConfig
from ..llm.base import LLMClient
from ..messaging.status import StatusMessenger
from ..repair.base import SelfRepairStrategy
from ..uncertainty.base import UncertaintyEstimator
from .base import PipelineResult, UncertaintyAwarePipeline


ANALYZE_PROMPT = (
    "You are a deliberate assistant. Analyze the task carefully and draft an initial answer.\n"
    "Task: {task}"
)

CRITIQUE_PROMPT = (
    "Self-critique the previous answer, focusing on the uncertainty signals noted below, "
    "then produce a refined solution.\n"
    "Task: {task}\n"
    "Draft answer: {draft}\n"
    "Uncertainty summary: {summary}"
)


@dataclass(slots=True)
class SelfHealingCoordinator:
    """Executes the analyze-critique-refine loop with token-level monitoring."""

    llm: LLMClient
    estimator: UncertaintyEstimator
    strategies: list[SelfRepairStrategy]
    messenger: StatusMessenger = field(default_factory=StatusMessenger)
    config: ProjectConfig = field(default_factory=ProjectConfig)
    pipeline: UncertaintyAwarePipeline = field(init=False)

    def __post_init__(self) -> None:
        self.pipeline = UncertaintyAwarePipeline(
            llm=self.llm,
            estimator=self.estimator,
            strategies=self.strategies,
            messenger=self.messenger,
            config=self.config,
        )

    def solve(self, task: str) -> PipelineResult:
        analyze_prompt = ANALYZE_PROMPT.format(task=task)
        analyze_result = self.pipeline.run(analyze_prompt)

        if self._is_confident(analyze_result):
            return analyze_result

        critique_prompt = CRITIQUE_PROMPT.format(
            task=task,
            draft=" ".join(analyze_result.step.generated_tokens),
            summary=self._summarize_uncertainty(analyze_result),
        )
        return self.pipeline.run(critique_prompt)

    def _is_confident(self, result: PipelineResult) -> bool:
        if not result.step.token_scores:
            return False
        return (
            result.step.token_scores[-1].total_uncertainty
            < self.config.thresholds.repair_activation_uncertainty
        )

    def _summarize_uncertainty(self, result: PipelineResult) -> str:
        summaries = []
        for score in result.step.token_scores[-5:]:
            summaries.append(
                f"{score.token} -> level={score.level.name}, U={score.total_uncertainty:.2f}"
            )
        return "; ".join(summaries) if summaries else "No uncertainty available."
