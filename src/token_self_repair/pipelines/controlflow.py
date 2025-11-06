"""Adapter that injects uncertainty monitoring into ControlFlow-style agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from ..messaging.status import StatusMessenger
from ..repair.base import SelfRepairStrategy
from ..uncertainty.base import UncertaintyEstimator
from ..config import ProjectConfig
from ..llm.base import LLMClient
from .base import PipelineResult, UncertaintyAwarePipeline


@dataclass(slots=True)
class ControlFlowStage:
    """Represents a single stage in a ControlFlow graph."""

    name: str
    prompt_template: str

    def render(self, task: str) -> str:
        return self.prompt_template.format(task=task)


@dataclass(slots=True)
class ControlFlowCoordinator:
    """Sequentially executes ControlFlow stages with uncertainty-aware monitoring."""

    llm: LLMClient
    estimator: UncertaintyEstimator
    strategies: List[SelfRepairStrategy]
    messenger: StatusMessenger = field(default_factory=StatusMessenger)
    config: ProjectConfig = field(default_factory=ProjectConfig)

    def __post_init__(self) -> None:
        self.pipeline = UncertaintyAwarePipeline(
            llm=self.llm,
            estimator=self.estimator,
            strategies=self.strategies,
            messenger=self.messenger,
            config=self.config,
        )

    def run(self, task: str, stages: Iterable[ControlFlowStage]) -> List[PipelineResult]:
        results: List[PipelineResult] = []
        for stage in stages:
            prompt = stage.render(task)
            self.messenger.logger.info(f"Executing ControlFlow stage '{stage.name}'")
            result = self.pipeline.run(prompt)
            results.append(result)
        return results
