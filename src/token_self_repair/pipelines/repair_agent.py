"""RepairAgent-style integration for code generation and automated fixes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from ..config import ProjectConfig
from ..llm.base import LLMClient
from ..messaging.status import StatusMessenger
from ..repair.base import SelfRepairStrategy
from ..uncertainty.base import UncertaintyEstimator
from .base import PipelineResult, UncertaintyAwarePipeline


REPAIR_PROMPT = (
    "You are an automated code repair agent. Fix the failing code region while "
    "respecting the constraints.\n"
    "Bug report: {bug_report}\n"
    "Failing tests:\n{failing_tests}\n"
    "Existing patch:\n{context}\n"
)


@dataclass(slots=True)
class RepairAgentCoordinator:
    """Wraps a self-repair pipeline for iterative code patching."""

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

    def repair(
        self,
        bug_report: str,
        failing_tests: Iterable[str],
        context: str,
    ) -> PipelineResult:
        prompt = REPAIR_PROMPT.format(
            bug_report=bug_report,
            failing_tests="\n".join(failing_tests),
            context=context,
        )
        return self.pipeline.run(prompt)
