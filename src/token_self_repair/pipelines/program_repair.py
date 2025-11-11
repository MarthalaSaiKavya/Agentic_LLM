"""Coordinator for multi-level program repair workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..config import ProjectConfig
from ..llm.base import LLMClient
from ..messaging.status import StatusMessenger
from ..repair.base import SelfRepairStrategy
from ..repair.multi_level import (
    LineLevelRepairStrategy,
    MethodLevelRepairStrategy,
    TestGuidedRepairStrategy,
)
from ..types import GenerationStep
from ..uncertainty import LogTokUEstimator, UncertaintyAggregator, UncertaintyMap, UncertaintyScores
from ..uncertainty.base import UncertaintyEstimator
from .base import PipelineResult, UncertaintyAwarePipeline

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProgramRepairResult:
    """Structured output produced by the repair coordinator."""

    patch: str
    pipeline_result: PipelineResult
    uncertainty_map: Optional[UncertaintyMap]
    repair_messages: List[str]


@dataclass(slots=True)
class ProgramRepairCoordinator:
    """Runs the UncertaintyAwarePipeline with multi-level repair strategies."""

    llm: LLMClient
    estimator: UncertaintyEstimator = field(default_factory=LogTokUEstimator)
    strategies: List[SelfRepairStrategy] = field(default_factory=list)
    messenger: StatusMessenger = field(default_factory=StatusMessenger)
    config: ProjectConfig = field(default_factory=ProjectConfig)
    aggregator: UncertaintyAggregator = field(default_factory=UncertaintyAggregator)
    pipeline: UncertaintyAwarePipeline = field(init=False)

    def __post_init__(self) -> None:
        if not self.strategies:
            self.strategies = [
                LineLevelRepairStrategy(),
                MethodLevelRepairStrategy(),
                TestGuidedRepairStrategy(),
            ]
        self.pipeline = UncertaintyAwarePipeline(
            llm=self.llm,
            estimator=self.estimator,
            strategies=self.strategies,
            messenger=self.messenger,
            config=self.config,
        )
        logger.info(
            "ProgramRepairCoordinator initialized with %d repair strategy(ies).", len(self.strategies)
        )

    def repair(
        self,
        buggy_code: str,
        *,
        description: str | None = None,
        failing_tests: Optional[List[str]] = None,
        max_tokens: int = 256,
    ) -> ProgramRepairResult:
        """Execute self-repair for the provided buggy code snippet."""

        prompt = self._build_prompt(buggy_code, description, failing_tests)
        logger.info(
            "Program repair run started (description=%s tests=%d).",
            bool(description),
            len(failing_tests or []),
        )
        pipeline_result = self.pipeline.run(prompt, max_tokens=max_tokens)
        patch = "".join(pipeline_result.step.generated_tokens).strip()
        u_map = self._build_uncertainty_map(pipeline_result.step)
        logger.info("Program repair run completed (tokens=%d).", len(pipeline_result.step.generated_tokens))
        return ProgramRepairResult(
            patch=patch,
            pipeline_result=pipeline_result,
            uncertainty_map=u_map,
            repair_messages=self.messenger.history.copy(),
        )

    def _build_prompt(
        self,
        buggy_code: str,
        description: Optional[str],
        failing_tests: Optional[List[str]],
    ) -> str:
        sections = [
            "You are a program repair assistant. Generate a minimal patch that fixes the issue while preserving style.",
            "Buggy code:",
            buggy_code,
        ]
        if description:
            sections.extend(["Bug report / context:", description])
        if failing_tests:
            sections.append("Failing tests:")
            sections.extend(f"- {test}" for test in failing_tests)
        sections.append("Provide the repaired code snippet:")
        prompt = "\n\n".join(sections)
        logger.debug("Program repair prompt composed with length %d.", len(prompt))
        return prompt

    def _build_uncertainty_map(self, step: GenerationStep) -> Optional[UncertaintyMap]:
        if not step.token_scores:
            return None
        scores = UncertaintyScores(
            eu=np.array([score.epistemic for score in step.token_scores], dtype=np.float32),
            au=np.array([score.aleatoric for score in step.token_scores], dtype=np.float32),
            total=np.array([score.total_uncertainty for score in step.token_scores], dtype=np.float32),
            entropy=np.array([score.entropy for score in step.token_scores], dtype=np.float32),
            token_texts=[score.token for score in step.token_scores],
        )
        response = "".join(step.generated_tokens).strip()
        try:
            return self.aggregator.build_uncertainty_map(scores, source_text=response, language="code", tokens=step.generated_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to build uncertainty map for repair output: %s", exc)
            return None


def default_program_repair_coordinator(model: LLMClient) -> ProgramRepairCoordinator:
    """Helper that wires the default strategies."""

    return ProgramRepairCoordinator(llm=model)
