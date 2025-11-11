"""Base pipeline for integrating uncertainty monitoring with LLM workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import ProjectConfig
from ..llm.base import LLMClient, StreamingBuffer, TokenLogit
from ..messaging.status import StatusMessenger
from ..repair.base import SelfRepairStrategy
from ..types import GenerationStep, TokenScore
from ..uncertainty.base import UncertaintyEstimator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineResult:
    """Final output of the pipeline."""

    step: GenerationStep
    messages: List[str]


@dataclass(slots=True)
class UncertaintyAwarePipeline:
    """Coordinates token monitoring, repair strategies, and user messaging."""

    llm: LLMClient
    estimator: UncertaintyEstimator
    strategies: List[SelfRepairStrategy]
    messenger: StatusMessenger
    config: ProjectConfig = field(default_factory=ProjectConfig)

    def run(self, prompt: str, *, max_tokens: int = 256) -> PipelineResult:
        base_prompt = prompt
        attempts = 0
        while attempts <= self.config.max_self_repairs:
            buffer = StreamingBuffer()
            step = GenerationStep(
                prompt=prompt,
                generated_tokens=[],
                token_scores=[],
                repair_attempt=attempts,
            )
            for token_logit in self.llm.generate(prompt, max_tokens=max_tokens):
                self._append_step(step, token_logit, buffer)
                if not step.token_scores:
                    continue
                last_score = step.token_scores[-1]
                self.messenger.notify_token(last_score)

            if not step.token_scores:
                step.final = True
                self.messenger.notify_completion(step)
                return PipelineResult(step=step, messages=[m.title for m in self.messenger.history])

            uncertainty_profile = self._compute_uncertainty_profile(step)
            self._log_patch_ranking(step, uncertainty_profile)

            final_score = step.token_scores[-1]
            if final_score.total_uncertainty < self.config.thresholds.repair_activation_uncertainty:
                step.final = True
                self.messenger.notify_completion(step)
                logger.info(
                    "Pipeline exiting without repair (uncertainty=%.3f threshold=%.3f).",
                    final_score.total_uncertainty,
                    self.config.thresholds.repair_activation_uncertainty,
                )
                return PipelineResult(step=step, messages=[m.title for m in self.messenger.history])

            if attempts == self.config.max_self_repairs:
                logger.info("Reached max repairs (%d); emitting final result.", self.config.max_self_repairs)
                self.messenger.notify_no_repair(step)
                step.final = True
                return PipelineResult(step=step, messages=[m.title for m in self.messenger.history])

            repair_instruction = self._execute_repair(step, uncertainty_profile)
            if not repair_instruction:
                logger.info("No repair strategy produced an instruction; stopping.")
                self.messenger.notify_no_repair(step)
                step.final = True
                return PipelineResult(step=step, messages=[m.title for m in self.messenger.history])

            self.messenger.notify_repair(step, repair_instruction)

            prompt = self._compose_repair_prompt(base_prompt, step.generated_tokens, repair_instruction)
            attempts += 1

        step.final = True
        self.messenger.notify_completion(step)
        return PipelineResult(step=step, messages=[m.title for m in self.messenger.history])

    def _append_step(
        self,
        step: GenerationStep,
        token_logit: TokenLogit,
        buffer: StreamingBuffer,
    ) -> None:
        buffer.append(token_logit)
        tokens, logits = buffer.snapshot()
        scores = list(self.estimator.score(tokens, logits))
        step.generated_tokens = tokens
        step.token_scores = list(scores)

    def _execute_repair(self, step: GenerationStep, profile: Optional[Dict[str, float]]) -> Optional[str]:
        for strategy in self._order_strategies(profile):
            if strategy.applies(step):
                logger.info(
                    "Selected repair strategy %s (avg_eu=%.3f avg_au=%.3f).",
                    strategy.__class__.__name__,
                    profile.get("avg_eu") if profile else -1,
                    profile.get("avg_au") if profile else -1,
                )
                return strategy.repair(step)
        return None

    def _compose_repair_prompt(
        self, base_prompt: str, generated_tokens: List[str], instruction: str
    ) -> str:
        output = " ".join(generated_tokens)
        return (
            f"{base_prompt}\n\n"
            "The assistant previously produced the following draft response:\n"
            f"{output}\n\n"
            "Revise the answer while following this directive:\n"
            f"{instruction}\n"
        )

    def _compute_uncertainty_profile(self, step: GenerationStep) -> Optional[Dict[str, float]]:
        if not step.token_scores:
            return None
        totals = np.array([score.total_uncertainty for score in step.token_scores], dtype=np.float32)
        epistemic = np.array([score.epistemic for score in step.token_scores], dtype=np.float32)
        aleatoric = np.array([score.aleatoric for score in step.token_scores], dtype=np.float32)
        profile = {
            "avg_total": float(np.mean(totals)),
            "avg_eu": float(np.mean(epistemic)),
            "avg_au": float(np.mean(aleatoric)),
            "max_total": float(np.max(totals)),
        }
        logger.debug(
            "Uncertainty profile computed: avg_total=%.3f avg_eu=%.3f avg_au=%.3f max_total=%.3f",
            profile["avg_total"],
            profile["avg_eu"],
            profile["avg_au"],
            profile["max_total"],
        )
        return profile

    def _log_patch_ranking(self, step: GenerationStep, profile: Optional[Dict[str, float]]) -> None:
        if not profile or not step.token_scores:
            return
        ranked = sorted(step.token_scores, key=lambda s: s.total_uncertainty, reverse=True)[:5]
        summary = ", ".join(f"{score.token or '[tok]'}:{score.total_uncertainty:.2f}" for score in ranked)
        logger.info("Top uncertain tokens: %s", summary)

    def _order_strategies(self, profile: Optional[Dict[str, float]]) -> List[SelfRepairStrategy]:
        if not profile:
            return list(self.strategies)
        eu = profile.get("avg_eu", 0.0)
        au = profile.get("avg_au", 0.0)
        if eu >= au:
            priority = ("method", "test", "line")
        else:
            priority = ("line", "test", "method")
        logger.debug(
            "Ordering strategies based on EU/AU (eu=%.3f au=%.3f priority=%s).", eu, au, priority
        )

        def focus_index(strategy: SelfRepairStrategy) -> int:
            focus = self._strategy_focus(strategy)
            try:
                return priority.index(focus)
            except ValueError:
                return len(priority)

        ordered = sorted(self.strategies, key=focus_index)
        return ordered

    @staticmethod
    def _strategy_focus(strategy: SelfRepairStrategy) -> str:
        focus = getattr(strategy, "repair_focus", None)
        if focus:
            return str(focus)
        name = strategy.__class__.__name__.lower()
        if "method" in name:
            return "method"
        if "line" in name:
            return "line"
        if "test" in name:
            return "test"
        if "constitution" in name:
            return "line"
        return "general"
