"""Base pipeline for integrating uncertainty monitoring with LLM workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from ..config import ProjectConfig
from ..llm.base import LLMClient, StreamingBuffer, TokenLogit
from ..messaging.status import StatusMessenger
from ..repair.base import SelfRepairStrategy
from ..types import GenerationStep, TokenScore
from ..uncertainty.base import UncertaintyEstimator


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

            final_score = step.token_scores[-1]
            if final_score.total_uncertainty < self.config.thresholds.repair_activation_uncertainty:
                step.final = True
                self.messenger.notify_completion(step)
                return PipelineResult(step=step, messages=[m.title for m in self.messenger.history])

            if attempts == self.config.max_self_repairs:
                self.messenger.notify_no_repair(step)
                step.final = True
                return PipelineResult(step=step, messages=[m.title for m in self.messenger.history])

            repair_instruction = self._execute_repair(step)
            if not repair_instruction:
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

    def _execute_repair(self, step: GenerationStep) -> Optional[str]:
        for strategy in self.strategies:
            if strategy.applies(step):
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
