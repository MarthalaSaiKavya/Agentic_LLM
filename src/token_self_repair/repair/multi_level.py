"""Multi-level repair strategies combining line, method, and test insights."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..types import GenerationStep, TokenScore
from ..uncertainty import UncertaintyAggregator, UncertaintyMap, UncertaintyScores
from .base import SelfRepairStrategy

logger = logging.getLogger(__name__)


def _scores_from_step(step: GenerationStep) -> Optional[UncertaintyScores]:
    if not step.token_scores:
        return None
    eu = np.array([score.epistemic for score in step.token_scores], dtype=np.float32)
    au = np.array([score.aleatoric for score in step.token_scores], dtype=np.float32)
    total = np.array([score.total_uncertainty for score in step.token_scores], dtype=np.float32)
    entropy = np.array([score.entropy for score in step.token_scores], dtype=np.float32)
    tokens = [score.token for score in step.token_scores]
    return UncertaintyScores(eu=eu, au=au, total=total, entropy=entropy, token_texts=tokens)


def _build_uncertainty_map(step: GenerationStep) -> Optional[UncertaintyMap]:
    scores = _scores_from_step(step)
    if not scores:
        return None
    aggregator = UncertaintyAggregator()
    response = "".join(step.generated_tokens).strip() or step.prompt
    try:
        u_map = aggregator.build_uncertainty_map(
            scores=scores, source_text=response, language="code", tokens=scores.token_texts or step.generated_tokens
        )
        return u_map
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build uncertainty map for repair step: %s", exc)
        return None


@dataclass(slots=True)
class LineLevelRepairStrategy(SelfRepairStrategy):
    """Targets the most uncertain line and requests a focused rewrite."""

    min_uncertainty: float = 0.35
    repair_focus: str = "line"

    def applies(self, step: GenerationStep) -> bool:
        applicable = bool(step.token_scores) and step.token_scores[-1].total_uncertainty >= self.min_uncertainty
        logger.debug(
            "LineLevelRepairStrategy applies=%s (uncertainty=%.3f threshold=%.3f)",
            applicable,
            step.token_scores[-1].total_uncertainty if step.token_scores else -1.0,
            self.min_uncertainty,
        )
        return applicable

    def repair(self, step: GenerationStep) -> Optional[str]:
        u_map = _build_uncertainty_map(step)
        if not u_map or not u_map.hotspots:
            logger.info("LineLevelRepairStrategy skipped: no hotspots available.")
            return None
        hotspot = next((h for h in u_map.hotspots if h.kind == "line"), None)
        if not hotspot:
            logger.info("LineLevelRepairStrategy found no line hotspot; falling back to first hotspot.")
            hotspot = u_map.hotspots[0]
        line_no = hotspot.identifier
        snippet = u_map.line_scores.get(line_no).text.strip() if u_map.line_scores.get(line_no) else ""
        instruction = (
            f"Focus on line {line_no} which shows high uncertainty (score={hotspot.score:.2f}). "
            "Rewrite this line to clarify logic and fix potential bugs. "
            f"Current line:\n{snippet or '(empty line)'}"
        )
        logger.info("LineLevelRepairStrategy issuing instruction for line %s.", line_no)
        step.repair_notes = instruction
        return instruction


@dataclass(slots=True)
class MethodLevelRepairStrategy(SelfRepairStrategy):
    """Escalates to method-level refactoring when persistent uncertainty remains."""

    min_attempt: int = 1
    repair_focus: str = "method"

    def applies(self, step: GenerationStep) -> bool:
        applicable = step.repair_attempt >= self.min_attempt and bool(step.token_scores)
        logger.debug(
            "MethodLevelRepairStrategy applies=%s (attempt=%d threshold=%d)",
            applicable,
            step.repair_attempt,
            self.min_attempt,
        )
        return applicable

    def repair(self, step: GenerationStep) -> Optional[str]:
        u_map = _build_uncertainty_map(step)
        if not u_map or not u_map.hotspots:
            logger.info("MethodLevelRepairStrategy skipped: no hotspots available.")
            return None
        hotspot = next((h for h in u_map.hotspots if h.kind == "method"), None)
        if not hotspot:
            logger.info("MethodLevelRepairStrategy found no method hotspot; using highest total hotspot.")
            hotspot = u_map.hotspots[0]
        method_name = hotspot.identifier
        method_score = hotspot.score
        instruction = (
            f"Method `{method_name}` remains uncertain (score={method_score:.2f}). "
            "Refactor this method with clearer contracts and add inline comments explaining tricky branches."
        )
        logger.info("MethodLevelRepairStrategy targeting method '%s'.", method_name)
        step.repair_notes = instruction
        return instruction


@dataclass(slots=True)
class TestGuidedRepairStrategy(SelfRepairStrategy):
    """Generates test-oriented repairs by inferring fragile tokens."""

    uncertainty_threshold: float = 0.25
    max_tokens: int = 10
    repair_focus: str = "test"

    def applies(self, step: GenerationStep) -> bool:
        applicable = bool(step.token_scores) and any(
            score.total_uncertainty >= self.uncertainty_threshold for score in step.token_scores
        )
        logger.debug(
            "TestGuidedRepairStrategy applies=%s (threshold=%.3f)",
            applicable,
            self.uncertainty_threshold,
        )
        return applicable

    def repair(self, step: GenerationStep) -> Optional[str]:
        uncertain_tokens = self._top_uncertain_tokens(step.token_scores)
        if not uncertain_tokens:
            logger.info("TestGuidedRepairStrategy skipped: no uncertain tokens extracted.")
            return None
        token_list = ", ".join(uncertain_tokens)
        instruction = (
            "Design regression tests targeting the unstable regions of the patch. "
            f"Focus on tokens/functions around: {token_list}. "
            "Add assertions that would fail if the bug resurfaces, then adjust the implementation to satisfy the new tests."
        )
        logger.info("TestGuidedRepairStrategy recommending tests for tokens: %s", token_list)
        step.repair_notes = instruction
        return instruction

    def _top_uncertain_tokens(self, scores: list[TokenScore]) -> list[str]:
        filtered = [s for s in scores if s.total_uncertainty >= self.uncertainty_threshold and s.token.strip()]
        filtered.sort(key=lambda s: s.total_uncertainty, reverse=True)
        tokens = []
        for score in filtered[: self.max_tokens]:
            token = score.token.strip()
            if token not in tokens:
                tokens.append(token)
        return tokens
