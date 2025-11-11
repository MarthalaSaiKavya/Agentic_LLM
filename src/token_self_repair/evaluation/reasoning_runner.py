"""Evaluation utilities for reasoning-centric uncertainty workflows."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .datasets import dataset_registry
from .metrics import expected_calibration_error, auroc, calibration_curve
from .judge import judge_answer
from ..pipelines.reasoning import ReasoningCoordinator, ReasoningResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReasoningSampleResult:
    prompt: str
    reference: str
    prediction: str
    final_uncertainty: float
    confidence: float
    correct: bool
    summary: str | None
    hotspots: List[tuple[str, str, float]]
    judge_explanation: str
    latency_seconds: float


@dataclass(slots=True)
class ReasoningBenchmarkResult:
    benchmark: str
    accuracy: float
    auroc: float
    calibration_error: float
    average_uncertainty: float
    average_latency: float
    calibration_bins: List[Tuple[float, float]]
    samples: List[ReasoningSampleResult]


class ReasoningEvaluationRunner:
    """Runs reasoning coordinators over benchmark datasets."""

    def __init__(self, coordinator_factory: Callable[[], ReasoningCoordinator]) -> None:
        self._factory = coordinator_factory

    def run(self, benchmark_name: str, *, max_samples: Optional[int] = None) -> ReasoningBenchmarkResult:
        registry = dataset_registry()
        if benchmark_name not in registry:
            raise KeyError(f"Benchmark '{benchmark_name}' not registered.")
        benchmark = registry[benchmark_name]

        sample_results: List[ReasoningSampleResult] = []
        uncertainties: List[float] = []
        latencies: List[float] = []

        for idx, sample in enumerate(benchmark.samples):
            if max_samples is not None and idx >= max_samples:
                break
            coordinator = self._factory()
            logger.info("Benchmark '%s': running sample %d prompt preview=%.60s", benchmark.name, idx + 1, sample.prompt)
            start = time.perf_counter()
            reasoning_result: ReasoningResult = coordinator.solve(sample.prompt)
            latency = time.perf_counter() - start
            latencies.append(latency)
            logger.info("Sample %d latency: %.3fs", idx + 1, latency)

            tokens = reasoning_result.pipeline_result.step.generated_tokens
            prediction = "".join(tokens).strip()
            if not prediction:
                prediction = ""  # ensure string

            final_uncertainty = (
                reasoning_result.pipeline_result.step.token_scores[-1].total_uncertainty
                if reasoning_result.pipeline_result.step.token_scores
                else 1.0
            )
            uncertainties.append(final_uncertainty)
            confidence = max(0.0, min(1.0, 1.0 - final_uncertainty))

            judge = judge_answer(sample.prompt, prediction, sample.reference)
            if judge.correct is None:
                fallback_correct = prediction.strip() == sample.reference.strip()
                correct = fallback_correct
                judge_explanation = judge.explanation + " Fallback to exact match comparison."
            else:
                correct = bool(judge.correct)
                judge_explanation = judge.explanation

            hotspots: List[tuple[str, str, float]] = []
            if reasoning_result.uncertainty_map:
                for hotspot in reasoning_result.uncertainty_map.hotspots[:5]:
                    identifier = hotspot.identifier
                    if hotspot.kind == "line":
                        identifier = f"line {identifier}"
                    elif hotspot.kind == "method":
                        identifier = f"method {identifier}"
                    hotspots.append((hotspot.kind, identifier, float(hotspot.score)))

            sample_results.append(
                ReasoningSampleResult(
                    prompt=sample.prompt,
                    reference=sample.reference,
                    prediction=prediction,
                    final_uncertainty=final_uncertainty,
                    confidence=confidence,
                    correct=correct,
                    summary=reasoning_result.summary,
                    hotspots=hotspots,
                    judge_explanation=judge_explanation,
                    latency_seconds=latency,
                )
            )

        if sample_results:
            labels = np.array([1.0 if s.correct else 0.0 for s in sample_results], dtype=float)
            confidences = 1.0 - np.array(uncertainties)
            accuracy = float(labels.mean())
            calibration_error = expected_calibration_error(confidences, labels)
            roc_auc = auroc(np.array(uncertainties), 1.0 - labels)
            cal_bins = calibration_curve(confidences, labels, bins=10)
            avg_uncertainty = float(np.mean(uncertainties))
            avg_latency = float(np.mean(latencies)) if latencies else 0.0
        else:
            accuracy = 0.0
            roc_auc = 0.5
            calibration_error = 0.0
            avg_uncertainty = 1.0
            avg_latency = 0.0
            cal_bins = (np.zeros(10), np.zeros(10))

        logger.info(
            "Benchmark '%s' complete: accuracy=%.3f auroc=%.3f ece=%.3f avg_uncertainty=%.3f avg_latency=%.3f",
            benchmark.name,
            accuracy,
            roc_auc,
            calibration_error,
            avg_uncertainty,
            avg_latency,
        )

        return ReasoningBenchmarkResult(
            benchmark=benchmark.name,
            accuracy=float(accuracy),
            auroc=float(roc_auc),
            calibration_error=float(calibration_error),
            average_uncertainty=avg_uncertainty,
            average_latency=avg_latency,
            calibration_bins=list(zip(cal_bins[0].tolist(), cal_bins[1].tolist())),
            samples=sample_results,
        )
