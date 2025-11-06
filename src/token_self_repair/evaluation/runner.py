"""Orchestrates evaluation runs over benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from .datasets import dataset_registry
from .metrics import expected_calibration_error
from ..pipelines.base import UncertaintyAwarePipeline


@dataclass(slots=True)
class EvaluationResult:
    benchmark: str
    score: float
    calibration_error: float
    predictions: List[str]
    references: List[str]
    uncertainties: List[float]
    status_counts: dict[str, int]


@dataclass(slots=True)
class EvaluationRunner:
    """Runs the self-repair pipeline across benchmark datasets."""

    pipeline_factory: Callable[[], UncertaintyAwarePipeline]

    def run(self, benchmark_name: str) -> EvaluationResult:
        registry = dataset_registry()
        if benchmark_name not in registry:
            raise KeyError(f"Benchmark '{benchmark_name}' not registered.")
        benchmark = registry[benchmark_name]
        predictions: List[str] = []
        references: List[str] = []
        uncertainties: List[float] = []
        status_counts = {
            "High confidence response": 0,
            "Moderate uncertainty detected - refining answer": 0,
            "Low confidence - results may vary": 0,
        }

        for sample in benchmark.samples:
            pipeline = self.pipeline_factory()
            result = pipeline.run(sample.prompt)
            output = " ".join(result.step.generated_tokens)
            predictions.append(output)
            references.append(sample.reference)
            final_uncertainty = (
                result.step.token_scores[-1].total_uncertainty
                if result.step.token_scores
                else 1.0
            )
            uncertainties.append(final_uncertainty)
            for message in pipeline.messenger.history:
                status_counts[message.title] = status_counts.get(message.title, 0) + 1

        labels = np.array([int(p == r) for p, r in zip(predictions, references)], dtype=float)
        probabilities = 1.0 - np.array(uncertainties)
        calibration_error = expected_calibration_error(probabilities, labels)
        score = benchmark.metric(predictions, references)

        return EvaluationResult(
            benchmark=benchmark.name,
            score=score,
            calibration_error=float(calibration_error),
            predictions=predictions,
            references=references,
            uncertainties=uncertainties,
            status_counts=status_counts,
        )
