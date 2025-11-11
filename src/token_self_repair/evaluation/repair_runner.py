"""Evaluation utilities for program repair benchmarks."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .datasets import TaskSample, dataset_registry
from .metrics import pass_at_k
from ..pipelines.program_repair import ProgramRepairCoordinator, ProgramRepairResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RepairSampleResult:
    prompt: str
    buggy_code: str
    failing_tests: List[str]
    patch: str
    reference_patch: str
    success: bool
    latency_seconds: float
    repair_messages: List[str]


@dataclass(slots=True)
class RepairBenchmarkResult:
    benchmark: str
    pass_rate: float
    pass_at_one: float
    average_latency: float
    samples: List[RepairSampleResult]


class ProgramRepairEvaluationRunner:
    """Runs the program repair coordinator across registered datasets."""

    def __init__(self, coordinator_factory: Callable[[], ProgramRepairCoordinator]) -> None:
        self._factory = coordinator_factory

    def run(self, benchmark_name: str, *, max_samples: Optional[int] = None) -> RepairBenchmarkResult:
        registry = dataset_registry()
        if benchmark_name not in registry:
            raise KeyError(f"Benchmark '{benchmark_name}' not registered.")
        benchmark = registry[benchmark_name]

        sample_results: List[RepairSampleResult] = []
        latencies: List[float] = []
        success_flags: List[bool] = []

        for idx, sample in enumerate(benchmark.samples):
            if max_samples is not None and idx >= max_samples:
                break
            if not sample.buggy_code:
                logger.warning("Sample %d in benchmark '%s' is missing buggy_code; skipping.", idx + 1, benchmark_name)
                continue
            coordinator = self._factory()
            failing_tests = sample.failing_tests or []
            logger.info(
                "Running repair sample %d/%d (benchmark=%s, tests=%d).",
                idx + 1,
                len(benchmark.samples),
                benchmark_name,
                len(failing_tests),
            )
            start = time.perf_counter()
            result: ProgramRepairResult = coordinator.repair(
                buggy_code=sample.buggy_code,
                description=sample.description,
                failing_tests=failing_tests,
            )
            latency = time.perf_counter() - start
            latencies.append(latency)
            success = self._compare_patch(result.patch, sample.reference)
            success_flags.append(success)
            sample_results.append(
                RepairSampleResult(
                    prompt=sample.prompt,
                    buggy_code=sample.buggy_code,
                    failing_tests=failing_tests,
                    patch=result.patch,
                    reference_patch=sample.reference,
                    success=success,
                    latency_seconds=latency,
                    repair_messages=[message.title for message in result.repair_messages],
                )
            )
            logger.info(
                "Sample %d completed: success=%s latency=%.3fs patch_preview=%.60s",
                idx + 1,
                success,
                latency,
                result.patch.replace("\n", " ")[:60],
            )

        if sample_results:
            pass_rate = float(sum(1 for res in sample_results if res.success) / len(sample_results))
            pass_at_one = pass_at_k([[res.success] for res in sample_results], k=1)
            avg_latency = float(sum(latencies) / len(latencies))
        else:
            pass_rate = 0.0
            pass_at_one = 0.0
            avg_latency = 0.0

        logger.info(
            "Program repair benchmark '%s' complete: pass_rate=%.3f pass@1=%.3f avg_latency=%.3fs",
            benchmark_name,
            pass_rate,
            pass_at_one,
            avg_latency,
        )

        return RepairBenchmarkResult(
            benchmark=benchmark.name,
            pass_rate=pass_rate,
            pass_at_one=pass_at_one,
            average_latency=avg_latency,
            samples=sample_results,
        )

    def _compare_patch(self, patch: str, reference: str) -> bool:
        normalized_patch = self._normalize(patch)
        normalized_reference = self._normalize(reference)
        return normalized_patch == normalized_reference

    @staticmethod
    def _normalize(text: str) -> str:
        return "\n".join(line.rstrip() for line in text.strip().splitlines())
