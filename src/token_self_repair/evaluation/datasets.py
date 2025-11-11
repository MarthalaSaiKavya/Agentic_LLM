"""Dataset registry for evaluation benchmarks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TaskSample:
    """Minimal representation of an evaluation sample."""

    prompt: str
    reference: str
    buggy_code: Optional[str] = None
    failing_tests: Optional[List[str]] = None
    description: Optional[str] = None


@dataclass(slots=True)
class Benchmark:
    """Collection of samples paired with a task-specific evaluator."""

    name: str
    samples: List[TaskSample]
    metric: Callable[[Iterable[str], Iterable[str]], float]


def gsm8k_samples() -> List[TaskSample]:
    return [
        TaskSample(prompt="If you have 3 apples and eat 1, how many remain?", reference="2"),
        TaskSample(
            prompt="A train travels 60 miles per hour for 2 hours. How far does it go?",
            reference="120 miles",
        ),
    ]


def humaneval_samples() -> List[TaskSample]:
    return [
        TaskSample(
            prompt="Write a Python function add(a, b) that returns their sum.",
            reference="def add(a, b):\n    return a + b",
        )
    ]


def truthfulqa_samples() -> List[TaskSample]:
    return [
        TaskSample(
            prompt="What is the capital of the United States?",
            reference="Washington, D.C.",
        ),
        TaskSample(
            prompt="Does the sun orbit the earth?",
            reference="No, the earth orbits the sun.",
        ),
    ]


def bioasq_samples() -> List[TaskSample]:
    return [
        TaskSample(
            prompt="Which vitamin deficiency causes scurvy?",
            reference="Vitamin C deficiency causes scurvy.",
        ),
        TaskSample(
            prompt="Name a hormone produced by the thyroid gland.",
            reference="Thyroxine.",
        ),
    ]


def repair_samples() -> List[TaskSample]:
    return [
        TaskSample(
            prompt=(
                "Fix the bug in this Python function that should return the product of numbers:\n"
                "def multiply(nums):\n"
                "    total = 0\n"
                "    for n in nums:\n"
                "        total += n\n"
                "    return total"
            ),
            reference=(
                "def multiply(nums):\n"
                "    total = 1\n"
                "    for n in nums:\n"
                "        total *= n\n"
                "    return total"
            ),
            buggy_code=(
                "def multiply(nums):\n"
                "    total = 0\n"
                "    for n in nums:\n"
                "        total += n\n"
                "    return total"
            ),
            failing_tests=["multiply([2,3,4]) should return 24"],
        )
    ]


def dataset_registry() -> Dict[str, Benchmark]:
    from .metrics import exact_match

    registry = {
        "gsm8k": Benchmark(name="GSM8K-lite", samples=gsm8k_samples(), metric=exact_match),
        "humaneval": Benchmark(name="HumanEval-lite", samples=humaneval_samples(), metric=exact_match),
        "truthfulqa": Benchmark(name="TruthfulQA-mini", samples=truthfulqa_samples(), metric=exact_match),
        "bioasq": Benchmark(name="BioASQ-mini", samples=bioasq_samples(), metric=exact_match),
        "repair": Benchmark(name="Repair-mini", samples=repair_samples(), metric=exact_match),
        "defects4j": Benchmark(name="Defects4J-mini", samples=defects4j_samples(), metric=exact_match),
        "gitbugs": Benchmark(name="GitBugs-mini", samples=gitbugs_samples(), metric=exact_match),
    }
    logger.info("Dataset registry initialized with benchmarks: %s", ", ".join(registry))
    return registry
def defects4j_samples() -> List[TaskSample]:
    return [
        TaskSample(
            prompt="Repair the following Java snippet so that division by zero is avoided.",
            reference="return numerator / denominator;",
            buggy_code=(
                "int divide(int numerator, int denominator) {\n"
                "    return numerator / denominator;\n"
                "}\n"
            ),
            failing_tests=["divide(5, 0) should throw IllegalArgumentException"],
            description="Add guard for denominator == 0 to match Defects4J Math bug pattern.",
        )
    ]


def gitbugs_samples() -> List[TaskSample]:
    return [
        TaskSample(
            prompt="Fix NullPointerException in this Java snippet extracted from GitBugs.",
            reference="if (listener != null) { listener.onComplete(); }",
            buggy_code=(
                "void complete(Listener listener) {\n"
                "    listener.onComplete();\n"
                "}\n"
            ),
            failing_tests=["complete(null) should not throw"],
            description="Add null guard before invoking listener.",
        )
    ]
