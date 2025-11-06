"""Dataset registry for evaluation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List


@dataclass(slots=True)
class TaskSample:
    """Minimal representation of an evaluation sample."""

    prompt: str
    reference: str


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
            prompt="A train travels 60 miles per hour for 2 hours. How far does it go?", reference="120 miles"
        ),
    ]


def humaneval_samples() -> List[TaskSample]:
    return [
        TaskSample(
            prompt="Write a Python function add(a, b) that returns their sum.",
            reference="def add(a, b):\n    return a + b",
        )
    ]


def dataset_registry() -> Dict[str, Benchmark]:
    from .metrics import exact_match

    return {
        "gsm8k": Benchmark(name="GSM8K-lite", samples=gsm8k_samples(), metric=exact_match),
        "humaneval": Benchmark(name="HumanEval-lite", samples=humaneval_samples(), metric=exact_match),
    }
