"""Utility functions for computing uncertainty and RAG metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import os

import numpy as np

from src.token_self_repair.uncertainty import LogTokUEstimator, UncertaintyAggregator, UncertaintyMap


@dataclass(slots=True)
class UncertaintyReport:
    map: UncertaintyMap
    avg_eu: float
    avg_au: float
    avg_logtoku: float
    avg_entropy: float


def compute_uncertainty(tokens: List[str], logits: np.ndarray, response_text: str) -> Optional[UncertaintyReport]:
    estimator = LogTokUEstimator()
    scores = estimator.analyze(logits, token_texts=tokens)
    if scores is None:
        return None

    aggregator = UncertaintyAggregator()
    u_map = aggregator.build_uncertainty_map(scores, source_text=response_text, language="text", tokens=tokens)
    return UncertaintyReport(
        map=u_map,
        avg_eu=scores.avg_eu,
        avg_au=scores.avg_au,
        avg_logtoku=scores.avg_total,
        avg_entropy=scores.avg_entropy,
    )


def compute_ragas(
    question: str,
    answer: str,
    contexts: List[str],
    reference: Optional[str] = None,
) -> Dict[str, float]:
    """Compute a subset of RAGAS metrics if the library is available."""
    if not os.getenv("OPENAI_API_KEY"):
        return {}

    contexts = [c for c in contexts if c.strip()]
    if not contexts:
        return {}

    try:
        from openai import OpenAI
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.llms import llm_factory
        from ragas.embeddings import embedding_factory
        from datasets import Dataset
    except ImportError:
        return {}

    data_dict = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    if reference:
        data_dict["reference"] = [reference]

    dataset = Dataset.from_dict(data_dict)

    metrics = [faithfulness, answer_relevancy]
    if reference:
        metrics.extend([context_precision, context_recall])

    try:
        client = OpenAI()
        llm = llm_factory("gpt-4o-mini", client=client)
        embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)
    except Exception:
        return {}

    try:
        result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    except Exception:
        return {}
    if hasattr(result, "metrics") and hasattr(result, "scores"):
        metrics_dict: Dict[str, float] = {}
        for metric_obj, score in zip(result.metrics, result.scores):
            name = getattr(metric_obj, "name", None) or getattr(metric_obj, "__name__", None) or str(metric_obj)
            metrics_dict[name] = float(score)
        return metrics_dict

    if isinstance(result, dict):
        return {k: float(v[0]) if isinstance(v, (list, tuple)) else float(v) for k, v in result.items()}

    return {}


