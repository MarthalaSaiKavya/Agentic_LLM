"""Utility functions for computing uncertainty and RAG metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import os

import numpy as np

from src.token_self_repair.uncertainty import LogTokUEstimator, UncertaintyAggregator, UncertaintyMap

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class UncertaintyReport:
    map: UncertaintyMap
    avg_eu: float
    avg_au: float
    avg_logtoku: float
    avg_entropy: float


def compute_uncertainty(tokens: List[str], logits: np.ndarray, response_text: str) -> Optional[UncertaintyReport]:
    logger.info(
        "Computing uncertainty for %d tokens with logits shape %s", len(tokens), getattr(logits, "shape", None)
    )
    estimator = LogTokUEstimator()
    scores = estimator.analyze(logits, token_texts=tokens)
    if scores is None:
        logger.warning("Uncertainty computation returned no scores.")
        return None

    aggregator = UncertaintyAggregator()
    u_map = aggregator.build_uncertainty_map(scores, source_text=response_text, language="text", tokens=tokens)
    logger.debug(
        "Uncertainty report avg_eu=%.4f avg_au=%.4f avg_logtoku=%.4f",
        scores.avg_eu,
        scores.avg_au,
        scores.avg_total,
    )
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
        logger.info("Skipping RAGAS metrics because OPENAI_API_KEY is not set.")
        return {}

    contexts = [c for c in contexts if c.strip()]
    if not contexts:
        logger.info("Skipping RAGAS metrics because no non-empty contexts were provided.")
        return {}

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        logger.warning("RAGAS dependencies missing; install ragas[openai] and langchain-openai to enable metrics.")
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

    ragas_model = os.getenv("RAGAS_OPENAI_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")

    try:
        llm = ChatOpenAI(
            model=ragas_model,
            temperature=0.0,
            max_tokens=512,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    except Exception:
        logger.exception("Failed to initialize LangChain OpenAI clients for RAGAS evaluation.")
        return {}

    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
        )
    except Exception:
        logger.exception("RAGAS evaluation failed for question '%s'.", question[:80])
        return {}

    evaluation_errors = getattr(result, "errors", None)
    if evaluation_errors:
        logger.warning(
            "RAGAS evaluation reported %d internal error(s); first error: %s",
            len(evaluation_errors) if hasattr(evaluation_errors, "__len__") else 1,
            evaluation_errors[0] if isinstance(evaluation_errors, list) and evaluation_errors else evaluation_errors,
        )
    if hasattr(result, "metrics") and hasattr(result, "scores"):
        metrics_dict: Dict[str, float] = {}
        for metric_obj, score in zip(result.metrics, result.scores):
            name = getattr(metric_obj, "name", None) or getattr(metric_obj, "__name__", None) or str(metric_obj)
            metrics_dict[name] = float(score)
        logger.info("RAGAS metrics computed: %s", metrics_dict)
        return metrics_dict

    if isinstance(result, dict):
        metrics_dict = {k: float(v[0]) if isinstance(v, (list, tuple)) else float(v) for k, v in result.items()}
        logger.info("RAGAS metrics computed (dict response): %s", metrics_dict)
        return metrics_dict

    return {}
