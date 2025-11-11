"""Metrics for evaluating calibration and downstream performance."""

from __future__ import annotations

import logging
from math import comb
from typing import Iterable, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC via trapezoidal integration."""
    order = np.argsort(scores)
    scores = scores[order]
    labels = labels[order]
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    tp = 0.0
    fp = 0.0
    prev_score = -np.inf
    auc = 0.0
    for score, label in zip(scores[::-1], labels[::-1]):
        if score != prev_score:
            auc += trapezoid_area(fp / neg, tp / pos, fp / neg, tp / pos)
            prev_score = score
        if label:
            tp += 1
        else:
            fp += 1
    auc += trapezoid_area(fp / neg, tp / pos, 1.0, 1.0)
    return min(1.0, max(0.0, auc))


def trapezoid_area(x1: float, y1: float, x2: float, y2: float) -> float:
    return (x2 - x1) * (y1 + y2) / 2.0


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Compute ECE by binning probabilities and averaging gaps."""
    bin_indices = np.minimum((probabilities * bins).astype(int), bins - 1)
    ece = 0.0
    for bin_id in range(bins):
        mask = bin_indices == bin_id
        if not np.any(mask):
            continue
        bin_probs = probabilities[mask]
        bin_labels = labels[mask]
        confidence = np.mean(bin_probs)
        accuracy = np.mean(bin_labels)
        ece += (len(bin_probs) / len(probabilities)) * abs(confidence - accuracy)
    return float(ece)


def exact_match(predictions, references) -> float:
    total = 0
    correct = 0
    for prediction, reference in zip(predictions, references):
        total += 1
        if normalize(prediction) == normalize(reference):
            correct += 1
    return correct / total if total else 0.0


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def pass_at_k(outcomes: Sequence[Sequence[bool]], k: int = 1) -> float:
    """
    Compute pass@k for code-generation style evaluations.

    Args:
        outcomes: Sequence where each entry contains booleans representing whether each
            attempt for a single task passed (True) or failed (False).
        k: Number of samples to consider per task when estimating success probability.

    Returns:
        Average probability that at least one of the k samples solves each task.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if not outcomes:
        return 0.0

    probabilities: list[float] = []
    for attempts in outcomes:
        n = len(attempts)
        c = sum(bool(a) for a in attempts)
        if n == 0 or c == 0:
            probabilities.append(0.0)
            continue
        k_eff = min(k, n)
        if k_eff == n:
            probabilities.append(1.0)
            continue
        total = comb(n, k_eff)
        fail = comb(n - c, k_eff) if n - c >= k_eff else 0
        probabilities.append(1.0 - fail / total)
    result = float(np.mean(probabilities))
    logger.info("pass@%d computed over %d task(s): %.3f", k, len(outcomes), result)
    return result


def latency_overhead(baseline: Iterable[float], augmented: Iterable[float]) -> float:
    """
    Measure the relative latency increase introduced by the uncertainty pipeline.

    Returns:
        Fractional overhead: (mean(augmented) - mean(baseline)) / mean(baseline).
        Negative values indicate speedups; zero means no change.
    """
    baseline = np.array(list(baseline), dtype=float)
    augmented = np.array(list(augmented), dtype=float)
    if baseline.size == 0 or augmented.size == 0:
        return 0.0
    base_mean = float(np.mean(baseline))
    aug_mean = float(np.mean(augmented))
    if base_mean == 0:
        return float("inf") if aug_mean > 0 else 0.0
    overhead = (aug_mean - base_mean) / base_mean
    logger.info(
        "Latency overhead computed (baseline_mean=%.3f, augmented_mean=%.3f): %.3f",
        base_mean,
        aug_mean,
        overhead,
    )
    return overhead


def user_trust_correlation(confidences: Sequence[float], trust_scores: Sequence[float]) -> float:
    """
    Compute Pearson correlation between model confidence and user trust ratings.

    Args:
        confidences: Model-reported confidences or 1 - uncertainty values.
        trust_scores: User survey ratings aligned with confidences.

    Returns:
        Correlation coefficient in [-1, 1]. Returns 0 if inputs are empty or constant.
    """
    if len(confidences) != len(trust_scores) or not confidences:
        return 0.0
    x = np.array(confidences, dtype=float)
    y = np.array(trust_scores, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    corr = float(np.nan_to_num(corr))
    logger.info("User trust correlation computed for %d sample(s): %.3f", len(confidences), corr)
    return corr


def calibration_curve(probabilities: Sequence[float], labels: Sequence[float], bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a calibration curve from probability/label pairs.

    Returns:
        Tuple of (bin_confidences, bin_accuracies) each shaped (bins,).
    """
    probs = np.asarray(probabilities, dtype=float)
    lbls = np.asarray(labels, dtype=float)
    if probs.size == 0 or lbls.size == 0 or probs.size != lbls.size:
        return np.zeros(bins), np.zeros(bins)
    probs = np.clip(probs, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_conf = np.zeros(bins)
    bin_acc = np.zeros(bins)
    for idx in range(bins):
        mask = (probs >= bin_edges[idx]) & (probs < bin_edges[idx + 1] if idx < bins - 1 else probs <= bin_edges[idx + 1])
        if not np.any(mask):
            continue
        bin_conf[idx] = float(np.mean(probs[mask]))
        bin_acc[idx] = float(np.mean(lbls[mask]))
    logger.info(
        "Calibration curve computed for %d sample(s) into %d bins.", len(probabilities), bins
    )
    return bin_conf, bin_acc
