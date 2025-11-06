"""Metrics for evaluating calibration and downstream performance."""

from __future__ import annotations

import numpy as np


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
