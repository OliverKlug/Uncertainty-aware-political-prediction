"""Evaluation metrics.

Computes Brier Score, Expected Calibration Error (ECE), ROC-AUC, and
log-loss for probabilistic forecasts of political/macro events.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class EvaluationResult(NamedTuple):
    """Aggregated evaluation result for one category (or overall)."""

    category: str
    n_events: int
    brier_score: float
    ece: float
    roc_auc: float
    log_loss: float


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and outcomes.

    Lower is better. Perfect calibration → 0, random → 0.25.
    """
    return float(np.mean((probs - outcomes) ** 2))


def expected_calibration_error(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) using equal-width probability bins.

    Parameters
    ----------
    probs:
        Predicted probabilities in [0, 1].
    outcomes:
        Binary outcomes (0 or 1).
    n_bins:
        Number of probability bins.

    Returns
    -------
    float
        ECE ∈ [0, 1]; lower is better.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_prob = probs[mask].mean()
        bin_acc = outcomes[mask].mean()
        ece += mask.sum() / n * abs(bin_prob - bin_acc)
    return float(ece)


def roc_auc(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Area under the ROC curve (discrimination ability).

    Returns 0.5 if only one class is present (degenerate case).
    """
    from sklearn.metrics import roc_auc_score

    if len(np.unique(outcomes)) < 2:
        logger.warning("Only one class present; ROC-AUC is undefined, returning 0.5.")
        return 0.5
    return float(roc_auc_score(outcomes, probs))


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy / log-loss.

    Penalises confident wrong predictions heavily.
    """
    probs_c = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(outcomes * np.log(probs_c) + (1 - outcomes) * np.log(1 - probs_c)))


# ---------------------------------------------------------------------------
# Aggregated evaluation
# ---------------------------------------------------------------------------

def evaluate(
    category: str,
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> EvaluationResult:
    """Compute all metrics for *category* given *probs* and *outcomes*."""
    return EvaluationResult(
        category=category,
        n_events=len(outcomes),
        brier_score=brier_score(probs, outcomes),
        ece=expected_calibration_error(probs, outcomes, n_bins=n_bins),
        roc_auc=roc_auc(probs, outcomes),
        log_loss=log_loss(probs, outcomes),
    )
