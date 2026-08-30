"""Evaluation metrics.

Computes Brier Score, Expected Calibration Error (ECE), ROC-AUC, and
log-loss for probabilistic forecasts of political/macro events.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_PROBS_OUTCOMES_MSG = "probs and outcomes must be 1-d, same length, finite"


def _as_prob_outcome(
    probs: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p = np.asarray(probs, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)
    if p.ndim != 1 or o.ndim != 1 or p.shape != o.shape:
        raise ValueError(_PROBS_OUTCOMES_MSG)
    if p.size == 0:
        raise ValueError("probs and outcomes must be non-empty")
    if not np.isfinite(p).all() or not np.isfinite(o).all():
        raise ValueError(_PROBS_OUTCOMES_MSG)
    return p, o


def _bin_mask(
    probs: NDArray[np.float64],
    lo: float,
    hi: float,
    *,
    last: bool,
) -> NDArray[np.bool_]:
    """Equal-width bin; the last bin is closed on the right so p=1 is counted."""
    if last:
        return (probs >= lo) & (probs <= hi)
    return (probs >= lo) & (probs < hi)


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
    p, o = _as_prob_outcome(probs, outcomes)
    return float(np.mean((p - o) ** 2))


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
    p, o = _as_prob_outcome(probs, outcomes)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    last_i = n_bins - 1
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = _bin_mask(p, float(lo), float(hi), last=i == last_i)
        if mask.sum() == 0:
            continue
        bin_prob = p[mask].mean()
        bin_acc = o[mask].mean()
        ece += mask.sum() / n * abs(bin_prob - bin_acc)
    return float(ece)


def roc_auc(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Area under the ROC curve (discrimination ability).

    Returns NaN if only one class is present (undefined).
    """
    from sklearn.metrics import roc_auc_score

    p, o = _as_prob_outcome(probs, outcomes)
    if len(np.unique(o)) < 2:
        logger.warning("Only one class present; ROC-AUC is undefined.")
        return float("nan")
    return float(roc_auc_score(o, p))


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy / log-loss.

    Penalises confident wrong predictions heavily.
    """
    p, o = _as_prob_outcome(probs, outcomes)
    probs_c = np.clip(p, eps, 1 - eps)
    return float(-np.mean(o * np.log(probs_c) + (1 - o) * np.log(1 - probs_c)))


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
