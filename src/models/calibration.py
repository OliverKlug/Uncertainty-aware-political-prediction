"""Calibration module.

Wraps raw classifier outputs with isotonic regression calibration fitted on a
temporal held-out split.  Bootstrap confidence intervals are computed on the
resulting calibration curve.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

N_BOOTSTRAP = 1_000
RANDOM_SEED = 42


class IsotonicCalibrator:
    """Isotonic regression calibrator with bootstrap CI.

    Parameters
    ----------
    category:
        Event category this calibrator is trained for (used for logging).
    """

    def __init__(self, category: str = "") -> None:
        self.category = category
        self._ir: "IsotonicRegression | None" = None  # type: ignore[name-defined]
        self._is_fitted = False

    def fit(
        self,
        raw_probs: np.ndarray,
        outcomes: np.ndarray,
    ) -> "IsotonicCalibrator":
        """Fit isotonic regression on calibration-split data.

        Parameters
        ----------
        raw_probs:
            Uncalibrated probabilities from the classifier.
        outcomes:
            True binary outcomes (0/1).

        Notes
        -----
        The calibration split **must** be temporally separated from the
        training split to avoid look-ahead leakage.
        """
        from sklearn.isotonic import IsotonicRegression

        raw = np.asarray(raw_probs, dtype=np.float64)
        y = np.asarray(outcomes, dtype=np.float64)
        if raw.ndim != 1 or y.ndim != 1 or raw.shape != y.shape:
            raise ValueError("raw_probs and outcomes must be 1-d and the same length")
        if raw.size == 0:
            raise ValueError("raw_probs must be non-empty")
        if not np.isfinite(raw).all() or not np.isfinite(y).all():
            raise ValueError("raw_probs and outcomes must be finite")

        self._ir = IsotonicRegression(out_of_bounds="clip")
        self._ir.fit(raw, y)
        self._is_fitted = True
        logger.info(
            "Fitted isotonic calibrator for category=%s on %d samples.",
            self.category,
            len(y),
        )
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Map *raw_probs* to calibrated probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Calibrator is not fitted. Call fit() first.")
        return self._ir.transform(raw_probs)  # type: ignore[union-attr]

    def bootstrap_ci(
        self,
        raw_probs: np.ndarray,
        outcomes: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap: int = N_BOOTSTRAP,
        rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bootstrap confidence intervals on the calibration curve.

        Parameters
        ----------
        raw_probs:
            Uncalibrated probabilities (calibration set).
        outcomes:
            True binary outcomes.
        alpha:
            Significance level (default 0.05 → 95 % CI).
        n_bootstrap:
            Number of bootstrap resamples.
        rng:
            NumPy random generator for reproducibility.

        Returns
        -------
        lower, upper:
            Arrays of shape (n_samples,) with lower and upper CI bounds for
            each input probability.
        """
        if rng is None:
            rng = np.random.default_rng(RANDOM_SEED)

        n = len(raw_probs)
        calibrated_boots = np.empty((n_bootstrap, n))

        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            from sklearn.isotonic import IsotonicRegression

            ir_boot = IsotonicRegression(out_of_bounds="clip")
            ir_boot.fit(raw_probs[idx], outcomes[idx])
            calibrated_boots[i] = ir_boot.transform(raw_probs)

        lower = np.percentile(calibrated_boots, 100 * alpha / 2, axis=0)
        upper = np.percentile(calibrated_boots, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

    def save(self, path: str) -> None:
        """Persist calibrator to *path* using joblib."""
        import joblib

        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted calibrator.")
        joblib.dump(self._ir, path)

    @classmethod
    def load(cls, category: str, path: str) -> "IsotonicCalibrator":
        """Load a persisted calibrator from *path*."""
        import joblib

        obj = cls(category)
        obj._ir = joblib.load(path)
        obj._is_fitted = True
        return obj


def apply_benjamini_hochberg(p_values: np.ndarray, fdr: float = 0.05) -> np.ndarray:
    """Return a boolean mask of rejected hypotheses under Benjamini-Hochberg.

    Applied when evaluating calibration across multiple categories
    simultaneously to control the false discovery rate.

    Parameters
    ----------
    p_values:
        Array of p-values, one per category.
    fdr:
        Desired false discovery rate (default 0.05).

    Returns
    -------
    np.ndarray of bool
        True where the null hypothesis is rejected.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_values must be a non-empty 1-d array")
    if not np.isfinite(p).all() or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p_values must be finite and in [0, 1]")
    if not (0.0 < fdr < 1.0):
        raise ValueError("fdr must be in (0, 1)")
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    # Step-up: largest k with p_(k) <= (k/n)*fdr, then reject 1..k.
    k_max = 0
    for k in range(1, n + 1):
        if sorted_p[k - 1] <= (k / n) * fdr:
            k_max = k
    rejected = np.zeros(n, dtype=bool)
    if k_max:
        rejected[order[:k_max]] = True
    return rejected
