"""Expanding-window out-of-sample backtesting engine.

Implements temporal-split backtesting where the training set grows with each
successive fold (no shuffling, no look-ahead leakage).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator, Iterable

import numpy as np

from src.evaluation.metrics import EvaluationResult, evaluate

logger = logging.getLogger(__name__)


@dataclass
class BacktestFold:
    """A single expanding-window fold."""

    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class BacktestResult:
    """Aggregated results across all folds for one category."""

    category: str
    fold_results: list[EvaluationResult] = field(default_factory=list)

    @property
    def mean_brier(self) -> float:
        return float(np.mean([r.brier_score for r in self.fold_results]))

    @property
    def mean_ece(self) -> float:
        return float(np.mean([r.ece for r in self.fold_results]))

    @property
    def mean_roc_auc(self) -> float:
        return float(np.mean([r.roc_auc for r in self.fold_results]))

    @property
    def mean_log_loss(self) -> float:
        return float(np.mean([r.log_loss for r in self.fold_results]))


def expanding_window_folds(
    n: int,
    min_train_size: int,
    step: int = 1,
) -> Generator[BacktestFold, None, None]:
    """Yield expanding-window folds over *n* temporally ordered samples.

    Parameters
    ----------
    n:
        Total number of samples.
    min_train_size:
        Minimum number of training samples for the first fold.
    step:
        Number of samples to advance the test window each fold.

    Yields
    ------
    BacktestFold
        Fold with ``train_indices`` and ``test_indices``.
    """
    if min_train_size >= n:
        raise ValueError(
            f"min_train_size ({min_train_size}) must be < n ({n})."
        )

    fold_idx = 0
    train_end = min_train_size
    while train_end < n:
        test_end = min(train_end + step, n)
        yield BacktestFold(
            fold_index=fold_idx,
            train_indices=np.arange(train_end),
            test_indices=np.arange(train_end, test_end),
        )
        train_end = test_end
        fold_idx += 1


class BacktestEngine:
    """Runs expanding-window OOS backtesting for a single event category.

    Parameters
    ----------
    category:
        Event category to backtest.
    min_train_size:
        Minimum number of events required before the first OOS evaluation.
    step:
        How many events to advance the test window each iteration.
    """

    def __init__(
        self,
        category: str,
        min_train_size: int = 30,
        step: int = 1,
    ) -> None:
        self.category = category
        self.min_train_size = min_train_size
        self.step = step

    def run(
        self,
        features: np.ndarray,
        outcomes: np.ndarray,
    ) -> BacktestResult:
        """Execute the backtest.

        Parameters
        ----------
        features:
            Feature matrix of shape (n_events, n_features), ordered chronologically.
        outcomes:
            Binary outcome array of shape (n_events,).

        Returns
        -------
        BacktestResult
            Fold-level and aggregate metrics.
        """
        from src.models.classifier import CategoryClassifier
        from src.models.calibration import IsotonicCalibrator

        n = len(outcomes)
        result = BacktestResult(category=self.category)

        for fold in expanding_window_folds(n, self.min_train_size, self.step):
            X_train = features[fold.train_indices]
            y_train = outcomes[fold.train_indices]
            X_test = features[fold.test_indices]
            y_test = outcomes[fold.test_indices]

            # Split train into fit / calibration halves (chronological)
            split = len(X_train) // 2 or 1
            X_fit, y_fit = X_train[:split], y_train[:split]
            X_cal, y_cal = X_train[split:], y_train[split:]

            clf = CategoryClassifier(self.category)
            clf.fit(X_fit, y_fit)

            raw_probs_cal = clf.predict_proba(X_cal)
            cal = IsotonicCalibrator(self.category)
            cal.fit(raw_probs_cal, y_cal)

            raw_probs_test = clf.predict_proba(X_test)
            probs_test = cal.calibrate(raw_probs_test)

            fold_eval = evaluate(self.category, probs_test, y_test)
            result.fold_results.append(fold_eval)
            logger.debug(
                "Fold %d | Brier=%.4f ECE=%.4f",
                fold.fold_index,
                fold_eval.brier_score,
                fold_eval.ece,
            )

        logger.info(
            "Backtest complete for %s: mean_brier=%.4f mean_ece=%.4f over %d folds.",
            self.category,
            result.mean_brier,
            result.mean_ece,
            len(result.fold_results),
        )
        return result
