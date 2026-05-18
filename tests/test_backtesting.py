"""Tests for src/backtesting/engine.py"""
import numpy as np
import pytest
from src.backtesting.engine import expanding_window_folds, BacktestEngine


def test_expanding_window_folds_count():
    folds = list(expanding_window_folds(n=10, min_train_size=5, step=1))
    assert len(folds) == 5  # folds at train_end=5,6,7,8,9


def test_expanding_window_folds_no_leakage():
    for fold in expanding_window_folds(n=20, min_train_size=5, step=2):
        assert fold.train_indices.max() < fold.test_indices.min()


def test_expanding_window_raises_bad_min_train():
    with pytest.raises(ValueError):
        list(expanding_window_folds(n=5, min_train_size=10))


def test_backtest_engine_runs(monkeypatch):
    """Integration smoke test: BacktestEngine completes without error."""
    cat = "macro_release"
    rng = np.random.default_rng(7)
    from src.models.classifier import CATEGORY_FEATURES
    n = 80
    n_features = len(CATEGORY_FEATURES[cat])
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, n).astype(float)

    engine = BacktestEngine(cat, min_train_size=20, step=5)
    result = engine.run(X, y)
    assert len(result.fold_results) > 0
    assert 0.0 <= result.mean_brier <= 1.0
