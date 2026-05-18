"""Tests for src/evaluation/metrics.py"""
import numpy as np
import pytest
from src.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    roc_auc,
    log_loss,
    evaluate,
)


def test_brier_score_perfect():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    outcomes = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(probs, outcomes) == pytest.approx(0.0)


def test_brier_score_worst():
    probs = np.array([0.0, 1.0])
    outcomes = np.array([1.0, 0.0])
    assert brier_score(probs, outcomes) == pytest.approx(1.0)


def test_ece_perfect_calibration():
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    outcomes = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    ece = expected_calibration_error(probs, outcomes, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_roc_auc_random():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 200)
    outcomes = rng.integers(0, 2, 200).astype(float)
    auc = roc_auc(probs, outcomes)
    assert 0.0 <= auc <= 1.0


def test_roc_auc_single_class_returns_half():
    probs = np.array([0.5, 0.6, 0.7])
    outcomes = np.array([1.0, 1.0, 1.0])
    assert roc_auc(probs, outcomes) == pytest.approx(0.5)


def test_log_loss_perfect():
    probs = np.array([1.0 - 1e-9, 1e-9])
    outcomes = np.array([1.0, 0.0])
    assert log_loss(probs, outcomes) < 1e-6


def test_evaluate_returns_correct_n_events():
    probs = np.array([0.4, 0.6, 0.7])
    outcomes = np.array([0.0, 1.0, 1.0])
    result = evaluate("election", probs, outcomes)
    assert result.n_events == 3
    assert result.category == "election"
