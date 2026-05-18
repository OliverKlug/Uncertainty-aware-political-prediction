"""Tests for src/models/calibration.py"""
import numpy as np
import pytest
from src.models.calibration import IsotonicCalibrator, apply_benjamini_hochberg


def _make_calibration_data(n: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    raw_probs = rng.uniform(0, 1, size=n)
    outcomes = (raw_probs + rng.normal(0, 0.2, size=n) > 0.5).astype(float)
    return raw_probs, outcomes


def test_calibrator_raises_before_fit():
    cal = IsotonicCalibrator("election")
    raw, _ = _make_calibration_data()
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.calibrate(raw)


def test_calibrator_output_in_unit_interval():
    raw, outcomes = _make_calibration_data()
    cal = IsotonicCalibrator("election")
    cal.fit(raw, outcomes)
    calibrated = cal.calibrate(raw)
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_calibrator_bootstrap_ci_shape():
    raw, outcomes = _make_calibration_data(n=60)
    cal = IsotonicCalibrator("election")
    cal.fit(raw, outcomes)
    lower, upper = cal.bootstrap_ci(raw, outcomes, n_bootstrap=50)
    assert lower.shape == raw.shape
    assert upper.shape == raw.shape
    assert np.all(lower <= upper)


def test_benjamini_hochberg_all_significant():
    p_values = np.array([0.001, 0.002, 0.003, 0.004])
    rejected = apply_benjamini_hochberg(p_values, fdr=0.05)
    assert rejected.all()


def test_benjamini_hochberg_none_significant():
    p_values = np.array([0.9, 0.8, 0.7, 0.6])
    rejected = apply_benjamini_hochberg(p_values, fdr=0.05)
    assert not rejected.any()
