"""Tests for src/models/classifier.py"""
import numpy as np
import pytest
from src.models.classifier import CategoryClassifier, CATEGORY_FEATURES, build_classifier_registry


def _make_data(category: str, n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_features = len(CATEGORY_FEATURES[category])
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, size=n).astype(float)
    return X, y


def test_classifier_raises_before_fit():
    clf = CategoryClassifier("election")
    X, _ = _make_data("election")
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_proba(X)


def test_classifier_predict_proba_shape():
    cat = "election"
    clf = CategoryClassifier(cat)
    X, y = _make_data(cat)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (len(X),)
    assert np.all((probs >= 0) & (probs <= 1))


def test_build_classifier_registry_has_all_categories():
    registry = build_classifier_registry()
    assert set(registry.keys()) == set(CATEGORY_FEATURES.keys())


def test_classifier_save_load(tmp_path):
    cat = "macro_release"
    clf = CategoryClassifier(cat)
    X, y = _make_data(cat)
    clf.fit(X, y)
    path = str(tmp_path / "clf.joblib")
    clf.save(path)
    loaded = CategoryClassifier.load(cat, path)
    np.testing.assert_allclose(clf.predict_proba(X), loaded.predict_proba(X))
