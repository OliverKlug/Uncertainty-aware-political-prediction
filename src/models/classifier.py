"""Category-specific event classifier.

Each of the 8 event categories has its own ``CategoryClassifier`` instance
trained on category-specific features.  The classifier outputs a raw
probability that is later calibrated before being exposed as a forecast.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature schemas per category
# ---------------------------------------------------------------------------

CATEGORY_FEATURES: dict[str, list[str]] = {
    "supreme_court_ruling": [
        "oral_argument_sentiment",
        "precedent_alignment",
        "cert_granted_unanimous",
        "lower_court_direction",
        "ideological_alignment_score",
    ],
    "federal_legislation": [
        "sponsor_count",
        "committee_markup",
        "bipartisan",
        "days_since_introduction",
        "chamber_majority_margin",
    ],
    "executive_action": [
        "prior_eo_count_term",
        "congress_opposition_index",
        "legal_challenge_likelihood",
    ],
    "election": [
        "incumbent_approval_rating",
        "generic_ballot_spread",
        "fundraising_ratio",
        "days_to_election",
        "polling_average",
    ],
    "geopolitical_shift": [
        "diplomatic_tension_index",
        "trade_volume_change",
        "ally_support_count",
    ],
    "central_bank_decision": [
        "current_rate",
        "inflation_delta_vs_target",
        "unemployment_delta",
        "market_implied_change",
        "forward_guidance_score",
    ],
    "macro_release": [
        "consensus_estimate",
        "prior_reading",
        "surprise_index_3m",
        "market_positioning_z",
    ],
    "regulatory_ruling": [
        "agency_prior_enforcement_rate",
        "complaint_severity_score",
        "political_appointee_alignment",
    ],
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class CategoryClassifier:
    """Logistic regression classifier for a single event category.

    Parameters
    ----------
    category:
        Event category this classifier is responsible for.
    """

    def __init__(self, category: str) -> None:
        self.category = category
        self.feature_names: list[str] = CATEGORY_FEATURES.get(category, [])
        self._model: Any = None  # set after fit()
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CategoryClassifier":
        """Fit the classifier on *X* (n_samples, n_features) and binary *y*.

        Uses sklearn LogisticRegression with L2 regularisation.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=500, C=1.0)),
            ]
        )
        pipe.fit(X, y)
        self._model = pipe
        self._is_fitted = True
        logger.info("Fitted classifier for category=%s on %d samples.", self.category, len(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted positive-class probabilities for *X*.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier is not fitted. Call fit() first.")
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        """Persist the fitted model to *path* using joblib."""
        import joblib

        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted classifier.")
        joblib.dump(self._model, path)
        logger.info("Saved classifier for category=%s to %s.", self.category, path)

    @classmethod
    def load(cls, category: str, path: str) -> "CategoryClassifier":
        """Load a persisted classifier from *path*."""
        import joblib

        obj = cls(category)
        obj._model = joblib.load(path)
        obj._is_fitted = True
        return obj


# ---------------------------------------------------------------------------
# Registry of all 8 classifiers
# ---------------------------------------------------------------------------

def build_classifier_registry() -> dict[str, CategoryClassifier]:
    """Instantiate (unfitted) classifiers for all 8 event categories."""
    return {cat: CategoryClassifier(cat) for cat in CATEGORY_FEATURES}
