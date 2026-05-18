"""Base rate library.

Constructs historical base rates per event category from the curated reference
dataset (1990–present).  These rates are used to anchor every forecast before
live signals are applied.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

_DEFAULT_LIBRARY_PATH = Path(__file__).parent.parent.parent / "data" / "base_rates.json"


class BaseRate(NamedTuple):
    """Historical base rate for one event category."""

    category: str
    positive_rate: float  # fraction of historical outcomes that were True
    n_events: int         # total events in the reference library
    min_events_threshold: int = 30  # below this the category is excluded

    @property
    def is_sufficient(self) -> bool:
        """Return True if the category has enough data to calibrate."""
        return self.n_events >= self.min_events_threshold


class BaseRateLibrary:
    """Manages per-category historical base rates.

    Parameters
    ----------
    path:
        JSON file containing the curated base rate data.
    """

    def __init__(self, path: Path = _DEFAULT_LIBRARY_PATH) -> None:
        self._rates: dict[str, BaseRate] = {}
        if path.exists():
            self._load(path)
        else:
            logger.warning(
                "Base rate library file not found at %s. "
                "Library will be empty until data is loaded.",
                path,
            )

    def _load(self, path: Path) -> None:
        with path.open() as fh:
            data: dict = json.load(fh)
        for category, entry in data.items():
            self._rates[category] = BaseRate(
                category=category,
                positive_rate=entry["positive_rate"],
                n_events=entry["n_events"],
            )
        logger.info("Loaded base rates for %d categories.", len(self._rates))

    def get(self, category: str) -> BaseRate:
        """Return the ``BaseRate`` for *category*.

        Raises
        ------
        KeyError
            If *category* is not present in the library.
        """
        if category not in self._rates:
            raise KeyError(f"No base rate found for category: {category!r}")
        return self._rates[category]

    def sufficient_categories(self) -> list[str]:
        """Return categories that meet the minimum event threshold."""
        return [r.category for r in self._rates.values() if r.is_sufficient]

    def prior(self, category: str) -> float:
        """Return the prior probability (positive rate) for *category*.

        Automatically returns 0.5 (uninformative) if data are insufficient.
        """
        try:
            rate = self.get(category)
        except KeyError:
            logger.warning(
                "Category %r not in library; using uninformative prior 0.5.",
                category,
            )
            return 0.5
        if not rate.is_sufficient:
            logger.warning(
                "Category %r has only %d events (threshold %d); "
                "using uninformative prior 0.5.",
                category,
                rate.n_events,
                rate.min_events_threshold,
            )
            return 0.5
        return rate.positive_rate

    def all_categories(self) -> list[str]:
        """Return all categories present in the library."""
        return list(self._rates.keys())
