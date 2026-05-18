"""CLI entry point for running backtests.

Usage
-----
    python src/backtesting/run_backtest.py --category all --window expanding
    python src/backtesting/run_backtest.py --category federal_legislation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.backtesting.engine import BacktestEngine
from src.models.classifier import CATEGORY_FEATURES

logger = logging.getLogger(__name__)

ALL_CATEGORIES = list(CATEGORY_FEATURES.keys())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OOS backtesting.")
    parser.add_argument(
        "--category",
        default="all",
        choices=["all"] + ALL_CATEGORIES,
        help="Event category to backtest, or 'all' for every category.",
    )
    parser.add_argument(
        "--window",
        default="expanding",
        choices=["expanding"],
        help="Backtesting window strategy (currently only 'expanding').",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=30,
        help="Minimum training-set size before first OOS evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/"),
        help="Directory to write backtest result JSON files.",
    )
    return parser.parse_args()


def run_category(
    category: str,
    min_train: int,
    output_dir: Path,
) -> None:
    """Load data, run backtest, and write results for *category*."""
    import numpy as np

    # TODO: replace with real data loading from the event store
    logger.warning(
        "No real data available for category=%s; using random placeholder data.",
        category,
    )
    rng = np.random.default_rng(0)
    n = 200
    n_features = len(CATEGORY_FEATURES[category])
    features = rng.standard_normal((n, n_features))
    outcomes = rng.integers(0, 2, size=n).astype(float)

    engine = BacktestEngine(category, min_train_size=min_train)
    result = engine.run(features, outcomes)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "category": category,
        "n_folds": len(result.fold_results),
        "mean_brier": round(result.mean_brier, 4),
        "mean_ece": round(result.mean_ece, 4),
        "mean_roc_auc": round(result.mean_roc_auc, 4),
        "mean_log_loss": round(result.mean_log_loss, 4),
    }
    out_path = output_dir / f"backtest_{category}.json"
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Wrote backtest results to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    categories = ALL_CATEGORIES if args.category == "all" else [args.category]
    for cat in categories:
        run_category(cat, args.min_train, args.output)
