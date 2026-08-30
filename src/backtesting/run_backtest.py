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
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Test-window length per fold (default 10; step=1 makes ROC undefined).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run on RNG data. Required until an event store exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for --synthetic (default: 0).",
    )
    return parser.parse_args()


def run_category(
    category: str,
    min_train: int,
    output_dir: Path,
    *,
    seed: int,
    step: int,
) -> None:
    """Run the expanding-window engine on synthetic draws for *category*."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = 200
    n_features = len(CATEGORY_FEATURES[category])
    features = rng.standard_normal((n, n_features))
    outcomes = rng.integers(0, 2, size=n).astype(np.float64)

    engine = BacktestEngine(category, min_train_size=min_train, step=step)
    result = engine.run(features, outcomes)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "data": "synthetic",
        "rng_seed": seed,
        "step": step,
        "category": category,
        "n_folds": len(result.fold_results),
        "mean_brier": result.mean_brier,
        "mean_ece": result.mean_ece,
        "mean_roc_auc": result.mean_roc_auc,
        "mean_log_loss": result.mean_log_loss,
    }
    out_path = output_dir / f"backtest_{category}.json"
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Wrote backtest results to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    if not args.synthetic:
        raise SystemExit(
            "No event store. Pass --synthetic to run the engine on RNG data."
        )
    categories = ALL_CATEGORIES if args.category == "all" else [args.category]
    for cat in categories:
        run_category(
            cat, args.min_train, args.output, seed=args.seed, step=args.step
        )
