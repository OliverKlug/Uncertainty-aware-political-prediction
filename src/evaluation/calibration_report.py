"""Calibration report generator.

Produces calibration curves and summary tables for all evaluated categories
and writes outputs to the results directory.

Usage
-----
    python src/evaluation/calibration_report.py --output results/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.evaluation.metrics import evaluate, expected_calibration_error

logger = logging.getLogger(__name__)


def calibration_curve(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a reliability diagram (calibration curve).

    Returns
    -------
    bin_centers, mean_probs, fraction_positive:
        Arrays of length ≤ n_bins describing each non-empty bin.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, mean_probs, frac_positive = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        mean_probs.append(probs[mask].mean())
        frac_positive.append(outcomes[mask].mean())

    return (
        np.array(bin_centers),
        np.array(mean_probs),
        np.array(frac_positive),
    )


def generate_report(
    category_results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    """Generate calibration curves and a JSON summary for all categories.

    Parameters
    ----------
    category_results:
        Mapping from category name to (probs, outcomes) arrays.
    output_dir:
        Directory where output artefacts are written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    for category, (probs, outcomes) in category_results.items():
        result = evaluate(category, probs, outcomes)
        _, mean_probs, frac_pos = calibration_curve(probs, outcomes)

        summary[category] = {
            "n_events": result.n_events,
            "brier_score": round(result.brier_score, 4),
            "ece": round(result.ece, 4),
            "roc_auc": round(result.roc_auc, 4),
            "log_loss": round(result.log_loss, 4),
            "calibration_curve": {
                "mean_predicted_prob": mean_probs.tolist(),
                "fraction_positive": frac_pos.tolist(),
            },
        }
        logger.info("Evaluated %s: Brier=%.4f ECE=%.4f", category, result.brier_score, result.ece)

    out_path = output_dir / "calibration_report.json"
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Report written to %s", out_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate calibration report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/"),
        help="Directory to write report artefacts (default: results/).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    # TODO: load real predictions from the results store
    logger.info("No predictions loaded; report will be empty.")
    generate_report({}, args.output)
