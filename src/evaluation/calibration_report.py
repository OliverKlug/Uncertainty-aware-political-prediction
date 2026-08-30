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
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.evaluation.metrics import _bin_mask, evaluate

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
    p = np.asarray(probs, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, mean_probs, frac_positive = [], [], []
    last_i = n_bins - 1

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = _bin_mask(p, float(lo), float(hi), last=i == last_i)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        mean_probs.append(p[mask].mean())
        frac_positive.append(o[mask].mean())

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
            "brier_score": result.brier_score,
            "ece": result.ece,
            "roc_auc": None if not np.isfinite(result.roc_auc) else result.roc_auc,
            "log_loss": result.log_loss,
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
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Write a report on RNG draws. Labelled synthetic; not market OOS.",
    )
    return parser.parse_args()


def _synthetic_category_results(rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n = 200
    probs = rng.uniform(0.0, 1.0, size=n)
    outcomes = (probs + rng.normal(0.0, 0.25, size=n) > 0.5).astype(np.float64)
    return {"synthetic_demo": (probs, outcomes)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if args.synthetic:
        rng = np.random.default_rng(0)
        payload = _synthetic_category_results(rng)
        generate_report(payload, args.output)
        note_path = args.output / "calibration_report.json"
        data = json.loads(note_path.read_text(encoding="utf-8"))
        data["_meta"] = {"data": "synthetic", "rng_seed": 0}
        note_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        logger.info("No event store; writing empty report. Pass --synthetic for RNG demo.")
        generate_report({}, args.output)
