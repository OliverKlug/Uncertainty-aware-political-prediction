"""Daily pipeline orchestration.

Runs the full ingest → classify → calibrate → score → log cycle once per day.

Usage
-----
    python src/pipeline/run_daily.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

STAGING_DIR = Path("data/staging")
RESULTS_DIR = Path("results")
AUDIT_LOG = Path("results/audit_log.jsonl")


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def _sign_entry(entry: dict) -> str:
    """Return a SHA-256 hex digest of the JSON-serialised *entry*."""
    payload = json.dumps(entry, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def append_audit_entry(entry: dict) -> None:
    """Append a signed, append-only entry to the audit log."""
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry["_sha256"] = _sign_entry({k: v for k, v in entry.items() if k != "_sha256"})
    with AUDIT_LOG.open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_ingest() -> list[Path]:
    """Step 1 – ingest raw event data from all configured sources."""
    from src.data.ingest import ingest_all

    logger.info("[1/4] Ingesting data sources …")
    staged = ingest_all(STAGING_DIR)
    logger.info("Staged %d source files.", len(staged))
    return staged


def step_classify(staged_paths: list[Path]) -> list[dict]:
    """Step 2 – parse staged files and route events to category classifiers."""
    from src.data.ingest import load_staged
    from src.data.parser import parse, PARSERS
    from src.data.base_rate_library import BaseRateLibrary

    library = BaseRateLibrary()
    events_classified: list[dict] = []

    for path in staged_paths:
        source = path.stem
        if source not in PARSERS:
            logger.debug("No parser for source=%s, skipping.", source)
            continue
        try:
            raw = load_staged(path)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("Could not load staged file %s, skipping.", path)
            continue
        events = parse(source, raw)
        for event in events:
            prior = library.prior(event.category)
            events_classified.append(
                {
                    "event_id": event.event_id,
                    "category": event.category,
                    "prior": prior,
                    "features": event.features,
                    "scheduled_date": event.scheduled_date.isoformat(),
                }
            )

    logger.info("[2/4] Classified %d events.", len(events_classified))
    return events_classified


def step_score(classified_events: list[dict]) -> list[dict]:
    """Step 3 – produce calibrated probability for each classified event."""
    # TODO: load fitted classifiers + calibrators from model store and score
    scored: list[dict] = []
    for ev in classified_events:
        ev["calibrated_prob"] = ev["prior"]  # placeholder: use prior only
        ev["ci_lower"] = max(0.0, ev["prior"] - 0.1)
        ev["ci_upper"] = min(1.0, ev["prior"] + 0.1)
        ev["model_version"] = "prior_only_v0"
        scored.append(ev)
    logger.info("[3/4] Scored %d events.", len(scored))
    return scored


def step_log(scored_events: list[dict]) -> None:
    """Step 4 – write each prediction to the append-only audit log."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for ev in scored_events:
        entry = {
            **ev,
            "calibration_timestamp": ts,
        }
        append_audit_entry(entry)
    logger.info("[4/4] Logged %d predictions to audit trail.", len(scored_events))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_daily() -> None:
    """Execute the full daily pipeline."""
    logger.info("=== Daily pipeline started at %s ===", datetime.datetime.now(datetime.timezone.utc).isoformat())
    staged = step_ingest()
    classified = step_classify(staged)
    scored = step_score(classified)
    step_log(scored)
    logger.info("=== Daily pipeline complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_daily()
