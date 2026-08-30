"""Staging-file helpers.

Live HTTP fetch is not implemented. ``fetch_source`` writes an empty JSON
payload so the daily pipeline can run without pretending it pulled a feed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SOURCES = [
    "congress_calendar",
    "scotus_docket",
    "executive_orders",
    "election_calendars",
    "central_bank_schedules",
    "macro_release_calendar",
    "geopolitical_feeds",
    "regulatory_dockets",
]

_EMPTY_PAYLOADS: dict[str, dict[str, list]] = {
    "congress_calendar": {"items": []},
    "scotus_docket": {"cases": []},
}


def fetch_source(source: str, staging_dir: Path) -> Path:
    """Write an empty staging file for *source*. Does not call any API."""
    if source not in SOURCES:
        raise ValueError(f"Unknown source: {source!r}. Choose from {SOURCES}.")

    staging_dir.mkdir(parents=True, exist_ok=True)
    out_path = staging_dir / f"{source}.json"
    payload = _EMPTY_PAYLOADS.get(source, {"items": []})
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    logger.warning(
        "No live fetch for source=%s; wrote empty staging file %s.",
        source,
        out_path,
    )
    return out_path


def ingest_all(staging_dir: Path) -> list[Path]:
    """Write empty staging files for every source identifier."""
    return [fetch_source(src, staging_dir) for src in SOURCES]


def load_staged(path: Path) -> Any:
    """Load a staged JSON file and return parsed content."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
