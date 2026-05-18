"""Data ingestion module.

Pulls raw event data from configured sources (government portals, legislative
calendars, docket feeds) and writes them to a local staging area for parsing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Supported source identifiers
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


def fetch_source(source: str, staging_dir: Path) -> Path:
    """Fetch raw data for *source* and write to *staging_dir*.

    Parameters
    ----------
    source:
        One of the identifiers listed in ``SOURCES``.
    staging_dir:
        Directory where raw files are written.

    Returns
    -------
    Path
        Path to the written staging file.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source: {source!r}. Choose from {SOURCES}.")

    staging_dir.mkdir(parents=True, exist_ok=True)
    out_path = staging_dir / f"{source}.json"
    logger.info("Fetching source=%s -> %s", source, out_path)
    # TODO: implement per-source HTTP / scraping logic
    return out_path


def ingest_all(staging_dir: Path) -> list[Path]:
    """Ingest all sources and return paths to staged files."""
    return [fetch_source(src, staging_dir) for src in SOURCES]


def load_staged(path: Path) -> Any:
    """Load a staged JSON file and return parsed content."""
    import json

    with path.open() as fh:
        return json.load(fh)
