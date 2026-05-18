"""Event parsing module.

Normalises raw ingested payloads into a shared ``Event`` schema, ready for
the base-rate library and category classifiers.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Shared event schema
# ---------------------------------------------------------------------------

EVENT_CATEGORIES = [
    "supreme_court_ruling",
    "federal_legislation",
    "executive_action",
    "election",
    "geopolitical_shift",
    "central_bank_decision",
    "macro_release",
    "regulatory_ruling",
]


@dataclass
class Event:
    """Canonical representation of a single political/macroeconomic event."""

    event_id: str
    category: str
    title: str
    description: str
    scheduled_date: datetime.date
    outcome: bool | None = None  # None if not yet resolved
    features: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    created_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    def __post_init__(self) -> None:
        if self.category not in EVENT_CATEGORIES:
            raise ValueError(
                f"Unknown category: {self.category!r}. "
                f"Choose from {EVENT_CATEGORIES}."
            )


# ---------------------------------------------------------------------------
# Parsers per source
# ---------------------------------------------------------------------------

def parse_congress_calendar(raw: dict[str, Any]) -> list[Event]:
    """Parse a Congress legislative calendar payload into Events."""
    events: list[Event] = []
    for item in raw.get("items", []):
        events.append(
            Event(
                event_id=item["id"],
                category="federal_legislation",
                title=item.get("title", ""),
                description=item.get("summary", ""),
                scheduled_date=datetime.date.fromisoformat(item["date"]),
                features={
                    "sponsor_count": item.get("sponsor_count", 0),
                    "committee_markup": item.get("committee_markup", False),
                    "bipartisan": item.get("bipartisan", False),
                },
                source="congress_calendar",
            )
        )
    return events


def parse_scotus_docket(raw: dict[str, Any]) -> list[Event]:
    """Parse a SCOTUS docket payload into Events."""
    events: list[Event] = []
    for item in raw.get("cases", []):
        events.append(
            Event(
                event_id=item["docket_number"],
                category="supreme_court_ruling",
                title=item.get("case_name", ""),
                description=item.get("question_presented", ""),
                scheduled_date=datetime.date.fromisoformat(
                    item.get("argument_date", datetime.date.today().isoformat())
                ),
                features={
                    "oral_argument_sentiment": item.get(
                        "oral_argument_sentiment", 0.0
                    ),
                    "precedent_alignment": item.get("precedent_alignment", 0.0),
                    "cert_granted_unanimous": item.get(
                        "cert_granted_unanimous", False
                    ),
                },
                source="scotus_docket",
            )
        )
    return events


# Registry mapping source name -> parser function
PARSERS: dict[str, Any] = {
    "congress_calendar": parse_congress_calendar,
    "scotus_docket": parse_scotus_docket,
    # TODO: add parsers for remaining sources
}


def parse(source: str, raw: dict[str, Any]) -> list[Event]:
    """Dispatch *raw* payload to the appropriate parser for *source*."""
    if source not in PARSERS:
        raise NotImplementedError(f"No parser implemented for source: {source!r}")
    return PARSERS[source](raw)
