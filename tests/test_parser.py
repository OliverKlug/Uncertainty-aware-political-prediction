"""Tests for src/data/parser.py"""
import datetime
import pytest
from src.data.parser import parse, Event, EVENT_CATEGORIES


def test_event_rejects_unknown_category():
    with pytest.raises(ValueError, match="Unknown category"):
        Event(
            event_id="x",
            category="unknown",
            title="",
            description="",
            scheduled_date=datetime.date.today(),
        )


def test_parse_congress_calendar_empty():
    events = parse("congress_calendar", {"items": []})
    assert events == []


def test_parse_congress_calendar_single_item():
    raw = {
        "items": [
            {
                "id": "HR1",
                "title": "Test Bill",
                "summary": "A test bill",
                "date": "2025-09-15",
                "sponsor_count": 5,
                "committee_markup": True,
                "bipartisan": False,
            }
        ]
    }
    events = parse("congress_calendar", raw)
    assert len(events) == 1
    ev = events[0]
    assert ev.event_id == "HR1"
    assert ev.category == "federal_legislation"
    assert ev.features["sponsor_count"] == 5


def test_parse_scotus_docket_single_case():
    raw = {
        "cases": [
            {
                "docket_number": "23-456",
                "case_name": "Doe v. Smith",
                "question_presented": "Whether X.",
                "argument_date": "2025-10-01",
                "oral_argument_sentiment": 0.3,
                "precedent_alignment": 0.7,
                "cert_granted_unanimous": True,
            }
        ]
    }
    events = parse("scotus_docket", raw)
    assert len(events) == 1
    assert events[0].category == "supreme_court_ruling"
    assert events[0].features["cert_granted_unanimous"] is True


def test_parse_unknown_source_raises():
    with pytest.raises(NotImplementedError):
        parse("nonexistent_source", {})
