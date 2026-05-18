"""Tests for src/data/base_rate_library.py"""
import json
import pytest
from src.data.base_rate_library import BaseRateLibrary, BaseRate


def test_base_rate_is_sufficient():
    rate = BaseRate(category="election", positive_rate=0.6, n_events=50)
    assert rate.is_sufficient is True


def test_base_rate_insufficient():
    rate = BaseRate(category="election", positive_rate=0.6, n_events=10)
    assert rate.is_sufficient is False


def test_library_prior_unknown_category_returns_half():
    lib = BaseRateLibrary(path=__import__("pathlib").Path("/nonexistent/path.json"))
    assert lib.prior("election") == 0.5


def test_library_loads_from_file(tmp_path):
    data = {
        "election": {"positive_rate": 0.65, "n_events": 100},
        "macro_release": {"positive_rate": 0.48, "n_events": 200},
    }
    p = tmp_path / "base_rates.json"
    p.write_text(json.dumps(data))
    lib = BaseRateLibrary(path=p)
    assert lib.prior("election") == pytest.approx(0.65)
    assert lib.prior("macro_release") == pytest.approx(0.48)


def test_library_get_missing_raises():
    lib = BaseRateLibrary(path=__import__("pathlib").Path("/nonexistent/path.json"))
    with pytest.raises(KeyError):
        lib.get("supreme_court_ruling")
