"""Tests for src/data/ingest.py"""
from pathlib import Path
import pytest
from src.data.ingest import fetch_source, ingest_all, SOURCES


def test_fetch_source_unknown_raises():
    with pytest.raises(ValueError, match="Unknown source"):
        fetch_source("nonexistent_source", Path("/tmp/staging"))


def test_fetch_source_returns_path(tmp_path):
    path = fetch_source(SOURCES[0], tmp_path)
    assert isinstance(path, Path)
    assert path.parent == tmp_path


def test_ingest_all_returns_list(tmp_path):
    paths = ingest_all(tmp_path)
    assert len(paths) == len(SOURCES)
