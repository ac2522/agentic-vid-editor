"""Tests for built-in eval dataset files."""

from __future__ import annotations

from pathlib import Path

import pytest

from ave.optimize.datasets import EvalDataset

DATA_DIR = Path(__file__).parent.parent.parent / "src" / "ave" / "optimize" / "data"


@pytest.mark.parametrize(
    "filename",
    [
        "tool_selection.jsonl",
        "role_routing.jsonl",
    ],
)
class TestBuiltinDatasets:
    def test_file_exists(self, filename: str):
        path = DATA_DIR / filename
        assert path.exists(), f"Built-in dataset missing: {path}"

    def test_parses_correctly(self, filename: str):
        path = DATA_DIR / filename
        ds = EvalDataset.from_jsonl(path)
        assert len(ds.items) > 0, f"Dataset {filename} has no items"

    def test_items_have_required_fields(self, filename: str):
        path = DATA_DIR / filename
        ds = EvalDataset.from_jsonl(path)
        for item in ds.items:
            assert item.id, f"Item missing id in {filename}"
            assert item.task, f"Item missing task in {filename}"
            assert isinstance(item.expected_tools, list)
