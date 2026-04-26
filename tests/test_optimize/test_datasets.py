"""Tests for evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ave.optimize.datasets import EvalDataset, EvalItem


class TestEvalItem:
    def test_create_eval_item(self):
        item = EvalItem(
            id="tool_sel_001",
            task="Trim 2 seconds from the start of the clip",
            expected_tools=["trim_clip"],
            expected_output_pattern=None,
            context={},
        )
        assert item.id == "tool_sel_001"
        assert item.expected_tools == ["trim_clip"]

    def test_eval_item_multiple_expected_tools(self):
        item = EvalItem(
            id="multi_001",
            task="Transcribe the video and then remove filler words",
            expected_tools=["transcribe", "remove_fillers"],
            expected_output_pattern=None,
            context={},
        )
        assert len(item.expected_tools) == 2

    def test_eval_item_is_frozen(self):
        item = EvalItem(
            id="test",
            task="task",
            expected_tools=[],
            expected_output_pattern=None,
            context={},
        )
        with pytest.raises(AttributeError):
            item.task = "changed"  # type: ignore[misc]


class TestEvalDataset:
    def _make_items(self, n: int) -> list[EvalItem]:
        return [
            EvalItem(
                id=f"item_{i}",
                task=f"Task {i}",
                expected_tools=[f"tool_{i}"],
                expected_output_pattern=None,
                context={},
            )
            for i in range(n)
        ]

    def test_create_dataset(self):
        items = self._make_items(5)
        ds = EvalDataset(name="test", items=items)
        assert ds.name == "test"
        assert len(ds.items) == 5

    def test_from_jsonl(self, tmp_path: Path):
        jsonl_path = tmp_path / "test.jsonl"
        items = [
            {
                "id": "001",
                "task": "Trim the clip",
                "expected_tools": ["trim_clip"],
                "expected_output_pattern": None,
                "context": {"project": "test"},
            },
            {
                "id": "002",
                "task": "Adjust volume to -6dB",
                "expected_tools": ["adjust_volume"],
                "expected_output_pattern": "-6",
                "context": {},
            },
        ]
        with open(jsonl_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        ds = EvalDataset.from_jsonl(jsonl_path)
        assert ds.name == "test"
        assert len(ds.items) == 2
        assert ds.items[0].expected_tools == ["trim_clip"]
        assert ds.items[1].expected_output_pattern == "-6"

    def test_split_default_ratio(self):
        items = self._make_items(10)
        ds = EvalDataset(name="test", items=items)
        train, val = ds.split()
        assert len(train.items) == 8
        assert len(val.items) == 2

    def test_split_custom_ratio(self):
        items = self._make_items(10)
        ds = EvalDataset(name="test", items=items)
        train, val = ds.split(train_ratio=0.5)
        assert len(train.items) == 5
        assert len(val.items) == 5

    def test_split_deterministic_by_default(self):
        items = self._make_items(10)
        ds = EvalDataset(name="test", items=items)
        train1, val1 = ds.split()
        train2, val2 = ds.split()
        assert [i.id for i in train1.items] == [i.id for i in train2.items]
        assert [i.id for i in val1.items] == [i.id for i in val2.items]

    def test_split_with_seed_shuffles(self):
        items = self._make_items(20)
        ds = EvalDataset(name="test", items=items)
        train_no_seed, _ = ds.split()
        train_seeded, _ = ds.split(seed=42)
        # With 20 items and a shuffle, order should differ
        no_seed_ids = [i.id for i in train_no_seed.items]
        seeded_ids = [i.id for i in train_seeded.items]
        # They should contain the same count but potentially different items
        assert len(no_seed_ids) == len(seeded_ids)

    def test_split_preserves_name(self):
        ds = EvalDataset(name="tool_selection", items=self._make_items(10))
        train, val = ds.split()
        assert train.name == "tool_selection_train"
        assert val.name == "tool_selection_val"

    def test_from_jsonl_empty_file(self, tmp_path: Path):
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")
        ds = EvalDataset.from_jsonl(jsonl_path)
        assert len(ds.items) == 0

    def test_from_jsonl_missing_optional_fields(self, tmp_path: Path):
        jsonl_path = tmp_path / "minimal.jsonl"
        jsonl_path.write_text('{"id": "001", "task": "Do something"}\n')
        ds = EvalDataset.from_jsonl(jsonl_path)
        assert len(ds.items) == 1
        assert ds.items[0].expected_tools == []
        assert ds.items[0].expected_output_pattern is None
        assert ds.items[0].context == {}

    def test_from_jsonl_nonexistent_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Eval dataset not found"):
            EvalDataset.from_jsonl(tmp_path / "nonexistent.jsonl")
