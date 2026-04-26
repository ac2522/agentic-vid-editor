"""Evaluation datasets for context optimization."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalItem:
    """Single evaluation item — a task + expected behavior."""

    id: str
    task: str
    expected_tools: list[str]
    expected_output_pattern: str | None
    context: dict[str, Any]


@dataclass
class EvalDataset:
    """Collection of evaluation items."""

    name: str
    items: list[EvalItem]

    @classmethod
    def from_jsonl(cls, path: Path) -> EvalDataset:
        """Load dataset from JSONL file."""
        items: list[EvalItem] = []
        name = path.stem
        if not path.exists():
            raise FileNotFoundError(f"Eval dataset not found: {path}")
        text = path.read_text().strip()
        if not text:
            return cls(name=name, items=[])
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(
                EvalItem(
                    id=data["id"],
                    task=data["task"],
                    expected_tools=data.get("expected_tools", []),
                    expected_output_pattern=data.get("expected_output_pattern"),
                    context=data.get("context", {}),
                )
            )
        return cls(name=name, items=items)

    def split(
        self, train_ratio: float = 0.8, seed: int | None = None
    ) -> tuple[EvalDataset, EvalDataset]:
        """Split into train/validation datasets.

        Args:
            train_ratio: Fraction of items for training (0.0-1.0).
            seed: If provided, shuffle items deterministically before splitting.
                  If None, split by position (no shuffle).
        """
        items = list(self.items)
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(items)
        split_idx = int(len(items) * train_ratio)
        return (
            EvalDataset(name=f"{self.name}_train", items=items[:split_idx]),
            EvalDataset(name=f"{self.name}_val", items=items[split_idx:]),
        )
