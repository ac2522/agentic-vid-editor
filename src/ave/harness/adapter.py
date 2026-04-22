"""Adapt harness Scenarios to Inspect AI Datasets.

Each Scenario becomes one Inspect AI Sample:
- ``input``: the colloquial user prompt
- ``target``: the scenario id (used only for identification; real scoring
  uses metadata)
- ``metadata``: {"scenario": Scenario} — the full Scenario model, so solvers
  and scorers can read any field they need.
"""

from __future__ import annotations

from typing import Sequence

from inspect_ai.dataset import MemoryDataset, Sample

from ave.harness.schema import Scenario


def scenarios_to_dataset(scenarios: Sequence[Scenario]) -> MemoryDataset:
    """Build an Inspect AI MemoryDataset from a list of Scenarios."""
    samples = [
        Sample(
            id=s.id,
            input=s.prompt,
            target=s.id,
            metadata={"scenario": s},
        )
        for s in scenarios
    ]
    return MemoryDataset(samples)
