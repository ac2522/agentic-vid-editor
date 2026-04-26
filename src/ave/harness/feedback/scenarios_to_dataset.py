"""Convert harness Scenarios into opik-compatible EvalDataset/JSONL.

The optimize/ module evaluates LLM context against an EvalDataset; this
adapter lets the harness's flagship scenarios feed that pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

from ave.harness.schema import Scenario
from ave.optimize.datasets import EvalDataset, EvalItem


logger = logging.getLogger(__name__)


def scenario_to_eval_item(scenario: Scenario) -> EvalItem | None:
    """Map one Scenario to an EvalItem; returns None when no plan tier."""
    plan = scenario.expected.plan
    if plan is None:
        return None
    expected_tools = list(plan.tools_required.all_of) + list(plan.tools_required.any_of)
    return EvalItem(
        id=scenario.id,
        task=scenario.prompt,
        expected_tools=expected_tools,
        expected_output_pattern=None,
        context={
            "tools_forbidden": list(plan.tools_forbidden),
            "irrelevance_allowed": plan.irrelevance_allowed,
            "forbidden_layers": list(scenario.scope.forbidden_layers),
            "tiers": list(scenario.tiers),
        },
    )


def scenarios_to_dataset(
    scenarios: Sequence[Scenario],
    *, name: str = "harness",
) -> EvalDataset:
    """Build an EvalDataset from a list of Scenarios; skips scenarios with no plan tier."""
    items: list[EvalItem] = []
    for s in scenarios:
        item = scenario_to_eval_item(s)
        if item is None:
            logger.info("Skipping %s: no plan tier expectations", s.id)
            continue
        items.append(item)
    return EvalDataset(name=name, items=items)


def write_dataset_to_jsonl(dataset: EvalDataset, path: Path) -> int:
    """Write an EvalDataset to JSONL. Returns the number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in dataset.items:
            row = {
                "id": item.id,
                "task": item.task,
                "expected_tools": list(item.expected_tools),
                "expected_output_pattern": item.expected_output_pattern,
                "context": item.context,
            }
            f.write(json.dumps(row) + "\n")
    return len(dataset.items)
