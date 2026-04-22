"""Tests for the Scenario -> Inspect AI Dataset adapter."""

import pytest

from tests.conftest import requires_inspect
from ave.harness.schema import (
    Expected,
    InputAsset,
    Inputs,
    PlanExpected,
    Scenario,
    ScopeSpec,
    ToolsRequired,
)


def _scenario() -> Scenario:
    return Scenario(
        id="reel.test",
        description="",
        tiers=("plan",),
        prompt="do a thing",
        scope=ScopeSpec(allowed_agents=("editor",), forbidden_layers=()),
        inputs=Inputs(assets=(InputAsset(id="c", ref="fixture://x.mp4"),)),
        expected=Expected(
            plan=PlanExpected(
                tools_required=ToolsRequired(all_of=("trim",), any_of=()),
                tools_forbidden=("apply_blend_mode",),
                irrelevance_allowed=False,
            )
        ),
    )


@requires_inspect
def test_scenarios_to_dataset_shape():
    from ave.harness.adapter import scenarios_to_dataset

    dataset = scenarios_to_dataset([_scenario()])
    samples = list(dataset)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.id == "reel.test"
    assert sample.input == "do a thing"
    # Metadata carries the full scenario for downstream solvers/scorers.
    assert "scenario" in sample.metadata
    assert sample.metadata["scenario"].id == "reel.test"


@requires_inspect
def test_empty_scenarios_yields_empty_dataset():
    from ave.harness.adapter import scenarios_to_dataset

    dataset = scenarios_to_dataset([])
    assert len(list(dataset)) == 0
