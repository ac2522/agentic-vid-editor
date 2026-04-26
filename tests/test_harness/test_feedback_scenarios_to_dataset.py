"""Tests for scenario → opik EvalDataset conversion."""
from pathlib import Path
import json
import pytest
pytest.importorskip("yaml")

from ave.harness.schema import (
    Expected, Inputs, PlanExpected, Scenario, ScopeSpec, ToolsRequired,
)


def _scenario(id="t1", with_plan=True) -> Scenario:
    return Scenario(
        id=id, tiers=("plan",), prompt="do the thing",
        scope=ScopeSpec(allowed_agents=("editor",), forbidden_layers=("video",)),
        inputs=Inputs(),
        expected=Expected(
            plan=PlanExpected(
                tools_required=ToolsRequired(all_of=("a",), any_of=("b", "c")),
                tools_forbidden=("forbidden_x",),
                irrelevance_allowed=False,
            ) if with_plan else None
        ),
    )


def test_scenario_to_eval_item_basic():
    from ave.harness.feedback.scenarios_to_dataset import scenario_to_eval_item
    item = scenario_to_eval_item(_scenario("reel.test"))
    assert item.id == "reel.test"
    assert item.task == "do the thing"
    assert "a" in item.expected_tools
    assert "b" in item.expected_tools
    assert item.expected_output_pattern is None
    assert item.context["tools_forbidden"] == ["forbidden_x"]
    assert item.context["forbidden_layers"] == ["video"]
    assert item.context["irrelevance_allowed"] is False


def test_scenarios_to_dataset_skips_no_plan_scenarios():
    from ave.harness.feedback.scenarios_to_dataset import scenarios_to_dataset
    s_with = _scenario("with", with_plan=True)
    s_without = _scenario("without", with_plan=False)
    ds = scenarios_to_dataset([s_with, s_without], name="test")
    assert ds.name == "test"
    assert len(ds.items) == 1
    assert ds.items[0].id == "with"


def test_scenarios_to_dataset_empty_list_yields_empty_dataset():
    from ave.harness.feedback.scenarios_to_dataset import scenarios_to_dataset
    ds = scenarios_to_dataset([], name="empty")
    assert ds.items == []


def test_write_dataset_to_jsonl_roundtrip(tmp_path: Path):
    """Writing then reading via EvalDataset.from_jsonl yields the same items."""
    from ave.harness.feedback.scenarios_to_dataset import (
        scenarios_to_dataset, write_dataset_to_jsonl,
    )
    from ave.optimize.datasets import EvalDataset

    ds = scenarios_to_dataset([_scenario("a"), _scenario("b")], name="rt")
    out = tmp_path / "ds.jsonl"
    n = write_dataset_to_jsonl(ds, out)
    assert n == 2
    assert out.exists()

    loaded = EvalDataset.from_jsonl(out)
    assert {i.id for i in loaded.items} == {"a", "b"}


def test_write_dataset_jsonl_format_keys(tmp_path: Path):
    """Each line should be a JSON object with the EvalItem keys."""
    from ave.harness.feedback.scenarios_to_dataset import (
        scenarios_to_dataset, write_dataset_to_jsonl,
    )
    ds = scenarios_to_dataset([_scenario("z")], name="x")
    out = tmp_path / "ds.jsonl"
    write_dataset_to_jsonl(ds, out)
    line = out.read_text().strip().splitlines()[0]
    obj = json.loads(line)
    assert obj["id"] == "z"
    assert obj["task"] == "do the thing"
    assert "expected_tools" in obj
    assert "context" in obj


def test_export_bundled_scenarios(tmp_path: Path):
    """Smoke test: convert all bundled flagship scenarios."""
    from ave.harness.feedback.scenarios_to_dataset import (
        scenarios_to_dataset, write_dataset_to_jsonl,
    )
    from ave.harness.loader import load_scenario_from_yaml
    from ave.harness.pytest_plugin import bundled_scenarios_dir, discover_plan_scenarios

    paths = discover_plan_scenarios(bundled_scenarios_dir())
    scenarios = [load_scenario_from_yaml(Path(p)) for p in paths]
    ds = scenarios_to_dataset(scenarios, name="flagship")
    out = tmp_path / "flagship.jsonl"
    n = write_dataset_to_jsonl(ds, out)
    assert n == 3  # all three flagship scenarios have plan expectations
    assert out.exists()
