"""Load harness scenarios from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from ave.harness.schema import Scenario


def load_scenario_from_yaml(path: Path) -> Scenario:
    """Parse a scenario YAML file into a Scenario model.

    Raises FileNotFoundError if path doesn't exist, yaml.YAMLError on bad
    YAML, and pydantic ValidationError on schema violations.
    """
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level dict in {path}, got {type(data).__name__}")
    return Scenario.model_validate(data)
