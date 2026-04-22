"""Load harness scenarios from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from ave.harness.schema import Scenario


def load_scenario_from_yaml(path: Path) -> Scenario:
    """Parse a scenario YAML file into a Scenario model.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    yaml.YAMLError
        If the YAML is syntactically invalid.
    ValueError
        If the YAML top-level is not a mapping.
    pydantic.ValidationError
        If the parsed data does not satisfy the Scenario schema.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level dict in {path}, got {type(data).__name__}")
    return Scenario.model_validate(data)
