"""Plugin manifest parsing and directory discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ToolStub:
    """Minimal tool info from plugin manifest (not the full implementation)."""

    name: str
    summary: str


@dataclass(frozen=True)
class PluginManifest:
    """Parsed plugin.yaml — loaded at startup, code loaded on demand."""

    name: str
    description: str
    version: str
    domain: str
    tools: tuple[ToolStub, ...]
    requires_python: tuple[str, ...] = ()
    requires_system: tuple[str, ...] = ()
    path: Path | None = None


_REQUIRED_FIELDS = ("name", "description", "version", "domain")


def parse_manifest(manifest_path: Path) -> PluginManifest:
    """Parse a plugin.yaml file into a PluginManifest."""
    with open(manifest_path) as f:
        data: dict[str, Any] = yaml.safe_load(f)

    for field_name in _REQUIRED_FIELDS:
        if field_name not in data:
            raise ValueError(f"plugin.yaml missing required field: {field_name}")

    tools_raw = data.get("tools", [])
    tools = tuple(ToolStub(name=t["name"], summary=t.get("summary", "")) for t in tools_raw)

    requires = data.get("requires", {})
    return PluginManifest(
        name=data["name"],
        description=data["description"],
        version=data["version"],
        domain=data["domain"],
        tools=tools,
        requires_python=tuple(requires.get("python", [])),
        requires_system=tuple(requires.get("system", [])),
        path=manifest_path.parent,
    )


def discover_plugins(search_dirs: list[Path]) -> list[PluginManifest]:
    """Scan directories for plugin.yaml files. Earlier dirs = higher priority."""
    seen_names: dict[str, PluginManifest] = {}

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for child in sorted(search_dir.iterdir()):
            if not child.is_dir():
                continue
            manifest_path = child / "plugin.yaml"
            if not manifest_path.exists():
                continue
            try:
                manifest = parse_manifest(manifest_path)
            except (ValueError, yaml.YAMLError):
                continue
            if manifest.name not in seen_names:
                seen_names[manifest.name] = manifest

    return list(seen_names.values())
