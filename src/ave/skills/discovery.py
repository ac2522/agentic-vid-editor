"""Skill frontmatter parsing and directory discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class SkillMeta:
    """Frontmatter metadata — loaded at startup. Body loaded on demand."""

    name: str
    description: str
    domain: str
    triggers: tuple[str, ...]
    path: Path


def parse_skill_frontmatter(skill_path: Path) -> SkillMeta:
    """Parse YAML frontmatter from a skill markdown file."""
    text = skill_path.read_text()
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"No YAML frontmatter found in {skill_path}")

    data: dict[str, Any] = yaml.safe_load(match.group(1))
    return SkillMeta(
        name=data["name"],
        description=data["description"],
        domain=data.get("domain", "general"),
        triggers=tuple(data.get("triggers", [])),
        path=skill_path,
    )


def discover_skills(search_dirs: list[Path]) -> list[SkillMeta]:
    """Scan directories for .md skill files. Earlier dirs = higher priority."""
    seen_names: dict[str, SkillMeta] = {}

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for md_file in sorted(search_dir.glob("*.md")):
            try:
                meta = parse_skill_frontmatter(md_file)
            except (ValueError, yaml.YAMLError, KeyError):
                continue
            if meta.name not in seen_names:
                seen_names[meta.name] = meta

    return list(seen_names.values())
