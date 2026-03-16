"""Tests for skill frontmatter parsing and discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from ave.skills.discovery import SkillMeta, parse_skill_frontmatter, discover_skills


class TestSkillMeta:
    def test_parse_frontmatter(self, tmp_path):
        skill = tmp_path / "grain.md"
        skill.write_text(
            "---\n"
            "name: cinematic-grain\n"
            "description: Apply cinematic film grain\n"
            "domain: color\n"
            "triggers:\n"
            "  - film grain\n"
            "  - grain effect\n"
            "---\n"
            "\n## Steps\n1. Do the thing\n"
        )
        meta = parse_skill_frontmatter(skill)
        assert meta.name == "cinematic-grain"
        assert meta.description == "Apply cinematic film grain"
        assert meta.domain == "color"
        assert meta.triggers == ("film grain", "grain effect")
        assert meta.path == skill

    def test_parse_missing_frontmatter(self, tmp_path):
        skill = tmp_path / "bad.md"
        skill.write_text("# No frontmatter here\n")
        with pytest.raises(ValueError, match="frontmatter"):
            parse_skill_frontmatter(skill)

    def test_default_domain_is_general(self, tmp_path):
        skill = tmp_path / "nodomain.md"
        skill.write_text(
            "---\n"
            "name: nodomain\n"
            "description: No domain specified\n"
            "triggers: []\n"
            "---\nbody\n"
        )
        meta = parse_skill_frontmatter(skill)
        assert meta.domain == "general"

    def test_meta_is_frozen(self, tmp_path):
        skill = tmp_path / "frozen.md"
        skill.write_text(
            "---\nname: frozen\ndescription: test\n"
            "domain: editing\ntriggers: []\n---\nbody\n"
        )
        meta = parse_skill_frontmatter(skill)
        with pytest.raises(AttributeError):
            meta.name = "changed"  # type: ignore[misc]


class TestDiscoverSkills:
    def test_discover_skills_from_directory(self, tmp_path):
        (tmp_path / "skill1.md").write_text(
            "---\nname: s1\ndescription: first\n"
            "domain: editing\ntriggers: []\n---\nbody\n"
        )
        (tmp_path / "skill2.md").write_text(
            "---\nname: s2\ndescription: second\n"
            "domain: color\ntriggers: []\n---\nbody\n"
        )
        (tmp_path / "not-a-skill.txt").write_text("ignored")
        skills = discover_skills([tmp_path])
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"s1", "s2"}

    def test_discover_priority_order(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        for d in (dir_a, dir_b):
            (d / "same.md").write_text(
                f"---\nname: same\ndescription: from {d.name}\n"
                f"domain: editing\ntriggers: []\n---\nbody\n"
            )
        skills = discover_skills([dir_a, dir_b])
        assert len(skills) == 1
        assert skills[0].description == "from a"

    def test_discover_skips_nonexistent_dirs(self):
        skills = discover_skills([Path("/nonexistent")])
        assert len(skills) == 0

    def test_discover_skips_invalid_frontmatter(self, tmp_path):
        (tmp_path / "bad.md").write_text("no frontmatter")
        (tmp_path / "good.md").write_text(
            "---\nname: good\ndescription: works\n"
            "domain: editing\ntriggers: []\n---\nbody\n"
        )
        skills = discover_skills([tmp_path])
        assert len(skills) == 1
        assert skills[0].name == "good"
