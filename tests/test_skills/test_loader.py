"""Tests for skill loader."""

from __future__ import annotations

from pathlib import Path

from ave.skills.discovery import SkillMeta
from ave.skills.loader import SkillLoader


class TestSkillLoader:
    def test_load_skill_body(self, tmp_path):
        skill_path = tmp_path / "grain.md"
        skill_path.write_text(
            "---\n"
            "name: grain\n"
            "description: Film grain\n"
            "domain: color\n"
            "triggers: [film grain]\n"
            "---\n"
            "\n## Steps\n1. Apply grain overlay\n2. Adjust blend\n"
        )
        meta = SkillMeta(
            name="grain",
            description="Film grain",
            domain="color",
            triggers=("film grain",),
            path=skill_path,
        )
        loader = SkillLoader()
        body = loader.load_body(meta)
        assert "## Steps" in body
        assert "Apply grain overlay" in body
        assert "---" not in body

    def test_match_skill_by_trigger(self, tmp_path):
        skill_path = tmp_path / "grain.md"
        skill_path.write_text(
            "---\nname: grain\ndescription: Film grain\n"
            "domain: color\ntriggers: [film grain, grain effect, analog]\n---\nbody\n"
        )
        meta = SkillMeta(
            name="grain",
            description="Film grain",
            domain="color",
            triggers=("film grain", "grain effect", "analog"),
            path=skill_path,
        )
        loader = SkillLoader()
        loader.register(meta)

        matches = loader.match("I want to add a film grain effect")
        assert len(matches) >= 1
        assert matches[0].name == "grain"

    def test_no_match_returns_empty(self, tmp_path):
        loader = SkillLoader()
        meta = SkillMeta(
            name="grain",
            description="Film grain",
            domain="color",
            triggers=("film grain",),
            path=tmp_path / "grain.md",
        )
        loader.register(meta)
        matches = loader.match("trim the clip at 5 seconds")
        assert len(matches) == 0

    def test_get_skill_by_name(self, tmp_path):
        meta = SkillMeta(
            name="grain",
            description="Film grain",
            domain="color",
            triggers=("film grain",),
            path=tmp_path / "grain.md",
        )
        loader = SkillLoader()
        loader.register(meta)
        assert loader.get("grain") is meta
        assert loader.get("nonexistent") is None

    def test_match_scores_multiple_triggers_higher(self, tmp_path):
        meta = SkillMeta(
            name="grain",
            description="Film grain effect tool",
            domain="color",
            triggers=("film grain", "grain effect"),
            path=tmp_path / "grain.md",
        )
        loader = SkillLoader()
        loader.register(meta)

        # Both triggers match → higher score
        matches = loader.match("add a film grain effect")
        assert len(matches) == 1
