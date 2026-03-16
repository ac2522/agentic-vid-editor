"""Tests for plugin/skill integration in EditingSession."""

from __future__ import annotations


import pytest

from ave.agent.session import EditingSession, SessionError


class TestSessionPluginIntegration:
    def test_session_discovers_plugins(self, tmp_path):
        plugin_dir = tmp_path / "plugins" / "hello"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            "name: hello\n"
            "description: Hello plugin\n"
            "version: 1.0.0\n"
            "domain: testing\n"
            "tools:\n"
            "  - name: say_hello\n"
            "    summary: Says hello\n"
        )
        (plugin_dir / "__init__.py").write_text(
            "def register(registry, namespace):\n"
            "    @registry.tool(domain='testing', namespace=namespace)\n"
            "    def say_hello(name: str) -> dict:\n"
            "        '''Say hello.'''\n"
            "        return {'msg': f'hello {name}'}\n"
        )
        session = EditingSession(
            plugin_dirs=[tmp_path / "plugins"],
        )
        results = session.search_tools("hello")
        assert any("say_hello" in r.name for r in results)

    def test_session_discovers_skills(self, tmp_path):
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()
        (skill_dir / "grain.md").write_text(
            "---\nname: grain\ndescription: Film grain effect\n"
            "domain: color\ntriggers: [film grain]\n---\n## Steps\n1. Do it\n"
        )
        session = EditingSession(
            skill_dirs=[skill_dir],
        )
        matches = session.match_skills("add film grain")
        assert len(matches) >= 1
        assert matches[0].name == "grain"

    def test_session_load_skill_body(self, tmp_path):
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()
        (skill_dir / "grain.md").write_text(
            "---\nname: grain\ndescription: Film grain\n"
            "domain: color\ntriggers: []\n---\n\n## Steps\n1. Apply overlay\n"
        )
        session = EditingSession(skill_dirs=[skill_dir])
        body = session.load_skill("grain")
        assert "Apply overlay" in body

    def test_session_load_unknown_skill_raises(self):
        session = EditingSession()
        with pytest.raises(SessionError, match="Unknown skill"):
            session.load_skill("nonexistent")

    def test_session_without_plugins_still_works(self):
        """Backward compat — session without plugins/skills works fine."""
        session = EditingSession()
        assert session.registry.tool_count > 0
        results = session.search_tools("trim")
        assert len(results) >= 1
