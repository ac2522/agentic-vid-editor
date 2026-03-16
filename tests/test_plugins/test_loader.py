"""Tests for plugin lazy loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from ave.agent.registry import ToolRegistry, RegistryError
from ave.plugins.discovery import PluginManifest, ToolStub
from ave.plugins.loader import PluginLoader


def _make_plugin(tmp_path: Path, name: str = "test-plugin") -> PluginManifest:
    """Create a real plugin directory with register function."""
    plugin_dir = tmp_path / name
    plugin_dir.mkdir(exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {name}\n"
        f"description: Test\n"
        f"version: 1.0.0\n"
        f"domain: testing\n"
        f"tools:\n"
        f"  - name: greet\n"
        f"    summary: Says hello\n"
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(registry, namespace):\n"
        "    @registry.tool(domain='testing', namespace=namespace)\n"
        "    def greet(name: str) -> dict:\n"
        "        '''Say hello.'''\n"
        "        return {'greeting': f'hello {name}'}\n"
    )
    return PluginManifest(
        name=name,
        description="Test",
        version="1.0.0",
        domain="testing",
        tools=(ToolStub(name="greet", summary="Says hello"),),
        path=plugin_dir,
    )


class TestPluginLoader:
    def test_register_summaries_without_loading_code(self, tmp_path):
        manifest = _make_plugin(tmp_path)
        registry = ToolRegistry()
        loader = PluginLoader(registry)
        loader.register_manifest(manifest)

        results = registry.search_tools("greet")
        assert len(results) >= 1
        assert not loader.is_loaded(manifest.name)

    def test_lazy_load_on_first_call(self, tmp_path):
        manifest = _make_plugin(tmp_path)
        registry = ToolRegistry()
        loader = PluginLoader(registry)
        loader.register_manifest(manifest)

        result = loader.call_plugin_tool(
            manifest.name, "greet", {"name": "world"}
        )
        assert result == {"greeting": "hello world"}
        assert loader.is_loaded(manifest.name)

    def test_stub_not_callable_before_load(self, tmp_path):
        manifest = _make_plugin(tmp_path)
        registry = ToolRegistry()
        loader = PluginLoader(registry)
        loader.register_manifest(manifest)

        with pytest.raises(RegistryError, match="stub"):
            registry.call_tool("greet", {})

    def test_load_failure_marks_unavailable(self, tmp_path):
        plugin_dir = tmp_path / "broken"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("raise ImportError('missing dep')")
        manifest = PluginManifest(
            name="broken",
            description="Broken plugin",
            version="1.0.0",
            domain="testing",
            tools=(ToolStub(name="nope", summary="Won't work"),),
            path=plugin_dir,
        )
        registry = ToolRegistry()
        loader = PluginLoader(registry)
        loader.register_manifest(manifest)

        with pytest.raises(RuntimeError, match="failed to load"):
            loader.call_plugin_tool("broken", "nope", {})

    def test_repeated_failure_gives_same_error(self, tmp_path):
        plugin_dir = tmp_path / "broken"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("raise ImportError('gone')")
        manifest = PluginManifest(
            name="broken",
            description="Broken",
            version="1.0.0",
            domain="testing",
            tools=(ToolStub(name="x", summary="x"),),
            path=plugin_dir,
        )
        registry = ToolRegistry()
        loader = PluginLoader(registry)
        loader.register_manifest(manifest)

        with pytest.raises(RuntimeError):
            loader.call_plugin_tool("broken", "x", {})
        # Second call should also fail without retrying import
        with pytest.raises(RuntimeError, match="failed to load"):
            loader.call_plugin_tool("broken", "x", {})
