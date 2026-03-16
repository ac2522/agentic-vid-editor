"""Tests for plugin manifest parsing and discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from ave.plugins.discovery import PluginManifest, parse_manifest, discover_plugins


class TestPluginManifest:
    def test_parse_valid_manifest(self, tmp_path):
        manifest = tmp_path / "plugin.yaml"
        manifest.write_text(
            "name: test-plugin\n"
            "description: A test plugin\n"
            "version: 1.0.0\n"
            "domain: editing\n"
            "tools:\n"
            "  - name: my_tool\n"
            "    summary: Does something\n"
        )
        result = parse_manifest(manifest)
        assert result.name == "test-plugin"
        assert result.description == "A test plugin"
        assert result.version == "1.0.0"
        assert result.domain == "editing"
        assert len(result.tools) == 1
        assert result.tools[0].name == "my_tool"
        assert result.tools[0].summary == "Does something"

    def test_parse_manifest_with_requirements(self, tmp_path):
        manifest = tmp_path / "plugin.yaml"
        manifest.write_text(
            "name: gpu-plugin\n"
            "description: Needs GPU\n"
            "version: 0.1.0\n"
            "domain: vfx\n"
            "tools: []\n"
            "requires:\n"
            "  python: [torch]\n"
            "  system: [cuda]\n"
        )
        result = parse_manifest(manifest)
        assert result.requires_python == ("torch",)
        assert result.requires_system == ("cuda",)

    def test_parse_manifest_missing_required_field(self, tmp_path):
        manifest = tmp_path / "plugin.yaml"
        manifest.write_text("name: incomplete\n")
        with pytest.raises(ValueError, match="description"):
            parse_manifest(manifest)

    def test_manifest_is_frozen(self, tmp_path):
        manifest = tmp_path / "plugin.yaml"
        manifest.write_text(
            "name: frozen\n"
            "description: test\n"
            "version: 1.0.0\n"
            "domain: test\n"
            "tools: []\n"
        )
        result = parse_manifest(manifest)
        with pytest.raises(AttributeError):
            result.name = "changed"  # type: ignore[misc]

    def test_parse_manifest_sets_path(self, tmp_path):
        manifest = tmp_path / "plugin.yaml"
        manifest.write_text(
            "name: pathed\n"
            "description: test\n"
            "version: 1.0.0\n"
            "domain: test\n"
            "tools: []\n"
        )
        result = parse_manifest(manifest)
        assert result.path == tmp_path


class TestDiscoverPlugins:
    def test_discover_from_single_directory(self, tmp_path):
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: my-plugin\n"
            "description: test\n"
            "version: 1.0.0\n"
            "domain: editing\n"
            "tools:\n"
            "  - name: do_thing\n"
            "    summary: Does a thing\n"
        )
        manifests = discover_plugins([tmp_path])
        assert len(manifests) == 1
        assert manifests[0].name == "my-plugin"

    def test_discover_respects_priority_order(self, tmp_path):
        """Earlier directories have higher priority."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        for d in (dir_a, dir_b):
            plugin = d / "same-name"
            plugin.mkdir(parents=True)
            (plugin / "plugin.yaml").write_text(
                f"name: same-name\n"
                f"description: from {d.name}\n"
                f"version: 1.0.0\n"
                f"domain: editing\n"
                f"tools: []\n"
            )
        manifests = discover_plugins([dir_a, dir_b])
        assert len(manifests) == 1
        assert manifests[0].description == "from a"

    def test_discover_skips_directories_without_manifest(self, tmp_path):
        no_manifest = tmp_path / "empty-plugin"
        no_manifest.mkdir()
        manifests = discover_plugins([tmp_path])
        assert len(manifests) == 0

    def test_discover_skips_nonexistent_directories(self):
        manifests = discover_plugins([Path("/nonexistent/path")])
        assert len(manifests) == 0

    def test_discover_multiple_plugins(self, tmp_path):
        for name in ("alpha", "beta", "gamma"):
            d = tmp_path / name
            d.mkdir()
            (d / "plugin.yaml").write_text(
                f"name: {name}\n"
                f"description: Plugin {name}\n"
                f"version: 1.0.0\n"
                f"domain: editing\n"
                f"tools: []\n"
            )
        manifests = discover_plugins([tmp_path])
        assert len(manifests) == 3
        names = {m.name for m in manifests}
        assert names == {"alpha", "beta", "gamma"}
