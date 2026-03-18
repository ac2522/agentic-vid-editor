"""Tests for proxy-first workflow."""

import pytest

from ave.tools.proxy import (
    ConformError,
    ConformResult,
    ProxyConfig,
    ProxyWorkflow,
)


class TestProxyConfig:
    def test_defaults(self):
        cfg = ProxyConfig()
        assert cfg.proxy_width == 854
        assert cfg.proxy_height == 480
        assert cfg.proxy_codec == "libx264"
        assert cfg.proxy_preset == "ultrafast"

    def test_frozen(self):
        cfg = ProxyConfig()
        with pytest.raises(AttributeError):
            cfg.proxy_width = 1920  # type: ignore[misc]


class TestProxyWorkflow:
    def test_get_editing_path_returns_proxy_when_available(self, tmp_path):
        proxy = tmp_path / "proxy.mp4"
        proxy.touch()
        full = tmp_path / "full.mxf"
        full.touch()
        wf = ProxyWorkflow()
        entry = {"proxy_path": str(proxy), "working_path": str(full)}
        assert wf.get_editing_path(entry) == proxy

    def test_get_editing_path_returns_fullres_when_no_proxy(self, tmp_path):
        full = tmp_path / "full.mxf"
        full.touch()
        wf = ProxyWorkflow()
        entry = {"proxy_path": None, "working_path": str(full)}
        assert wf.get_editing_path(entry) == full

    def test_get_editing_path_returns_fullres_when_proxy_missing_on_disk(self, tmp_path):
        full = tmp_path / "full.mxf"
        full.touch()
        wf = ProxyWorkflow()
        entry = {"proxy_path": str(tmp_path / "nonexistent.mp4"), "working_path": str(full)}
        assert wf.get_editing_path(entry) == full

    def test_conform_timeline_replaces_paths(self, tmp_path):
        # Create full-res files so validation passes
        full1 = tmp_path / "full1.mxf"
        full1.touch()
        full2 = tmp_path / "full2.mxf"
        full2.touch()

        xges = '<ges><clip uri="/proxy/a.mp4"/><clip uri="/proxy/b.mp4"/></ges>'
        mapping = {"/proxy/a.mp4": str(full1), "/proxy/b.mp4": str(full2)}

        wf = ProxyWorkflow()
        result = wf.conform_timeline(xges, mapping)
        assert isinstance(result, ConformResult)
        assert result.swaps == 2

    def test_conform_timeline_raises_on_missing_fullres(self, tmp_path):
        missing = tmp_path / "does_not_exist.mxf"
        xges = '<clip uri="/proxy/a.mp4"/>'
        mapping = {"/proxy/a.mp4": str(missing)}

        wf = ProxyWorkflow()
        with pytest.raises(ConformError) as exc_info:
            wf.conform_timeline(xges, mapping)
        assert missing in exc_info.value.missing_files

    def test_conform_error_includes_missing_files(self, tmp_path):
        m1 = tmp_path / "m1.mxf"
        m2 = tmp_path / "m2.mxf"
        err = ConformError([m1, m2])
        assert err.missing_files == [m1, m2]
        assert "m1.mxf" in str(err)

    def test_conform_timeline_swap_count(self, tmp_path):
        full = tmp_path / "full.mxf"
        full.touch()
        # Proxy path appears twice in the XML
        xges = '<clip uri="/proxy/a.mp4"/> <clip uri="/proxy/a.mp4"/>'
        mapping = {"/proxy/a.mp4": str(full)}
        wf = ProxyWorkflow()
        result = wf.conform_timeline(xges, mapping)
        assert result.swaps == 2

    def test_build_path_mapping(self):
        entries = [
            {"proxy_path": "/proxy/a.mp4", "working_path": "/full/a.mxf"},
            {"proxy_path": "/proxy/b.mp4", "working_path": "/full/b.mxf"},
        ]
        wf = ProxyWorkflow()
        mapping = wf.build_path_mapping(entries)
        assert mapping == {
            "/proxy/a.mp4": "/full/a.mxf",
            "/proxy/b.mp4": "/full/b.mxf",
        }
