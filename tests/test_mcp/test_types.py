"""Tests for MCP response data models."""

from __future__ import annotations

import pytest

from ave.mcp.types import EditResult, ProjectState, PreviewResult, AssetInfo


class TestEditResult:
    def test_create_success(self):
        r = EditResult(
            success=True,
            description="Added cross dissolve between clips 3 and 4",
            tools_used=["ave:editing.transition"],
            preview_path="/tmp/preview.jpg",
        )
        assert r.success is True
        assert "cross dissolve" in r.description

    def test_create_failure(self):
        r = EditResult(
            success=False,
            description="No clips found",
            tools_used=[],
            error="clip_not_found",
        )
        assert r.success is False
        assert r.error == "clip_not_found"

    def test_frozen(self):
        r = EditResult(success=True, description="ok")
        with pytest.raises(AttributeError):
            r.success = False  # type: ignore[misc]


class TestProjectState:
    def test_create_with_clips(self):
        s = ProjectState(
            clip_count=3,
            duration_ns=5_000_000_000,
            layers=1,
            clips=[{"name": "clip1"}],
        )
        assert s.clip_count == 3
        assert len(s.clips) == 1


class TestPreviewResult:
    def test_create(self):
        r = PreviewResult(path="/tmp/frame.jpg", format="jpeg", width=1920, height=1080)
        assert r.path == "/tmp/frame.jpg"


class TestAssetInfo:
    def test_create(self):
        a = AssetInfo(
            asset_id="abc123",
            path="/media/clip.mov",
            codec="prores",
            width=1920,
            height=1080,
            duration_ns=10_000_000_000,
            color_space="bt709",
        )
        assert a.codec == "prores"
