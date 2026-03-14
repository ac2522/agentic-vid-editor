"""Tests for ave.web.api pure REST endpoint functions."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ave.web.api import get_assets_response, get_timeline_response


class TestGetTimelineResponse:
    def test_calls_to_dict(self):
        model = MagicMock()
        model.to_dict.return_value = {"tracks": [], "duration_ns": 0}
        result = get_timeline_response(model)
        model.to_dict.assert_called_once()
        assert result == {"tracks": [], "duration_ns": 0}

    def test_returns_dict(self):
        model = MagicMock()
        model.to_dict.return_value = {"tracks": [{"id": "V0"}], "duration_ns": 5_000_000_000}
        result = get_timeline_response(model)
        assert isinstance(result, dict)
        assert result["duration_ns"] == 5_000_000_000


class TestGetAssetsResponse:
    def test_reads_registry_file(self, tmp_path: Path):
        registry = tmp_path / "registry.json"
        registry.write_text(json.dumps([
            {
                "asset_id": "abc123",
                "original_path": "/media/clip01.mp4",
                "working_path": "/work/clip01.mxf",
                "proxy_path": "/proxy/clip01.mp4",
                "original_fps": 23.976,
                "conformed_fps": 24.0,
                "duration_seconds": 10.5,
                "width": 1920,
                "height": 1080,
                "codec": "h264",
                "camera_color_space": "sRGB",
                "camera_transfer": "linear",
                "idt_reference": None,
                "transcription_path": None,
                "visual_analysis_path": None,
            }
        ]))
        result = get_assets_response(registry)
        assert "assets" in result
        assert len(result["assets"]) == 1
        asset = result["assets"][0]
        assert asset["id"] == "abc123"
        assert asset["name"] == "clip01.mp4"
        assert asset["duration_ns"] == 10_500_000_000
        assert asset["resolution"] == "1920x1080"
        assert asset["fps"] == 23.976
        assert asset["thumbnail_url"] == "/api/assets/abc123/thumbnail"

    def test_multiple_assets(self, tmp_path: Path):
        registry = tmp_path / "registry.json"
        entries = [
            {
                "asset_id": f"id{i}",
                "original_path": f"/media/clip{i}.mp4",
                "working_path": f"/work/clip{i}.mxf",
                "proxy_path": None,
                "original_fps": 30.0,
                "conformed_fps": 30.0,
                "duration_seconds": 5.0,
                "width": 3840,
                "height": 2160,
                "codec": "hevc",
                "camera_color_space": None,
                "camera_transfer": None,
                "idt_reference": None,
                "transcription_path": None,
                "visual_analysis_path": None,
            }
            for i in range(3)
        ]
        registry.write_text(json.dumps(entries))
        result = get_assets_response(registry)
        assert len(result["assets"]) == 3
        assert result["assets"][1]["id"] == "id1"
        assert result["assets"][2]["resolution"] == "3840x2160"

    def test_missing_file_returns_empty_list(self, tmp_path: Path):
        registry = tmp_path / "nonexistent.json"
        result = get_assets_response(registry)
        assert result == {"assets": []}

    def test_invalid_json_returns_empty_list(self, tmp_path: Path):
        registry = tmp_path / "bad.json"
        registry.write_text("not valid json {{{")
        result = get_assets_response(registry)
        assert result == {"assets": []}

    def test_empty_array_returns_empty_list(self, tmp_path: Path):
        registry = tmp_path / "empty.json"
        registry.write_text("[]")
        result = get_assets_response(registry)
        assert result == {"assets": []}

    def test_duration_conversion_to_nanoseconds(self, tmp_path: Path):
        registry = tmp_path / "registry.json"
        registry.write_text(json.dumps([
            {
                "asset_id": "x",
                "original_path": "/v/test.mov",
                "working_path": "/w/test.mxf",
                "proxy_path": None,
                "original_fps": 24.0,
                "conformed_fps": 24.0,
                "duration_seconds": 0.001,
                "width": 640,
                "height": 480,
                "codec": "prores",
                "camera_color_space": None,
                "camera_transfer": None,
                "idt_reference": None,
                "transcription_path": None,
                "visual_analysis_path": None,
            }
        ]))
        result = get_assets_response(registry)
        assert result["assets"][0]["duration_ns"] == 1_000_000

    def test_name_extracted_from_path_basename(self, tmp_path: Path):
        registry = tmp_path / "registry.json"
        registry.write_text(json.dumps([
            {
                "asset_id": "deep",
                "original_path": "/a/b/c/d/my_video.mkv",
                "working_path": "/w/my_video.mxf",
                "proxy_path": None,
                "original_fps": 25.0,
                "conformed_fps": 25.0,
                "duration_seconds": 1.0,
                "width": 720,
                "height": 576,
                "codec": "ffv1",
                "camera_color_space": None,
                "camera_transfer": None,
                "idt_reference": None,
                "transcription_path": None,
                "visual_analysis_path": None,
            }
        ]))
        result = get_assets_response(registry)
        assert result["assets"][0]["name"] == "my_video.mkv"
