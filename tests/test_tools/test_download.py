"""Tests for yt-dlp download tool — pure logic layer."""

from __future__ import annotations

import json

import pytest

from ave.tools.download import (
    DownloadFormat,
    DownloadQuality,
    DownloadRequest,
    DownloadResult,
    FormatInfo,
    SearchResult,
    build_download_args,
    build_list_formats_args,
    build_search_args,
    parse_format_list,
    parse_search_results,
)


# ---------------------------------------------------------------------------
# DownloadFormat / DownloadQuality enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_download_format_values(self):
        assert DownloadFormat.VIDEO.value == "video"
        assert DownloadFormat.AUDIO.value == "audio"
        assert DownloadFormat.BOTH.value == "both"

    def test_download_quality_values(self):
        assert DownloadQuality.BEST.value == "best"
        assert DownloadQuality.WORST.value == "worst"


# ---------------------------------------------------------------------------
# build_search_args
# ---------------------------------------------------------------------------


class TestBuildSearchArgs:
    def test_basic_search(self):
        args = build_search_args("lofi hip hop")
        assert "ytsearch5:lofi hip hop" in " ".join(args)
        assert "--dump-json" in args
        assert "--flat-playlist" in args
        assert "--no-download" in args

    def test_custom_max_results(self):
        args = build_search_args("jazz piano", max_results=10)
        assert "ytsearch10:jazz piano" in " ".join(args)

    def test_max_results_clamped_to_1(self):
        args = build_search_args("test", max_results=0)
        assert "ytsearch1:test" in " ".join(args)

    def test_max_results_clamped_to_50(self):
        args = build_search_args("test", max_results=100)
        assert "ytsearch50:test" in " ".join(args)


# ---------------------------------------------------------------------------
# build_list_formats_args
# ---------------------------------------------------------------------------


class TestBuildListFormatsArgs:
    def test_basic_list(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        args = build_list_formats_args(url)
        assert url in args
        assert "--dump-json" in args
        assert "--no-download" in args

    def test_no_playlist(self):
        url = "https://www.youtube.com/watch?v=abc123"
        args = build_list_formats_args(url)
        assert "--no-playlist" in args


# ---------------------------------------------------------------------------
# build_download_args
# ---------------------------------------------------------------------------


class TestBuildDownloadArgs:
    def test_best_video_and_audio(self):
        args = build_download_args(
            url="https://www.youtube.com/watch?v=abc123",
            format=DownloadFormat.BOTH,
            quality=DownloadQuality.BEST,
            output_dir="/tmp/downloads",
        )
        assert "-f" in args
        fmt_idx = args.index("-f")
        assert args[fmt_idx + 1] == "bestvideo*+bestaudio/best"
        assert "-o" in args

    def test_audio_only_best(self):
        args = build_download_args(
            url="https://www.youtube.com/watch?v=abc123",
            format=DownloadFormat.AUDIO,
            quality=DownloadQuality.BEST,
            output_dir="/tmp/downloads",
        )
        assert "-x" in args
        assert "--audio-format" in args
        fmt_idx = args.index("--audio-format")
        assert args[fmt_idx + 1] == "mp3"

    def test_audio_only_custom_format(self):
        args = build_download_args(
            url="https://www.youtube.com/watch?v=abc123",
            format=DownloadFormat.AUDIO,
            quality=DownloadQuality.BEST,
            output_dir="/tmp/downloads",
            audio_format="flac",
        )
        fmt_idx = args.index("--audio-format")
        assert args[fmt_idx + 1] == "flac"

    def test_video_only_best(self):
        args = build_download_args(
            url="https://www.youtube.com/watch?v=abc123",
            format=DownloadFormat.VIDEO,
            quality=DownloadQuality.BEST,
            output_dir="/tmp/downloads",
        )
        assert "-f" in args
        fmt_idx = args.index("-f")
        assert args[fmt_idx + 1] == "bestvideo"

    def test_worst_quality(self):
        args = build_download_args(
            url="https://www.youtube.com/watch?v=abc123",
            format=DownloadFormat.BOTH,
            quality=DownloadQuality.WORST,
            output_dir="/tmp/downloads",
        )
        fmt_idx = args.index("-f")
        assert args[fmt_idx + 1] == "worstvideo*+worstaudio/worst"

    def test_output_template(self):
        args = build_download_args(
            url="https://www.youtube.com/watch?v=abc123",
            format=DownloadFormat.BOTH,
            quality=DownloadQuality.BEST,
            output_dir="/tmp/out",
        )
        out_idx = args.index("-o")
        assert args[out_idx + 1].startswith("/tmp/out/")
        assert "%(title)s" in args[out_idx + 1]

    def test_includes_no_playlist(self):
        args = build_download_args(
            url="https://www.youtube.com/watch?v=abc123",
            format=DownloadFormat.BOTH,
            quality=DownloadQuality.BEST,
            output_dir="/tmp/out",
        )
        assert "--no-playlist" in args

    def test_custom_format_id(self):
        """User can pass a specific format ID string instead of quality enum."""
        args = build_download_args(
            url="https://www.youtube.com/watch?v=abc123",
            format=DownloadFormat.BOTH,
            quality=DownloadQuality.BEST,
            output_dir="/tmp/out",
            format_id="137+140",
        )
        fmt_idx = args.index("-f")
        assert args[fmt_idx + 1] == "137+140"

    def test_max_height_filter(self):
        """Quality can be constrained by max height (e.g. 720p)."""
        args = build_download_args(
            url="https://www.youtube.com/watch?v=abc123",
            format=DownloadFormat.BOTH,
            quality=DownloadQuality.BEST,
            output_dir="/tmp/out",
            max_height=720,
        )
        fmt_idx = args.index("-f")
        assert "height<=720" in args[fmt_idx + 1]

    def test_includes_print_json(self):
        """Should include --print-json for machine-readable output."""
        args = build_download_args(
            url="https://www.youtube.com/watch?v=abc123",
            format=DownloadFormat.BOTH,
            quality=DownloadQuality.BEST,
            output_dir="/tmp/out",
        )
        assert "--dump-json" in args


# ---------------------------------------------------------------------------
# parse_search_results
# ---------------------------------------------------------------------------


class TestParseSearchResults:
    def test_parses_single_result(self):
        data = json.dumps({
            "id": "abc123",
            "title": "Test Video",
            "webpage_url": "https://www.youtube.com/watch?v=abc123",
            "duration": 120,
            "uploader": "Test Channel",
            "view_count": 5000,
        })
        results = parse_search_results(data)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.video_id == "abc123"
        assert r.title == "Test Video"
        assert r.url == "https://www.youtube.com/watch?v=abc123"
        assert r.duration_seconds == 120
        assert r.uploader == "Test Channel"

    def test_parses_multiple_results(self):
        lines = []
        for i in range(3):
            lines.append(json.dumps({
                "id": f"vid{i}",
                "title": f"Video {i}",
                "webpage_url": f"https://www.youtube.com/watch?v=vid{i}",
                "duration": 60 * (i + 1),
                "uploader": f"Channel {i}",
                "view_count": 100 * (i + 1),
            }))
        raw = "\n".join(lines)
        results = parse_search_results(raw)
        assert len(results) == 3
        assert results[0].video_id == "vid0"
        assert results[2].video_id == "vid2"

    def test_handles_missing_fields(self):
        data = json.dumps({
            "id": "xyz",
            "title": "Minimal",
            "webpage_url": "https://youtube.com/watch?v=xyz",
        })
        results = parse_search_results(data)
        assert len(results) == 1
        assert results[0].duration_seconds == 0
        assert results[0].uploader == ""

    def test_empty_input(self):
        results = parse_search_results("")
        assert results == []


# ---------------------------------------------------------------------------
# parse_format_list
# ---------------------------------------------------------------------------


class TestParseFormatList:
    def test_parses_formats(self):
        data = json.dumps({
            "id": "abc",
            "title": "Test",
            "formats": [
                {
                    "format_id": "137",
                    "ext": "mp4",
                    "resolution": "1920x1080",
                    "fps": 30,
                    "vcodec": "avc1.640028",
                    "acodec": "none",
                    "filesize": 50_000_000,
                    "format_note": "1080p",
                    "tbr": 4000.0,
                },
                {
                    "format_id": "140",
                    "ext": "m4a",
                    "resolution": "audio only",
                    "fps": None,
                    "vcodec": "none",
                    "acodec": "mp4a.40.2",
                    "filesize": 3_000_000,
                    "format_note": "medium",
                    "tbr": 128.0,
                },
            ],
        })
        formats = parse_format_list(data)
        assert len(formats) == 2
        assert isinstance(formats[0], FormatInfo)
        assert formats[0].format_id == "137"
        assert formats[0].ext == "mp4"
        assert formats[0].has_video is True
        assert formats[0].has_audio is False
        assert formats[1].format_id == "140"
        assert formats[1].has_video is False
        assert formats[1].has_audio is True

    def test_empty_formats(self):
        data = json.dumps({"id": "abc", "title": "Test", "formats": []})
        assert parse_format_list(data) == []

    def test_format_with_both_streams(self):
        data = json.dumps({
            "id": "abc",
            "title": "Test",
            "formats": [
                {
                    "format_id": "18",
                    "ext": "mp4",
                    "resolution": "640x360",
                    "fps": 30,
                    "vcodec": "avc1.42001E",
                    "acodec": "mp4a.40.2",
                    "filesize": 10_000_000,
                    "format_note": "360p",
                    "tbr": 500.0,
                },
            ],
        })
        formats = parse_format_list(data)
        assert len(formats) == 1
        assert formats[0].has_video is True
        assert formats[0].has_audio is True


# ---------------------------------------------------------------------------
# DownloadRequest dataclass
# ---------------------------------------------------------------------------


class TestDownloadRequest:
    def test_frozen(self):
        req = DownloadRequest(
            url="https://youtube.com/watch?v=abc",
            format=DownloadFormat.BOTH,
            quality=DownloadQuality.BEST,
            output_dir="/tmp",
        )
        with pytest.raises(AttributeError):
            req.url = "changed"  # type: ignore[misc]

    def test_defaults(self):
        req = DownloadRequest(
            url="https://youtube.com/watch?v=abc",
            format=DownloadFormat.BOTH,
            quality=DownloadQuality.BEST,
            output_dir="/tmp",
        )
        assert req.audio_format == "mp3"
        assert req.format_id is None
        assert req.max_height is None


# ---------------------------------------------------------------------------
# DownloadResult dataclass
# ---------------------------------------------------------------------------


class TestDownloadResult:
    def test_frozen(self):
        res = DownloadResult(
            title="Test",
            url="https://youtube.com/watch?v=abc",
            filepath="/tmp/Test.mp4",
            format_id="137+140",
            duration_seconds=120,
        )
        with pytest.raises(AttributeError):
            res.title = "changed"  # type: ignore[misc]
