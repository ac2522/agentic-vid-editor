"""Tests for yt-dlp download operations layer (subprocess execution)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from ave.tools.download import DownloadFormat, DownloadQuality
from ave.tools.download_ops import (
    DownloadError,
    search_youtube,
    list_formats,
    download_media,
)


# ---------------------------------------------------------------------------
# search_youtube
# ---------------------------------------------------------------------------


class TestSearchYoutube:
    def test_returns_search_results(self):
        fake_output = "\n".join([
            json.dumps({
                "id": "abc123",
                "title": "Lo-fi beats",
                "webpage_url": "https://www.youtube.com/watch?v=abc123",
                "duration": 3600,
                "uploader": "ChilledCow",
                "view_count": 1_000_000,
            }),
            json.dumps({
                "id": "def456",
                "title": "Jazz piano",
                "webpage_url": "https://www.youtube.com/watch?v=def456",
                "duration": 1800,
                "uploader": "JazzFM",
                "view_count": 50_000,
            }),
        ])

        with patch("ave.tools.download_ops._run_ytdlp") as mock_run:
            mock_run.return_value = fake_output
            results = search_youtube("lofi hip hop", max_results=2)

        assert len(results) == 2
        assert results[0].video_id == "abc123"
        assert results[0].title == "Lo-fi beats"
        assert results[1].video_id == "def456"

        # Verify yt-dlp was called with correct args
        call_args = mock_run.call_args[0][0]
        assert "ytsearch2:lofi hip hop" in " ".join(call_args)

    def test_raises_on_failure(self):
        with patch("ave.tools.download_ops._run_ytdlp") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "yt-dlp", stderr=b"error")
            with pytest.raises(DownloadError, match="Search failed"):
                search_youtube("nonexistent")


# ---------------------------------------------------------------------------
# list_formats
# ---------------------------------------------------------------------------


class TestListFormats:
    def test_returns_format_list(self):
        fake_output = json.dumps({
            "id": "abc123",
            "title": "Test Video",
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
            ],
        })

        with patch("ave.tools.download_ops._run_ytdlp") as mock_run:
            mock_run.return_value = fake_output
            formats = list_formats("https://www.youtube.com/watch?v=abc123")

        assert len(formats) == 1
        assert formats[0].format_id == "137"
        assert formats[0].has_video is True
        assert formats[0].has_audio is False

    def test_raises_on_invalid_url(self):
        with patch("ave.tools.download_ops._run_ytdlp") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "yt-dlp", stderr=b"not a URL")
            with pytest.raises(DownloadError, match="Failed to list formats"):
                list_formats("not-a-url")


# ---------------------------------------------------------------------------
# download_media
# ---------------------------------------------------------------------------


class TestDownloadMedia:
    def test_download_returns_result(self, tmp_path):
        fake_output = json.dumps({
            "id": "abc123",
            "title": "Test Song",
            "webpage_url": "https://www.youtube.com/watch?v=abc123",
            "duration": 240,
            "requested_downloads": [
                {"filepath": str(tmp_path / "Test Song.mp4")},
            ],
            "format_id": "137+140",
        })

        with patch("ave.tools.download_ops._run_ytdlp") as mock_run:
            mock_run.return_value = fake_output
            result = download_media(
                url="https://www.youtube.com/watch?v=abc123",
                format=DownloadFormat.BOTH,
                quality=DownloadQuality.BEST,
                output_dir=str(tmp_path),
            )

        assert result.title == "Test Song"
        assert result.url == "https://www.youtube.com/watch?v=abc123"
        assert result.format_id == "137+140"
        assert result.duration_seconds == 240

    def test_download_audio_only(self, tmp_path):
        fake_output = json.dumps({
            "id": "abc123",
            "title": "Cool Song",
            "webpage_url": "https://www.youtube.com/watch?v=abc123",
            "duration": 180,
            "requested_downloads": [
                {"filepath": str(tmp_path / "Cool Song.mp3")},
            ],
            "format_id": "140",
        })

        with patch("ave.tools.download_ops._run_ytdlp") as mock_run:
            mock_run.return_value = fake_output
            result = download_media(
                url="https://www.youtube.com/watch?v=abc123",
                format=DownloadFormat.AUDIO,
                quality=DownloadQuality.BEST,
                output_dir=str(tmp_path),
                audio_format="mp3",
            )

        assert result.filepath.endswith(".mp3")

        # Verify -x flag was passed
        call_args = mock_run.call_args[0][0]
        assert "-x" in call_args

    def test_download_raises_on_failure(self, tmp_path):
        with patch("ave.tools.download_ops._run_ytdlp") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "yt-dlp", stderr=b"Video unavailable"
            )
            with pytest.raises(DownloadError, match="Download failed"):
                download_media(
                    url="https://www.youtube.com/watch?v=bad",
                    format=DownloadFormat.BOTH,
                    quality=DownloadQuality.BEST,
                    output_dir=str(tmp_path),
                )

    def test_download_with_format_id(self, tmp_path):
        fake_output = json.dumps({
            "id": "abc123",
            "title": "Specific Format",
            "webpage_url": "https://www.youtube.com/watch?v=abc123",
            "duration": 300,
            "requested_downloads": [
                {"filepath": str(tmp_path / "Specific Format.mp4")},
            ],
            "format_id": "247+251",
        })

        with patch("ave.tools.download_ops._run_ytdlp") as mock_run:
            mock_run.return_value = fake_output
            result = download_media(
                url="https://www.youtube.com/watch?v=abc123",
                format=DownloadFormat.BOTH,
                quality=DownloadQuality.BEST,
                output_dir=str(tmp_path),
                format_id="247+251",
            )

        call_args = mock_run.call_args[0][0]
        fmt_idx = call_args.index("-f")
        assert call_args[fmt_idx + 1] == "247+251"
        assert result.format_id == "247+251"


# ---------------------------------------------------------------------------
# Agent tool registration
# ---------------------------------------------------------------------------


class TestDownloadToolRegistration:
    def test_download_tools_registered(self):
        from ave.agent.session import EditingSession

        session = EditingSession()
        results = session.search_tools(domain="download")
        names = {t.name for t in results}
        assert "search_youtube" in names
        assert "download_media" in names
        assert "list_available_formats" in names

    def test_search_finds_download_tools(self):
        from ave.agent.session import EditingSession

        session = EditingSession()

        # "download song" should find download_media
        results = session.search_tools("download song")
        names = [t.name for t in results]
        assert "download_media" in names

        # "youtube search" should find search_youtube
        results = session.search_tools("youtube search")
        names = [t.name for t in results]
        assert "search_youtube" in names

    def test_search_youtube_no_prerequisites(self):
        from ave.agent.session import EditingSession

        session = EditingSession()
        schema = session.get_tool_schema("search_youtube")
        assert schema.requires == []
