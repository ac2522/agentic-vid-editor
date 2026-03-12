"""yt-dlp download tool — pure logic layer.

Builds CLI argument lists for yt-dlp commands and parses JSON output.
No subprocess calls — those belong in the ops/agent layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class DownloadFormat(Enum):
    """What to download."""

    VIDEO = "video"
    AUDIO = "audio"
    BOTH = "both"


class DownloadQuality(Enum):
    """Quality selection."""

    BEST = "best"
    WORST = "worst"


@dataclass(frozen=True)
class DownloadRequest:
    """Validated download request parameters."""

    url: str
    format: DownloadFormat
    quality: DownloadQuality
    output_dir: str
    audio_format: str = "mp3"
    format_id: str | None = None
    max_height: int | None = None


@dataclass(frozen=True)
class DownloadResult:
    """Result of a completed download."""

    title: str
    url: str
    filepath: str
    format_id: str
    duration_seconds: int


@dataclass(frozen=True)
class SearchResult:
    """A single search result from yt-dlp."""

    video_id: str
    title: str
    url: str
    duration_seconds: int = 0
    uploader: str = ""
    view_count: int = 0


@dataclass(frozen=True)
class FormatInfo:
    """Available format info for a video."""

    format_id: str
    ext: str
    resolution: str
    fps: int | None
    vcodec: str
    acodec: str
    filesize: int | None
    format_note: str
    bitrate: float | None
    has_video: bool
    has_audio: bool


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


def build_search_args(query: str, max_results: int = 5) -> list[str]:
    """Build yt-dlp args to search YouTube and return JSON metadata."""
    n = max(1, min(50, max_results))
    return [
        "--dump-json",
        "--flat-playlist",
        "--no-download",
        f"ytsearch{n}:{query}",
    ]


def build_list_formats_args(url: str) -> list[str]:
    """Build yt-dlp args to list available formats as JSON."""
    return [
        "--dump-json",
        "--no-download",
        "--no-playlist",
        url,
    ]


def _format_selector(
    fmt: DownloadFormat,
    quality: DownloadQuality,
    format_id: str | None = None,
    max_height: int | None = None,
) -> str:
    """Compute the -f format selector string."""
    if format_id:
        return format_id

    if fmt == DownloadFormat.VIDEO:
        base = "bestvideo" if quality == DownloadQuality.BEST else "worstvideo"
        if max_height:
            return f"{base}[height<={max_height}]"
        return base

    if fmt == DownloadFormat.AUDIO:
        return "bestaudio" if quality == DownloadQuality.BEST else "worstaudio"

    # BOTH
    if quality == DownloadQuality.BEST:
        if max_height:
            return f"bestvideo*[height<={max_height}]+bestaudio/best[height<={max_height}]"
        return "bestvideo*+bestaudio/best"

    # WORST + BOTH
    return "worstvideo*+worstaudio/worst"


def build_download_args(
    url: str,
    format: DownloadFormat,
    quality: DownloadQuality,
    output_dir: str,
    audio_format: str = "mp3",
    format_id: str | None = None,
    max_height: int | None = None,
) -> list[str]:
    """Build yt-dlp args to download media."""
    args: list[str] = []

    # Format selection
    if format == DownloadFormat.AUDIO and not format_id:
        args.extend(["-x", "--audio-format", audio_format])
        selector = _format_selector(format, quality, format_id, max_height)
        args.extend(["-f", selector])
    else:
        selector = _format_selector(format, quality, format_id, max_height)
        args.extend(["-f", selector])

    # Output template
    args.extend(["-o", f"{output_dir}/%(title)s.%(ext)s"])

    # Safety flags
    args.append("--no-playlist")

    # Machine-readable output
    args.append("--dump-json")

    args.append(url)

    return args


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_search_results(raw_output: str) -> list[SearchResult]:
    """Parse yt-dlp --dump-json search output into SearchResult list."""
    if not raw_output.strip():
        return []

    results: list[SearchResult] = []
    for line in raw_output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        results.append(SearchResult(
            video_id=data.get("id", ""),
            title=data.get("title", ""),
            url=data.get("webpage_url", ""),
            duration_seconds=data.get("duration", 0) or 0,
            uploader=data.get("uploader", "") or "",
            view_count=data.get("view_count", 0) or 0,
        ))

    return results


def parse_format_list(raw_output: str) -> list[FormatInfo]:
    """Parse yt-dlp --dump-json output into FormatInfo list."""
    if not raw_output.strip():
        return []

    data = json.loads(raw_output)
    formats_data = data.get("formats", [])

    formats: list[FormatInfo] = []
    for f in formats_data:
        vcodec = f.get("vcodec", "none") or "none"
        acodec = f.get("acodec", "none") or "none"
        has_video = vcodec != "none"
        has_audio = acodec != "none"

        formats.append(FormatInfo(
            format_id=f.get("format_id", ""),
            ext=f.get("ext", ""),
            resolution=f.get("resolution", ""),
            fps=f.get("fps"),
            vcodec=vcodec,
            acodec=acodec,
            filesize=f.get("filesize"),
            format_note=f.get("format_note", ""),
            bitrate=f.get("tbr"),
            has_video=has_video,
            has_audio=has_audio,
        ))

    return formats
