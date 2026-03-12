"""yt-dlp download operations — subprocess execution layer.

Wraps pure logic from ave.tools.download with actual yt-dlp subprocess calls.
"""

from __future__ import annotations

import json
import subprocess

from ave.tools.download import (
    DownloadFormat,
    DownloadQuality,
    DownloadResult,
    FormatInfo,
    SearchResult,
    build_download_args,
    build_list_formats_args,
    build_search_args,
    parse_format_list,
    parse_search_results,
)


class DownloadError(Exception):
    """Raised when a yt-dlp operation fails."""


def _run_ytdlp(args: list[str], timeout: int = 300) -> str:
    """Run yt-dlp with given args and return stdout."""
    result = subprocess.run(
        ["yt-dlp", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    result.check_returncode()
    return result.stdout


def search_youtube(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search YouTube and return structured results."""
    args = build_search_args(query, max_results)
    try:
        output = _run_ytdlp(args)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode()
        raise DownloadError(f"Search failed: {stderr}") from e
    return parse_search_results(output)


def list_formats(url: str) -> list[FormatInfo]:
    """List available formats for a URL."""
    args = build_list_formats_args(url)
    try:
        output = _run_ytdlp(args)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode()
        raise DownloadError(f"Failed to list formats: {stderr}") from e
    return parse_format_list(output)


def download_media(
    url: str,
    format: DownloadFormat = DownloadFormat.BOTH,
    quality: DownloadQuality = DownloadQuality.BEST,
    output_dir: str = ".",
    audio_format: str = "mp3",
    format_id: str | None = None,
    max_height: int | None = None,
) -> DownloadResult:
    """Download media from a URL using yt-dlp."""
    args = build_download_args(
        url=url,
        format=format,
        quality=quality,
        output_dir=output_dir,
        audio_format=audio_format,
        format_id=format_id,
        max_height=max_height,
    )
    try:
        output = _run_ytdlp(args)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode()
        raise DownloadError(f"Download failed: {stderr}") from e

    # Parse the JSON output to get file path and metadata
    data = json.loads(output.strip().splitlines()[-1])

    filepath = ""
    downloads = data.get("requested_downloads", [])
    if downloads:
        filepath = downloads[0].get("filepath", "")
    if not filepath:
        filepath = data.get("_filename", "")

    return DownloadResult(
        title=data.get("title", ""),
        url=data.get("webpage_url", url),
        filepath=filepath,
        format_id=data.get("format_id", ""),
        duration_seconds=data.get("duration", 0) or 0,
    )
