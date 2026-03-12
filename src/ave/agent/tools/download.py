"""Agent tool registration for yt-dlp download operations."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_download_tools(registry: ToolRegistry) -> None:
    """Register download/search tools in the agent registry."""

    @registry.tool(
        domain="download",
        requires=[],
        provides=["search_results"],
        tags=[
            "youtube", "search video", "find song", "find music",
            "look up video", "search youtube", "find on youtube",
            "music search", "video search", "yt-dlp",
        ],
    )
    def search_youtube(query: str, max_results: int = 5) -> str:
        """Search YouTube for videos matching a query and return results."""
        from ave.tools.download_ops import search_youtube as _search

        results = _search(query, max_results)
        return "\n".join(
            f"{r.title} | {r.url} | {r.duration_seconds}s | {r.uploader}"
            for r in results
        )

    @registry.tool(
        domain="download",
        requires=[],
        provides=["formats_listed"],
        tags=[
            "available formats", "video quality", "audio quality",
            "what quality", "resolution options", "format list",
            "check quality", "available resolutions",
        ],
    )
    def list_available_formats(url: str) -> str:
        """List available download formats and qualities for a video URL."""
        from ave.tools.download_ops import list_formats as _list

        formats = _list(url)
        lines = []
        for f in formats:
            streams = []
            if f.has_video:
                streams.append(f"video:{f.vcodec}")
            if f.has_audio:
                streams.append(f"audio:{f.acodec}")
            size = f"{f.filesize // 1_000_000}MB" if f.filesize else "?"
            lines.append(
                f"{f.format_id} | {f.ext} | {f.resolution} | "
                f"{'+'.join(streams)} | {size} | {f.format_note}"
            )
        return "\n".join(lines)

    @registry.tool(
        domain="download",
        requires=[],
        provides=["media_downloaded"],
        tags=[
            "download video", "download audio", "download song",
            "download music", "download from youtube", "yt-dlp",
            "rip audio", "extract audio", "save video", "grab video",
            "get song", "get video", "fetch video", "fetch audio",
            "highest quality", "lowest quality", "best quality",
            "mp3", "flac", "wav", "audio only", "video only",
        ],
    )
    def download_media(
        url: str,
        download_format: str = "both",
        quality: str = "best",
        output_dir: str = ".",
        audio_format: str = "mp3",
        format_id: str = "",
        max_height: int = 0,
    ) -> str:
        """Download video/audio from YouTube or other supported sites.

        Args:
            url: Video URL or search query.
            download_format: What to download — 'video', 'audio', or 'both'.
            quality: Quality level — 'best' or 'worst'.
            output_dir: Directory to save the downloaded file.
            audio_format: Audio format when extracting audio (mp3, flac, wav, m4a, opus).
            format_id: Specific format ID from list_available_formats (overrides quality).
            max_height: Max video height in pixels (e.g. 720, 1080). 0 = no limit.
        """
        from ave.tools.download import DownloadFormat, DownloadQuality
        from ave.tools.download_ops import download_media as _download

        fmt = DownloadFormat(download_format)
        qual = DownloadQuality(quality)

        result = _download(
            url=url,
            format=fmt,
            quality=qual,
            output_dir=output_dir,
            audio_format=audio_format,
            format_id=format_id or None,
            max_height=max_height or None,
        )

        return (
            f"Downloaded: {result.title}\n"
            f"File: {result.filepath}\n"
            f"Format: {result.format_id}\n"
            f"Duration: {result.duration_seconds}s"
        )
