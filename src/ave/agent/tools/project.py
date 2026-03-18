"""Project domain tool registration."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_project_tools(registry: ToolRegistry) -> None:
    """Register project domain tools."""

    @registry.tool(
        domain="project",
        requires=[],
        provides=["media_probed"],
        tags=[
            "media info",
            "file info",
            "codec info",
            "resolution",
            "duration",
            "framerate",
            "metadata",
            "inspect file",
            "video info",
        ],
    )
    def probe_media(path: str):
        """Probe a media file and return structured metadata (codec, resolution, duration)."""
        from pathlib import Path

        from ave.ingest.probe import probe_media as _probe_media

        return _probe_media(Path(path))

    @registry.tool(
        domain="project",
        requires=["media_probed"],
        provides=["media_ingested"],
        tags=[
            "import",
            "add media",
            "bring in",
            "load footage",
            "add clip",
            "register asset",
            "import footage",
        ],
    )
    def ingest_media(
        source: str,
        project_dir: str,
        asset_id: str,
        registry_path: str,
        project_fps: float = 24.0,
        codec: str = "dnxhd",
        profile: str = "dnxhr_hqx",
    ):
        """Ingest a media file: probe, transcode to working + proxy, and register."""
        from pathlib import Path

        from ave.ingest.registry import AssetRegistry
        from ave.ingest.transcoder import ingest as _ingest

        asset_registry = AssetRegistry(Path(registry_path))
        return _ingest(
            source=Path(source),
            project_dir=Path(project_dir),
            asset_id=asset_id,
            registry=asset_registry,
            project_fps=project_fps,
            codec=codec,
            profile=profile,
        )

    @registry.tool(
        domain="project",
        requires=["timeline_loaded"],
        provides=["clip_exists"],
        tags=[
            "clip list",
            "clips",
            "timeline clips",
            "clip info",
            "clip names",
            "what clips",
            "show clips",
            "clip inventory",
            "clip count",
            "which clips",
            "list all clips",
        ],
    )
    def list_clips(xges_path: str) -> list:
        """List all clips on the timeline with IDs, positions, durations, and track info."""
        from ave.tools.timeline_info import list_timeline_clips

        return list_timeline_clips(xges_path)

    @registry.tool(
        domain="project",
        requires=["timeline_loaded"],
        provides=[],
        tags=[
            "timeline info",
            "timeline duration",
            "timeline summary",
            "how long",
            "framerate",
            "fps",
            "layer count",
            "project info",
        ],
    )
    def timeline_info(xges_path: str) -> dict:
        """Get timeline summary: duration, framerate, layer count, clip count."""
        from ave.tools.timeline_info import get_timeline_info

        return get_timeline_info(xges_path)
