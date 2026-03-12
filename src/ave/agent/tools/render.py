"""Render domain tool registration."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_render_tools(registry: ToolRegistry) -> None:
    """Register render domain tools."""

    @registry.tool(
        domain="render",
        requires=["timeline_loaded"],
        provides=["proxy_rendered"],
        tags=["preview render", "low res", "offline", "proxy file",
              "lightweight render", "draft"],
    )
    def render_proxy(xges_path: str, output_path: str, height: int = 480):
        """Render an XGES timeline to an H.264 proxy MP4."""
        from pathlib import Path

        from ave.render.proxy import render_proxy as _render_proxy

        return _render_proxy(Path(xges_path), Path(output_path), height)

    @registry.tool(
        domain="render",
        requires=["timeline_loaded"],
        provides=["segment_rendered"],
        tags=["chunk render", "partial render", "segment export",
              "fragmented mp4"],
    )
    def render_segment(
        xges_path: str,
        output_path: str,
        start_ns: int,
        end_ns: int,
        height: int = 480,
    ):
        """Render a segment of a timeline to fragmented MP4."""
        from pathlib import Path

        from ave.render.segment import render_segment as _render_segment

        return _render_segment(Path(xges_path), Path(output_path), start_ns, end_ns, height)

    @registry.tool(
        domain="render",
        requires=["timeline_loaded"],
        provides=["segments_computed"],
        tags=["split timeline", "chunk boundaries", "render planning",
              "parallel render"],
    )
    def compute_segments(duration_ns: int, segment_duration_ns: int = 5_000_000_000):
        """Compute segment boundaries for splitting a timeline into chunks."""
        from ave.render.segment import compute_segment_boundaries

        return compute_segment_boundaries(duration_ns, segment_duration_ns)

    @registry.tool(
        domain="render",
        requires=["timeline_loaded"],
        provides=["preset_rendered"],
        tags=["export preset", "youtube", "instagram", "prores", "render preset",
              "output format", "delivery"],
    )
    def render_with_preset(xges_path: str, preset_name: str, output_path: str) -> str:
        """Render timeline using a named preset (e.g. youtube_4k, instagram_reel, prores_master)."""
        import dataclasses
        import json

        from ave.render.presets import get_preset, validate_preset

        preset = get_preset(preset_name)
        errors = validate_preset(preset)
        if errors:
            raise ValueError(f"Invalid preset '{preset_name}': {'; '.join(errors)}")
        return json.dumps(dataclasses.asdict(preset))

    @registry.tool(
        domain="render",
        requires=[],
        provides=[],
        tags=["available presets", "render options", "export formats",
              "output presets", "show presets"],
    )
    def list_render_presets() -> str:
        """List all available render presets with their descriptions."""
        import json

        from ave.render.presets import list_presets

        return json.dumps(list_presets())
