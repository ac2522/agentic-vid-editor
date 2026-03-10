"""Render domain tool registration."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_render_tools(registry: ToolRegistry) -> None:
    """Register render domain tools."""

    @registry.tool(
        domain="render",
        requires=["timeline_loaded"],
        provides=["proxy_rendered"],
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
    )
    def compute_segments(duration_ns: int, segment_duration_ns: int = 5_000_000_000):
        """Compute segment boundaries for splitting a timeline into chunks."""
        from ave.render.segment import compute_segment_boundaries

        return compute_segment_boundaries(duration_ns, segment_duration_ns)
