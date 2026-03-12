"""Interchange domain tool registration."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_interchange_tools(registry: ToolRegistry) -> None:
    """Register interchange domain tools."""

    @registry.tool(
        domain="interchange",
        requires=["timeline_loaded"],
        provides=["otio_exported"],
        tags=["export timeline", "OpenTimelineIO", "OTIO", "FCP XML", "EDL",
              "DaVinci Resolve", "Premiere", "Final Cut", "round trip",
              "interchange"],
    )
    def export_otio(
        timeline_data_json: str,
        output_path: str,
        fps: float = 24.0,
    ):
        """Export the current timeline to OpenTimelineIO format."""
        import json
        from pathlib import Path

        from ave.interchange.otio_export import export_timeline

        timeline_data = json.loads(timeline_data_json)
        result_path = export_timeline(timeline_data, Path(output_path), fps)
        return {"output_path": str(result_path)}

    @registry.tool(
        domain="interchange",
        requires=[],
        provides=["timeline_loaded"],
        tags=["import timeline", "load edit", "open project", "import EDL",
              "import FCP XML", "load OTIO", "import from DaVinci",
              "import from Premiere"],
    )
    def import_otio(otio_path: str):
        """Import a timeline from an OpenTimelineIO file."""
        from pathlib import Path

        from ave.interchange.otio_import import import_timeline

        result = import_timeline(Path(otio_path))
        return result
