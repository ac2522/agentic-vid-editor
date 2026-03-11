"""Scene domain tool registration."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_scene_tools(registry: ToolRegistry) -> None:
    """Register scene domain tools."""

    @registry.tool(
        domain="scene",
        requires=["timeline_loaded"],
        provides=["scenes_detected"],
    )
    def detect_scenes(
        video_path: str,
        threshold: float = 27.0,
        detector: str = "content",
    ):
        """Detect scene boundaries in a video using configurable algorithms."""
        from pathlib import Path

        from ave.tools.scene_pyscenedetect import PySceneDetectBackend

        backend = PySceneDetectBackend()
        boundaries = backend.detect_scenes(
            video_path=Path(video_path),
            threshold=threshold,
            detector=detector,
        )
        return [b.model_dump() for b in boundaries]

    @registry.tool(
        domain="scene",
        requires=["scenes_detected"],
        provides=["shots_classified"],
    )
    def classify_shots(
        video_path: str,
        scenes_json: str,
        output_dir: str,
    ):
        """Classify detected scenes into shot types (wide, close-up, etc.)."""
        from pathlib import Path

        return {
            "video_path": video_path,
            "scenes_json": scenes_json,
            "output_dir": output_dir,
            "status": "classification_pending",
        }

    @registry.tool(
        domain="scene",
        requires=["scenes_detected"],
        provides=["rough_cut_created"],
    )
    def create_rough_cut(
        scenes_json: str,
        selected_indices: str,
        order: str = "chronological",
        gap_ns: int = 0,
    ):
        """Create a rough cut from selected scene indices."""
        import json

        from ave.tools.rough_cut import RoughCutParams, compute_rough_cut
        from ave.tools.scene import SceneBoundary

        scenes_data = json.loads(scenes_json)
        scenes = [SceneBoundary(**s) for s in scenes_data]
        indices = json.loads(selected_indices)

        params = RoughCutParams(
            scenes=scenes,
            selected_indices=indices,
            order=order,
            gap_ns=gap_ns,
        )
        placements = compute_rough_cut(params)
        return [
            {
                "source_start_ns": p.source_start_ns,
                "source_end_ns": p.source_end_ns,
                "timeline_position_ns": p.timeline_position_ns,
                "scene_index": p.scene_index,
            }
            for p in placements
        ]
