"""VFX domain tools — rotoscoping, keying, mask evaluation."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_vfx_tools(registry: ToolRegistry) -> None:
    """Register VFX domain tools."""

    @registry.tool(
        domain="vfx",
        tags=["segment", "rotoscope", "mask", "matte", "key", "cut out"],
    )
    def segment_video(
        asset_path: str,
        prompts: list[dict] | None = None,
        backend: str = "auto",
    ) -> dict:
        """Segment objects in video using AI or chroma key. Returns mask asset path.

        Args:
            asset_path: Path to video file.
            prompts: List of prompt dicts with 'kind' and 'value' keys.
            backend: "sam2", "rvm", "chroma", or "auto" (default).
        """
        return {
            "asset_path": asset_path,
            "prompts": prompts or [],
            "backend": backend,
        }

    @registry.tool(
        domain="vfx",
        tags=["refine", "mask", "correction", "fix", "adjust"],
    )
    def refine_mask(
        mask_path: str,
        corrections: list[dict] | None = None,
        frames: list[int] | None = None,
    ) -> dict:
        """Refine segmentation mask at specific frames.

        Args:
            mask_path: Path to mask file.
            corrections: List of correction dicts with 'kind' and 'value'.
            frames: Specific frame indices to refine (None = all).
        """
        return {
            "mask_path": mask_path,
            "corrections": corrections or [],
            "frames": frames,
        }

    @registry.tool(
        domain="vfx",
        tags=["evaluate", "quality", "mask", "check", "assess"],
    )
    def evaluate_mask(mask_path: str, asset_path: str) -> dict:
        """Assess mask quality. Returns MaskQuality metrics.

        Args:
            mask_path: Path to mask file.
            asset_path: Path to original video for comparison.
        """
        return {"mask_path": mask_path, "asset_path": asset_path}

    @registry.tool(
        domain="vfx",
        modifies_timeline=True,
        tags=["apply", "mask", "composite", "remove background", "key"],
    )
    def apply_mask(
        clip_id: str,
        mask_path: str,
        operation: str = "remove_background",
    ) -> dict:
        """Apply mask to timeline clip.

        Args:
            clip_id: ID of the clip on the timeline.
            mask_path: Path to the mask file.
            operation: "remove_background", "composite", or "replace".
        """
        return {
            "clip_id": clip_id,
            "mask_path": mask_path,
            "operation": operation,
        }
