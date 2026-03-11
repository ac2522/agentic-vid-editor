"""Rough cut tools — scene-to-timeline rough cut workflow.

Pure logic layer: computes clip placements from scene detection results.
No GES dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from ave.tools.scene import SceneBoundary
from ave.tools.vision import SceneTag


class RoughCutError(Exception):
    """Raised when rough cut parameter validation fails."""


@dataclass(frozen=True)
class RoughCutParams:
    """Parameters for computing a rough cut."""

    scenes: list[SceneBoundary]
    selected_indices: list[int]
    order: str  # "chronological" | "custom"
    gap_ns: int = 0


@dataclass(frozen=True)
class ClipPlacement:
    """A single clip placement on the timeline."""

    source_start_ns: int
    source_end_ns: int
    timeline_position_ns: int
    scene_index: int


def select_scenes_by_tags(
    scenes: list[SceneBoundary],
    tags: list[SceneTag],
    include_labels: set[str] | None = None,
    exclude_labels: set[str] | None = None,
) -> list[int]:
    """Filter scenes by their classification labels.

    Args:
        scenes: Scene boundaries.
        tags: Scene tags (one per scene, matched by index).
        include_labels: If set, only include scenes whose top_label is in this set.
        exclude_labels: If set, exclude scenes whose top_label is in this set.
            include_labels takes priority over exclude_labels for the same label.

    Returns:
        Sorted list of scene indices that pass the filter.
    """
    result: list[int] = []
    for i, tag in enumerate(tags):
        if i >= len(scenes):
            break
        label = tag.top_label

        if include_labels is not None:
            # When include_labels is specified, only keep scenes with matching label
            if label in include_labels:
                result.append(i)
            continue

        if exclude_labels is not None and label in exclude_labels:
            continue

        result.append(i)

    return result


def select_scenes_by_duration(
    scenes: list[SceneBoundary],
    min_duration_ns: int | None = None,
    max_duration_ns: int | None = None,
) -> list[int]:
    """Filter scenes by duration range.

    Args:
        scenes: Scene boundaries.
        min_duration_ns: Minimum duration (inclusive). None means no minimum.
        max_duration_ns: Maximum duration (inclusive). None means no maximum.

    Returns:
        Sorted list of scene indices within the duration range.
    """
    result: list[int] = []
    for i, scene in enumerate(scenes):
        dur = scene.duration_ns
        if min_duration_ns is not None and dur < min_duration_ns:
            continue
        if max_duration_ns is not None and dur > max_duration_ns:
            continue
        result.append(i)
    return result


def compute_rough_cut(params: RoughCutParams) -> list[ClipPlacement]:
    """Compute clip placements for a rough cut.

    Args:
        params: Rough cut parameters.

    Returns:
        List of ClipPlacement with computed timeline positions.

    Raises:
        RoughCutError: If parameters are invalid.
    """
    if not params.scenes:
        raise RoughCutError("Cannot compute rough cut: scenes list is empty")

    if not params.selected_indices:
        raise RoughCutError("Cannot compute rough cut: selected_indices is empty")

    num_scenes = len(params.scenes)
    for idx in params.selected_indices:
        if idx < 0 or idx >= num_scenes:
            raise RoughCutError(
                f"Scene index {idx} out of range [0, {num_scenes})"
            )

    # Determine clip order
    if params.order == "chronological":
        ordered = sorted(
            params.selected_indices,
            key=lambda i: params.scenes[i].start_ns,
        )
    else:
        ordered = list(params.selected_indices)

    # Build placements
    placements: list[ClipPlacement] = []
    timeline_pos = 0

    for i, scene_idx in enumerate(ordered):
        scene = params.scenes[scene_idx]
        placements.append(
            ClipPlacement(
                source_start_ns=scene.start_ns,
                source_end_ns=scene.end_ns,
                timeline_position_ns=timeline_pos,
                scene_index=scene_idx,
            )
        )
        timeline_pos += scene.duration_ns
        if i < len(ordered) - 1:
            timeline_pos += params.gap_ns

    return placements
