"""Import OpenTimelineIO files into AVE timeline data format.

Converts OTIO timelines to AVE's internal dict representation with all
times in nanoseconds. Uses lazy imports so opentimelineio is only required
at call time.
"""

from __future__ import annotations

from pathlib import Path

SUPPORTED_IMPORT_FORMATS = {".otio", ".fcpxml", ".edl", ".aaf"}

_NS_PER_SECOND = 1_000_000_000


class OTIOImportError(Exception):
    """Raised when OTIO import fails."""


def rational_time_to_ns(rt: "otio.opentime.RationalTime") -> int:  # noqa: F821
    """Convert OTIO RationalTime to nanoseconds."""
    seconds = rt.value / rt.rate
    return round(seconds * _NS_PER_SECOND)


def time_range_to_ns(tr: "otio.opentime.TimeRange") -> tuple[int, int]:  # noqa: F821
    """Convert OTIO TimeRange to (start_ns, duration_ns)."""
    start_ns = rational_time_to_ns(tr.start_time)
    duration_ns = rational_time_to_ns(tr.duration)
    return start_ns, duration_ns


def otio_clip_to_dict(clip: "otio.schema.Clip") -> dict:  # noqa: F821
    """Convert OTIO Clip to AVE clip dict.

    Returns:
        {name, source_path, start_ns, duration_ns, in_point_ns}
    """
    import opentimelineio as otio  # noqa: F811

    name = clip.name or ""

    # Extract source path from media reference
    source_path = ""
    if clip.media_reference and not isinstance(
        clip.media_reference, otio.schema.MissingReference
    ):
        if hasattr(clip.media_reference, "target_url"):
            url = clip.media_reference.target_url
            # Strip file:// prefix if present
            if url.startswith("file://"):
                source_path = url[7:]
            else:
                source_path = url

    # Timeline position
    if clip.range_in_parent() is not None:
        start_ns, duration_ns = time_range_to_ns(clip.range_in_parent())
    else:
        start_ns = 0
        duration_ns = 0

    # In-point from source range
    in_point_ns = 0
    if clip.source_range is not None:
        in_point_ns = rational_time_to_ns(clip.source_range.start_time)

    return {
        "name": name,
        "source_path": source_path,
        "start_ns": start_ns,
        "duration_ns": duration_ns,
        "in_point_ns": in_point_ns,
    }


def otio_track_to_layer(
    track: "otio.schema.Track", layer_index: int  # noqa: F821
) -> tuple[dict, list[str]]:
    """Convert OTIO Track to AVE layer dict.

    Returns:
        Tuple of (layer_dict, warnings) where layer_dict is
        {layer_index, clips: list[dict]}
    """
    import opentimelineio as otio  # noqa: F811

    clips: list[dict] = []
    warnings: list[str] = []

    for child in track.each_child():
        if isinstance(child, otio.schema.Clip):
            # Check for generator references (color bars, etc.)
            if child.media_reference and isinstance(
                child.media_reference, otio.schema.GeneratorReference
            ):
                warnings.append(
                    f"Skipped generator clip '{child.name}': "
                    f"generator clips are not supported"
                )
                continue
            clips.append(otio_clip_to_dict(child))
        elif isinstance(child, (otio.schema.Stack, otio.schema.Track)):
            # Nested composition — flatten with warning
            warnings.append(
                f"Flattened nested composition '{child.name}' in track "
                f"'{track.name}'"
            )
            nested_layer, nested_warnings = otio_track_to_layer(
                child, layer_index
            )
            clips.extend(nested_layer["clips"])
            warnings.extend(nested_warnings)
        elif isinstance(child, otio.schema.Gap):
            # Gaps are implicit in AVE (clips just have start positions)
            pass
        elif isinstance(child, otio.schema.Transition):
            warnings.append(
                f"Skipped transition '{child.name}': "
                f"transitions are not directly importable"
            )

    # Collect effect warnings from clips that have effects
    for child in track.each_child():
        if isinstance(child, otio.schema.Clip) and child.effects:
            for effect in child.effects:
                warnings.append(
                    f"Skipped effect '{effect.name}' on clip "
                    f"'{child.name}': effects are not importable"
                )

    return {"layer_index": layer_index, "clips": clips}, warnings


def import_timeline(otio_path: Path) -> dict:
    """Import OTIO file to AVE timeline data dict.

    Returns:
        {
            "name": str,
            "duration_ns": int,
            "layers": list[dict],
            "warnings": list[str],
        }

    Raises:
        OTIOImportError on failure.
    """
    otio_path = Path(otio_path)

    if not otio_path.exists():
        raise OTIOImportError(f"File not found: {otio_path}")

    suffix = otio_path.suffix.lower()
    if suffix not in SUPPORTED_IMPORT_FORMATS:
        raise OTIOImportError(
            f"Unsupported format '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_IMPORT_FORMATS))}"
        )

    try:
        import opentimelineio as otio  # noqa: F811
    except ImportError:
        raise OTIOImportError(
            "opentimelineio is not installed. "
            "Install it with: pip install opentimelineio"
        )

    try:
        timeline = otio.adapters.read_from_file(str(otio_path))
    except Exception as exc:
        raise OTIOImportError(f"Failed to read '{otio_path}': {exc}") from exc

    name = timeline.name or otio_path.stem
    layers: list[dict] = []
    all_warnings: list[str] = []

    for i, track in enumerate(timeline.tracks):
        layer, warnings = otio_track_to_layer(track, i)
        layers.append(layer)
        all_warnings.extend(warnings)

    # Compute total duration from all clips across layers
    max_end_ns = 0
    for layer in layers:
        for clip in layer["clips"]:
            end = clip["start_ns"] + clip["duration_ns"]
            if end > max_end_ns:
                max_end_ns = end

    return {
        "name": name,
        "duration_ns": max_end_ns,
        "layers": layers,
        "warnings": all_warnings,
    }
