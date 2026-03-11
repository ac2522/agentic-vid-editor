"""Shot classification integration pipeline.

Connects scene detection -> frame extraction -> embedding -> classification -> registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ave.tools.scene import SceneBoundary, extract_keyframes
from ave.tools.vision import (
    FrameEmbedding,
    VisualAnalysis,
    save_analysis,
    tag_frames,
)

if TYPE_CHECKING:
    import numpy as np

    from ave.ingest.registry import AssetRegistry
    from ave.tools.vision import VisionBackend

SHOT_LABELS: list[str] = [
    "extreme wide shot",
    "wide shot",
    "medium wide shot",
    "medium shot",
    "medium close-up",
    "close-up",
    "extreme close-up",
    "over-the-shoulder",
    "two-shot",
    "insert shot",
    "aerial shot",
    "point-of-view",
]


class ClassificationError(Exception):
    """Raised when classification operations fail."""


def load_keyframe_as_array(path: Path) -> np.ndarray:
    """Load JPEG keyframe as (H, W, 3) uint8 numpy array.

    Uses PIL (Pillow) for loading.
    """
    import numpy as _np

    try:
        from PIL import Image
    except ImportError as e:
        raise ClassificationError("Pillow is required for keyframe loading") from e

    if not path.exists():
        raise ClassificationError(f"Keyframe not found: {path}")

    img = Image.open(path).convert("RGB")
    return _np.array(img, dtype=_np.uint8)


def classify_video(
    video_path: Path,
    scenes: list[SceneBoundary],
    backend: VisionBackend,
    output_dir: Path,
    asset_id: str = "unknown",
    labels: list[str] | None = None,
) -> VisualAnalysis:
    """Run the full classification pipeline on a video.

    For each scene: extract middle keyframe, embed, classify via zero-shot.

    Args:
        video_path: Path to source video.
        scenes: Detected scene boundaries.
        backend: Vision embedding backend (VisionBackend protocol).
        output_dir: Directory for keyframes and analysis output.
        asset_id: Identifier for this asset.
        labels: Custom label list; defaults to SHOT_LABELS.

    Returns:
        VisualAnalysis with embeddings and tags.

    Raises:
        ClassificationError: If scenes list is empty.
    """
    if not scenes:
        raise ClassificationError("Cannot classify video with empty scenes list")

    labels = labels or SHOT_LABELS

    # 1. Extract keyframes
    keyframe_dir = output_dir / "keyframes"
    keyframe_paths = extract_keyframes(video_path, scenes, keyframe_dir)

    # 2. Load keyframes as numpy arrays
    images = [load_keyframe_as_array(p) for p in keyframe_paths]

    # 3. Batch embed frames
    image_embeddings = backend.embed_batch(images)

    # 4. Build FrameEmbedding objects
    frame_embeddings: list[FrameEmbedding] = []
    for i, (scene, emb) in enumerate(zip(scenes, image_embeddings)):
        fe = FrameEmbedding(
            frame_idx=scene.mid_frame,
            timestamp_ns=(scene.start_ns + scene.end_ns) // 2,
            embedding=emb,
            scene_id=f"scene_{i:04d}",
        )
        frame_embeddings.append(fe)

    # 5. Embed label texts
    label_embeddings: dict[str, list[float]] = {}
    for label in labels:
        label_embeddings[label] = backend.embed_text(label)

    # 6. Zero-shot classification
    tags = tag_frames(frame_embeddings, label_embeddings)

    # 7. Build result
    analysis = VisualAnalysis(
        asset_id=asset_id,
        scenes=[s.model_dump() for s in scenes],
        frame_embeddings=frame_embeddings,
        tags=tags,
    )

    # 8. Save to JSON
    analysis_path = output_dir / "visual_analysis.json"
    save_analysis(analysis, analysis_path)

    return analysis


def classify_and_register(
    asset_id: str,
    registry: AssetRegistry,
    video_path: Path,
    scenes: list[SceneBoundary],
    backend: VisionBackend,
    output_dir: Path,
    labels: list[str] | None = None,
) -> VisualAnalysis:
    """Run classification and update the asset registry.

    Args:
        asset_id: Registry asset identifier.
        registry: AssetRegistry instance.
        video_path: Path to source video.
        scenes: Detected scene boundaries.
        backend: Vision embedding backend.
        output_dir: Directory for output files.
        labels: Optional custom labels.

    Returns:
        VisualAnalysis result.
    """
    analysis = classify_video(
        video_path=video_path,
        scenes=scenes,
        backend=backend,
        output_dir=output_dir,
        asset_id=asset_id,
        labels=labels,
    )

    # Update registry entry
    entry = registry.get(asset_id)
    entry.visual_analysis_path = output_dir / "visual_analysis.json"
    registry.save()

    return analysis
