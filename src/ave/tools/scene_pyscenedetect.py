"""PySceneDetect backend for scene boundary detection.

Conditional import of scenedetect — only required when this backend is used.
"""

from __future__ import annotations

from pathlib import Path

from ave.tools.scene import SceneBoundary, SceneError


class PySceneDetectBackend:
    """Scene detection using PySceneDetect library.

    Supports multiple detection algorithms: content, adaptive, threshold, hash.
    Returns scene boundaries with nanosecond timestamps.
    """

    def detect_scenes(
        self,
        video_path: Path,
        threshold: float = 27.0,
        detector: str = "content",
    ) -> list[SceneBoundary]:
        """Detect scene boundaries in a video file.

        Args:
            video_path: Path to video file.
            threshold: Detection sensitivity (meaning varies by detector).
            detector: Algorithm — "content", "adaptive", "threshold", "hash".

        Returns:
            List of SceneBoundary with nanosecond timestamps.

        Raises:
            SceneError: If scenedetect is not installed or detector is unknown.
        """
        try:
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import (
                AdaptiveDetector,
                ContentDetector,
                HashDetector,
                ThresholdDetector,
            )
        except ImportError:
            raise SceneError(
                "PySceneDetect not installed. Install with: pip install scenedetect[opencv]"
            )

        detectors = {
            "content": lambda: ContentDetector(threshold=threshold),
            "adaptive": lambda: AdaptiveDetector(adaptive_threshold=threshold),
            "threshold": lambda: ThresholdDetector(threshold=threshold),
            "hash": lambda: HashDetector(threshold=threshold),
        }

        if detector not in detectors:
            raise SceneError(
                f"Unknown detector: {detector}. Valid options: {', '.join(detectors.keys())}"
            )

        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(detectors[detector]())
        scene_manager.detect_scenes(video)

        scene_list = scene_manager.get_scene_list()
        fps = video.frame_rate

        boundaries = []
        for start_tc, end_tc in scene_list:
            start_ns = int(start_tc.get_seconds() * 1_000_000_000)
            end_ns = int(end_tc.get_seconds() * 1_000_000_000)
            boundaries.append(SceneBoundary(start_ns=start_ns, end_ns=end_ns, fps=fps))

        # If no scenes detected, return the whole video as one scene
        if not boundaries:
            duration_ns = int(video.duration.get_seconds() * 1_000_000_000)
            boundaries.append(SceneBoundary(start_ns=0, end_ns=duration_ns, fps=fps))

        return boundaries
