"""Tests for shot classification integration pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from ave.tools.classify import (  # noqa: E402
    SHOT_LABELS,
    ClassificationError,
    classify_and_register,
    classify_video,
    load_keyframe_as_array,
)
from ave.tools.scene import SceneBoundary  # noqa: E402
from ave.tools.vision import SceneTag, VisualAnalysis  # noqa: E402


# ---------------------------------------------------------------------------
# Mock VisionBackend
# ---------------------------------------------------------------------------


class MockVisionBackend:
    """Deterministic mock backend for testing the classification pipeline."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed_image(self, image: np.ndarray) -> list[float]:
        # Return a vector seeded by the mean pixel value for slight variation
        seed = float(image.mean()) / 255.0
        return [0.1 + seed * 0.01 * i for i in range(self.dim)]

    def embed_text(self, text: str) -> list[float]:
        # Return a vector seeded by the hash of the text
        seed = hash(text) % 1000 / 1000.0
        return [0.2 + seed * 0.01 * i for i in range(self.dim)]

    def embed_batch(self, images: list[np.ndarray]) -> list[list[float]]:
        return [self.embed_image(img) for img in images]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_backend() -> MockVisionBackend:
    return MockVisionBackend()


@pytest.fixture()
def sample_scenes() -> list[SceneBoundary]:
    """Two scenes at 24 fps."""
    return [
        SceneBoundary(start_ns=0, end_ns=2_000_000_000, fps=24.0),
        SceneBoundary(start_ns=2_000_000_000, end_ns=4_000_000_000, fps=24.0),
    ]


@pytest.fixture()
def keyframes_dir(tmp_path: Path, sample_scenes: list[SceneBoundary]) -> Path:
    """Create fake JPEG keyframe files (small valid images)."""
    from PIL import Image

    kf_dir = tmp_path / "keyframes"
    kf_dir.mkdir()
    for i in range(len(sample_scenes)):
        img = Image.new("RGB", (64, 48), color=(100 + i * 30, 50, 200))
        img.save(kf_dir / f"keyframe_{i:04d}.jpg")
    return kf_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShotLabels:
    def test_at_least_10_entries(self):
        assert len(SHOT_LABELS) >= 10

    def test_contains_expected_labels(self):
        assert "close-up" in SHOT_LABELS
        assert "wide shot" in SHOT_LABELS
        assert "medium shot" in SHOT_LABELS


class TestClassificationError:
    def test_is_exception(self):
        with pytest.raises(ClassificationError):
            raise ClassificationError("test error")


class TestLoadKeyframeAsArray:
    def test_loads_jpeg(self, tmp_path: Path):
        from PIL import Image

        img = Image.new("RGB", (80, 60), color=(255, 0, 0))
        path = tmp_path / "frame.jpg"
        img.save(path)

        arr = load_keyframe_as_array(path)
        assert arr.shape == (60, 80, 3)
        assert arr.dtype == np.uint8

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(ClassificationError):
            load_keyframe_as_array(tmp_path / "nonexistent.jpg")


class TestClassifyVideo:
    def test_produces_visual_analysis(
        self,
        tmp_path: Path,
        mock_backend: MockVisionBackend,
        sample_scenes: list[SceneBoundary],
        keyframes_dir: Path,
        monkeypatch,
    ):
        """classify_video produces a VisualAnalysis with correct scene count."""
        # Monkeypatch extract_keyframes to avoid needing FFmpeg
        keyframe_paths = sorted(keyframes_dir.glob("keyframe_*.jpg"))

        monkeypatch.setattr(
            "ave.tools.classify.extract_keyframes",
            lambda video_path, boundaries, output_dir, strategy="middle": keyframe_paths,
        )

        output_dir = tmp_path / "output"
        result = classify_video(
            video_path=Path("/fake/video.mp4"),
            scenes=sample_scenes,
            backend=mock_backend,
            output_dir=output_dir,
            asset_id="test-asset",
        )

        assert isinstance(result, VisualAnalysis)
        assert len(result.frame_embeddings) == 2
        assert len(result.tags) == 2

    def test_each_scene_gets_nonempty_labels(
        self,
        tmp_path: Path,
        mock_backend: MockVisionBackend,
        sample_scenes: list[SceneBoundary],
        keyframes_dir: Path,
        monkeypatch,
    ):
        keyframe_paths = sorted(keyframes_dir.glob("keyframe_*.jpg"))
        monkeypatch.setattr(
            "ave.tools.classify.extract_keyframes",
            lambda video_path, boundaries, output_dir, strategy="middle": keyframe_paths,
        )

        result = classify_video(
            video_path=Path("/fake/video.mp4"),
            scenes=sample_scenes,
            backend=mock_backend,
            output_dir=tmp_path / "out",
            asset_id="test-asset",
        )

        for tag in result.tags:
            assert isinstance(tag, SceneTag)
            assert len(tag.labels) > 0
            # All scores should be finite floats
            for score in tag.labels.values():
                assert isinstance(score, float)

    def test_custom_labels(
        self,
        tmp_path: Path,
        mock_backend: MockVisionBackend,
        sample_scenes: list[SceneBoundary],
        keyframes_dir: Path,
        monkeypatch,
    ):
        keyframe_paths = sorted(keyframes_dir.glob("keyframe_*.jpg"))
        monkeypatch.setattr(
            "ave.tools.classify.extract_keyframes",
            lambda video_path, boundaries, output_dir, strategy="middle": keyframe_paths,
        )

        custom_labels = ["daytime", "nighttime", "indoor", "outdoor"]
        result = classify_video(
            video_path=Path("/fake/video.mp4"),
            scenes=sample_scenes,
            backend=mock_backend,
            output_dir=tmp_path / "out",
            asset_id="test-asset",
            labels=custom_labels,
        )

        # Each tag should only contain the custom labels
        for tag in result.tags:
            assert set(tag.labels.keys()) == set(custom_labels)

    def test_empty_scenes_raises(self, tmp_path: Path, mock_backend: MockVisionBackend):
        with pytest.raises(ClassificationError):
            classify_video(
                video_path=Path("/fake/video.mp4"),
                scenes=[],
                backend=mock_backend,
                output_dir=tmp_path / "out",
                asset_id="test-asset",
            )

    def test_saves_analysis_json(
        self,
        tmp_path: Path,
        mock_backend: MockVisionBackend,
        sample_scenes: list[SceneBoundary],
        keyframes_dir: Path,
        monkeypatch,
    ):
        keyframe_paths = sorted(keyframes_dir.glob("keyframe_*.jpg"))
        monkeypatch.setattr(
            "ave.tools.classify.extract_keyframes",
            lambda video_path, boundaries, output_dir, strategy="middle": keyframe_paths,
        )

        output_dir = tmp_path / "output"
        classify_video(
            video_path=Path("/fake/video.mp4"),
            scenes=sample_scenes,
            backend=mock_backend,
            output_dir=output_dir,
            asset_id="test-asset",
        )

        analysis_path = output_dir / "visual_analysis.json"
        assert analysis_path.exists()


class TestClassifyAndRegister:
    def test_updates_registry(
        self,
        tmp_path: Path,
        mock_backend: MockVisionBackend,
        sample_scenes: list[SceneBoundary],
        keyframes_dir: Path,
        monkeypatch,
    ):
        from ave.ingest.registry import AssetEntry, AssetRegistry

        keyframe_paths = sorted(keyframes_dir.glob("keyframe_*.jpg"))
        monkeypatch.setattr(
            "ave.tools.classify.extract_keyframes",
            lambda video_path, boundaries, output_dir, strategy="middle": keyframe_paths,
        )

        registry = AssetRegistry(tmp_path / "registry.json")
        entry = AssetEntry(
            asset_id="vid-001",
            original_path=Path("/fake/original.mp4"),
            working_path=Path("/fake/working.mp4"),
            proxy_path=Path("/fake/proxy.mp4"),
            original_fps=24.0,
            conformed_fps=24.0,
            duration_seconds=4.0,
            width=1920,
            height=1080,
            codec="h264",
            camera_color_space="sRGB",
            camera_transfer="sRGB",
        )
        registry.add(entry)

        output_dir = tmp_path / "output"
        result = classify_and_register(
            asset_id="vid-001",
            registry=registry,
            video_path=Path("/fake/video.mp4"),
            scenes=sample_scenes,
            backend=mock_backend,
            output_dir=output_dir,
        )

        assert isinstance(result, VisualAnalysis)
        updated = registry.get("vid-001")
        assert updated.visual_analysis_path is not None
        assert updated.visual_analysis_path.exists()
