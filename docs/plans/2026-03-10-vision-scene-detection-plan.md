# Vision & Scene Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add modular scene detection (PySceneDetect) and vision embedding (SigLIP 2) tools with protocol-based swappable backends.

**Architecture:** Protocol-based interfaces in `vision.py` and `scene.py` define what any backend must implement. Concrete backends (`vision_siglip2.py`, `scene_pyscenedetect.py`) implement those contracts. Pure logic (cosine similarity, tag scoring, boundary math) is separated from I/O and backend calls, testable without any dependencies.

**Tech Stack:** PySceneDetect (scene cuts), SigLIP 2 via HuggingFace transformers + ONNX Runtime (vision embeddings), Pydantic (data models), NumPy (vector math)

---

### Task 1: Data Models & Pure Logic — Scene Boundaries

**Files:**
- Create: `src/ave/tools/scene.py`
- Create: `tests/test_tools/test_scene.py`

**Step 1: Write failing tests for SceneBoundary model**

```python
# tests/test_tools/test_scene.py
"""Unit tests for scene detection — data models and pure logic."""

from ave.tools.scene import SceneBoundary, SceneError


class TestSceneBoundary:
    def test_create_boundary(self):
        b = SceneBoundary(start_ns=0, end_ns=2_000_000_000, fps=24.0)
        assert b.start_ns == 0
        assert b.end_ns == 2_000_000_000
        assert b.fps == 24.0

    def test_duration_ns(self):
        b = SceneBoundary(start_ns=1_000_000_000, end_ns=3_000_000_000, fps=24.0)
        assert b.duration_ns == 2_000_000_000

    def test_start_frame_derived(self):
        b = SceneBoundary(start_ns=0, end_ns=1_000_000_000, fps=24.0)
        assert b.start_frame == 0

    def test_end_frame_derived(self):
        b = SceneBoundary(start_ns=0, end_ns=1_000_000_000, fps=24.0)
        assert b.end_frame == 24

    def test_mid_frame_derived(self):
        b = SceneBoundary(start_ns=0, end_ns=2_000_000_000, fps=24.0)
        assert b.mid_frame == 24  # middle of 0-48

    def test_boundary_with_offset(self):
        b = SceneBoundary(start_ns=5_000_000_000, end_ns=7_000_000_000, fps=30.0)
        assert b.start_frame == 150
        assert b.end_frame == 210

    def test_metadata_key_constants_exist(self):
        from ave.tools.scene import AGENT_META_SCENE_ID
        assert AGENT_META_SCENE_ID == "agent:scene-id"
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tools/test_scene.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ave.tools.scene'`

**Step 3: Implement SceneBoundary and constants**

```python
# src/ave/tools/scene.py
"""Scene detection tools — shot boundary detection with swappable backends.

Pure logic layer for data models and boundary computation.
Scene detection engine integration is conditional.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    pass


class SceneError(Exception):
    """Raised when scene detection fails."""


# GES metadata key constants
AGENT_META_SCENE_ID = "agent:scene-id"
AGENT_META_SCENE_START = "agent:scene-start-ns"
AGENT_META_SCENE_END = "agent:scene-end-ns"


class SceneBoundary(BaseModel):
    """A detected scene/shot boundary with timestamps.

    Nanosecond timestamps are authoritative. Frame numbers are derived.
    """

    start_ns: int
    end_ns: int
    fps: float

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def start_frame(self) -> int:
        return round(self.start_ns * self.fps / 1_000_000_000)

    @property
    def end_frame(self) -> int:
        return round(self.end_ns * self.fps / 1_000_000_000)

    @property
    def mid_frame(self) -> int:
        mid_ns = (self.start_ns + self.end_ns) // 2
        return round(mid_ns * self.fps / 1_000_000_000)


class SceneBackend(Protocol):
    """Protocol for scene detection backends. Type-annotation only."""

    def detect_scenes(
        self,
        video_path: Path,
        threshold: float,
        detector: str = "content",
    ) -> list[SceneBoundary]: ...
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tools/test_scene.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/ave/tools/scene.py tests/test_tools/test_scene.py
git commit -m "feat(scene): SceneBoundary model + metadata constants"
```

---

### Task 2: Data Models & Pure Logic — Vision Embeddings

**Files:**
- Create: `src/ave/tools/vision.py`
- Create: `tests/test_tools/test_vision.py`

**Step 1: Write failing tests for vision data models and pure logic**

```python
# tests/test_tools/test_vision.py
"""Unit tests for vision tools — data models and pure math."""

import math

import pytest

from ave.tools.vision import (
    FrameEmbedding,
    SceneTag,
    SimilarityResult,
    VisualAnalysis,
    VisionError,
    cosine_similarity,
    similarity_search,
    tag_frames,
)


class TestCosineSimlarity:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_normalized_vectors(self):
        a = [0.6, 0.8]
        b = [0.8, 0.6]
        expected = 0.6 * 0.8 + 0.8 * 0.6  # 0.96
        assert cosine_similarity(a, b) == pytest.approx(expected, abs=0.001)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)


class TestFrameEmbedding:
    def test_create(self):
        fe = FrameEmbedding(
            frame_idx=10,
            timestamp_ns=416_666_667,
            embedding=[0.1, 0.2, 0.3],
            scene_id="scene_0",
        )
        assert fe.frame_idx == 10
        assert len(fe.embedding) == 3

    def test_embedding_array_property(self):
        fe = FrameEmbedding(
            frame_idx=0,
            timestamp_ns=0,
            embedding=[1.0, 2.0, 3.0],
        )
        arr = fe.embedding_array
        assert arr.shape == (3,)
        assert arr[0] == 1.0

    def test_json_roundtrip(self):
        fe = FrameEmbedding(
            frame_idx=5,
            timestamp_ns=1_000_000_000,
            embedding=[0.1, 0.2],
            scene_id="s1",
        )
        data = fe.model_dump()
        fe2 = FrameEmbedding(**data)
        assert fe2.embedding == fe.embedding
        assert fe2.scene_id == "s1"


class TestSimilaritySearch:
    def test_search_returns_ranked_results(self):
        embeddings = [
            FrameEmbedding(frame_idx=0, timestamp_ns=0, embedding=[1.0, 0.0]),
            FrameEmbedding(frame_idx=1, timestamp_ns=1000, embedding=[0.0, 1.0]),
            FrameEmbedding(frame_idx=2, timestamp_ns=2000, embedding=[0.9, 0.1]),
        ]
        query = [1.0, 0.0]
        results = similarity_search(query, embeddings)
        assert len(results) == 3
        assert results[0].rank == 1
        # frame 0 (exact match) should be first
        assert results[0].frame_embedding.frame_idx == 0
        assert results[0].score == pytest.approx(1.0, abs=0.01)
        # frame 2 (similar) should be second
        assert results[1].frame_embedding.frame_idx == 2

    def test_search_empty_embeddings(self):
        results = similarity_search([1.0, 0.0], [])
        assert results == []

    def test_search_top_k(self):
        embeddings = [
            FrameEmbedding(frame_idx=i, timestamp_ns=i * 1000, embedding=[float(i), 0.0])
            for i in range(10)
        ]
        results = similarity_search([9.0, 0.0], embeddings, top_k=3)
        assert len(results) == 3


class TestTagFrames:
    def test_tag_single_frame(self):
        embeddings = [
            FrameEmbedding(frame_idx=0, timestamp_ns=0, embedding=[1.0, 0.0], scene_id="s0"),
        ]
        labels = {
            "outdoor": [1.0, 0.0],
            "indoor": [0.0, 1.0],
        }
        tags = tag_frames(embeddings, labels)
        assert len(tags) == 1
        assert tags[0].scene_id == "s0"
        assert tags[0].labels["outdoor"] > tags[0].labels["indoor"]

    def test_tag_multiple_frames_same_scene(self):
        embeddings = [
            FrameEmbedding(frame_idx=0, timestamp_ns=0, embedding=[1.0, 0.0], scene_id="s0"),
            FrameEmbedding(frame_idx=1, timestamp_ns=1000, embedding=[0.9, 0.1], scene_id="s0"),
        ]
        labels = {"outdoor": [1.0, 0.0]}
        tags = tag_frames(embeddings, labels)
        # Should aggregate per scene
        assert len(tags) == 1
        assert tags[0].scene_id == "s0"


class TestSceneTag:
    def test_create(self):
        tag = SceneTag(scene_id="s0", labels={"outdoor": 0.9, "indoor": 0.1})
        assert tag.labels["outdoor"] == 0.9

    def test_top_label(self):
        tag = SceneTag(scene_id="s0", labels={"outdoor": 0.9, "indoor": 0.1, "night": 0.5})
        assert tag.top_label == "outdoor"
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tools/test_vision.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement vision.py pure logic**

```python
# src/ave/tools/vision.py
"""Vision tools — image embeddings, similarity search, and zero-shot tagging.

Pure logic layer for data models and vector math.
Vision backend integration (SigLIP 2, etc.) is conditional.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel


class VisionError(Exception):
    """Raised when vision operations fail."""


class FrameEmbedding(BaseModel):
    """Embedding vector for a single video frame."""

    frame_idx: int
    timestamp_ns: int
    embedding: list[float]
    scene_id: str | None = None

    @property
    def embedding_array(self) -> np.ndarray:
        """Return embedding as numpy array for vector operations."""
        return np.array(self.embedding, dtype=np.float32)


class SceneTag(BaseModel):
    """Zero-shot classification labels for a scene."""

    scene_id: str
    labels: dict[str, float]

    @property
    def top_label(self) -> str:
        """Return the highest-confidence label."""
        return max(self.labels, key=self.labels.get)


class SimilarityResult(BaseModel):
    """A single result from similarity search."""

    frame_embedding: FrameEmbedding
    score: float
    rank: int


class VisualAnalysis(BaseModel):
    """Complete visual analysis result for an asset."""

    asset_id: str
    scenes: list[dict] = []  # SceneBoundary dicts (avoid circular import)
    frame_embeddings: list[FrameEmbedding] = []
    tags: list[SceneTag] = []


class VisionBackend(Protocol):
    """Protocol for vision embedding backends. Type-annotation only."""

    def embed_image(self, image: np.ndarray) -> list[float]: ...
    def embed_text(self, text: str) -> list[float]: ...
    def embed_batch(self, images: list[np.ndarray]) -> list[list[float]]: ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def similarity_search(
    query: list[float],
    embeddings: list[FrameEmbedding],
    top_k: int | None = None,
) -> list[SimilarityResult]:
    """Search frame embeddings by cosine similarity to a query vector."""
    if not embeddings:
        return []

    scored = []
    for fe in embeddings:
        score = cosine_similarity(query, fe.embedding)
        scored.append((fe, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        scored = scored[:top_k]

    return [
        SimilarityResult(frame_embedding=fe, score=score, rank=i + 1)
        for i, (fe, score) in enumerate(scored)
    ]


def tag_frames(
    embeddings: list[FrameEmbedding],
    label_embeddings: dict[str, list[float]],
) -> list[SceneTag]:
    """Zero-shot classify frames by cosine similarity to label embeddings.

    Aggregates scores per scene_id (averages across frames in the same scene).
    """
    # Group embeddings by scene_id
    scenes: dict[str, list[FrameEmbedding]] = {}
    for fe in embeddings:
        sid = fe.scene_id or f"frame_{fe.frame_idx}"
        scenes.setdefault(sid, []).append(fe)

    tags = []
    for scene_id, scene_embeddings in scenes.items():
        label_scores: dict[str, float] = {}
        for label, label_emb in label_embeddings.items():
            scores = [cosine_similarity(fe.embedding, label_emb) for fe in scene_embeddings]
            label_scores[label] = sum(scores) / len(scores)
        tags.append(SceneTag(scene_id=scene_id, labels=label_scores))

    return tags


def save_analysis(analysis: VisualAnalysis, path: Path) -> None:
    """Save visual analysis to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis.model_dump(), indent=2))


def load_analysis(path: Path) -> VisualAnalysis:
    """Load visual analysis from JSON file."""
    data = json.loads(path.read_text())
    return VisualAnalysis(**data)
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tools/test_vision.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/ave/tools/vision.py tests/test_tools/test_vision.py
git commit -m "feat(vision): data models, cosine similarity, search, tagging"
```

---

### Task 3: VisualAnalysis Save/Load + Test Markers

**Files:**
- Modify: `tests/test_tools/test_vision.py` (add save/load tests)
- Modify: `tests/conftest.py` (add markers)
- Modify: `pyproject.toml` (add markers + optional deps)

**Step 1: Write failing tests**

Add to `tests/test_tools/test_vision.py`:

```python
class TestVisualAnalysis:
    def test_save_load_roundtrip(self, tmp_path):
        analysis = VisualAnalysis(
            asset_id="clip_001",
            frame_embeddings=[
                FrameEmbedding(frame_idx=0, timestamp_ns=0, embedding=[0.1, 0.2]),
            ],
            tags=[
                SceneTag(scene_id="s0", labels={"outdoor": 0.9}),
            ],
        )
        path = tmp_path / "analysis.json"
        save_analysis(analysis, path)
        loaded = load_analysis(path)
        assert loaded.asset_id == "clip_001"
        assert len(loaded.frame_embeddings) == 1
        assert loaded.frame_embeddings[0].embedding == [0.1, 0.2]
        assert loaded.tags[0].labels["outdoor"] == 0.9

    def test_empty_analysis(self):
        analysis = VisualAnalysis(asset_id="empty")
        assert analysis.frame_embeddings == []
        assert analysis.tags == []
        assert analysis.scenes == []
```

**Step 2: Run — should pass since save_analysis/load_analysis already exist**

Run: `.venv/bin/python -m pytest tests/test_tools/test_vision.py -v`
Expected: PASS

**Step 3: Add test markers and optional deps**

Add to `tests/conftest.py`:

```python
def _scenedetect_available() -> bool:
    try:
        import scenedetect  # noqa: F401
        return True
    except ImportError:
        return False


def _vision_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False
```

And add to the skip markers section:

```python
requires_scenedetect = pytest.mark.skipif(
    not _scenedetect_available(), reason="PySceneDetect not available"
)
requires_vision = pytest.mark.skipif(
    not _vision_available(), reason="ONNX Runtime not available"
)
```

Add to `pyproject.toml` markers:

```toml
"scenedetect: marks tests that require PySceneDetect",
"vision: marks tests that require ONNX Runtime + vision model",
```

Add optional deps to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=6.0",
    "ruff>=0.9.0",
]
vision = [
    "onnxruntime-gpu>=1.20",
    "numpy>=1.26",
    "transformers>=4.49",
]
scene = [
    "scenedetect[opencv]>=0.6",
]
```

**Step 4: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -v -m "not slow" --tb=short`
Expected: All existing tests pass + new tests pass

**Step 5: Commit**

```bash
git add tests/test_tools/test_vision.py tests/conftest.py pyproject.toml
git commit -m "feat: vision save/load + test markers + optional deps"
```

---

### Task 4: PySceneDetect Backend

**Files:**
- Create: `src/ave/tools/scene_pyscenedetect.py`
- Add to: `tests/test_tools/test_scene.py`

**Step 1: Write failing tests for PySceneDetect backend**

Add to `tests/test_tools/test_scene.py`:

```python
from tests.conftest import requires_ffmpeg, requires_scenedetect


@requires_scenedetect
@requires_ffmpeg
class TestPySceneDetectBackend:
    def test_detect_scenes_on_synthetic_video(self, fixtures_dir, tmp_path):
        """Detect cuts in a video with known scene changes."""
        from ave.tools.scene_pyscenedetect import PySceneDetectBackend
        from tests.fixtures.generate import generate_color_bars

        # Create two different clips and concatenate them
        clip_a = tmp_path / "clip_a.mp4"
        clip_b = tmp_path / "clip_b.mp4"
        combined = tmp_path / "combined.mp4"

        generate_color_bars(clip_a, duration=2, width=320, height=240)
        generate_color_bars(clip_b, duration=2, width=320, height=240, fps=24)

        # Concatenate with FFmpeg (creates a hard cut)
        import subprocess
        filelist = tmp_path / "filelist.txt"
        filelist.write_text(f"file '{clip_a}'\nfile '{clip_b}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(filelist), "-c", "copy", str(combined)],
            capture_output=True, check=True,
        )

        backend = PySceneDetectBackend()
        scenes = backend.detect_scenes(combined, threshold=27.0)
        # Should find at least 1 scene (the whole video) or 2 if cut detected
        assert len(scenes) >= 1
        assert all(isinstance(s, SceneBoundary) for s in scenes)
        assert scenes[0].start_ns == 0

    def test_detect_scenes_returns_valid_timestamps(self, fixtures_dir, tmp_path):
        from ave.tools.scene_pyscenedetect import PySceneDetectBackend
        from tests.fixtures.generate import generate_av_clip

        clip = tmp_path / "clip.mp4"
        generate_av_clip(clip, duration=3, width=320, height=240)

        backend = PySceneDetectBackend()
        scenes = backend.detect_scenes(clip, threshold=27.0)
        assert len(scenes) >= 1
        for scene in scenes:
            assert scene.start_ns >= 0
            assert scene.end_ns > scene.start_ns
            assert scene.fps > 0

    def test_detect_scenes_adaptive_detector(self, fixtures_dir, tmp_path):
        from ave.tools.scene_pyscenedetect import PySceneDetectBackend
        from tests.fixtures.generate import generate_av_clip

        clip = tmp_path / "clip.mp4"
        generate_av_clip(clip, duration=3, width=320, height=240)

        backend = PySceneDetectBackend()
        scenes = backend.detect_scenes(clip, threshold=3.0, detector="adaptive")
        assert len(scenes) >= 1

    def test_invalid_detector_raises(self, tmp_path):
        from ave.tools.scene_pyscenedetect import PySceneDetectBackend
        from tests.fixtures.generate import generate_av_clip

        clip = tmp_path / "clip.mp4"
        generate_av_clip(clip, duration=1, width=320, height=240)

        backend = PySceneDetectBackend()
        with pytest.raises(SceneError, match="Unknown detector"):
            backend.detect_scenes(clip, threshold=27.0, detector="invalid")
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_tools/test_scene.py::TestPySceneDetectBackend -v`
Expected: FAIL or SKIP (if scenedetect not installed)

**Step 3: Implement PySceneDetect backend**

```python
# src/ave/tools/scene_pyscenedetect.py
"""PySceneDetect backend for scene boundary detection."""

from __future__ import annotations

from pathlib import Path

from ave.tools.scene import SceneBoundary, SceneError


class PySceneDetectBackend:
    """Scene detection using PySceneDetect library."""

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
        """
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import (
                ContentDetector,
                AdaptiveDetector,
                ThresholdDetector,
                HashDetector,
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
                f"Unknown detector: {detector}. "
                f"Valid options: {', '.join(detectors.keys())}"
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
            boundaries.append(
                SceneBoundary(start_ns=start_ns, end_ns=end_ns, fps=fps)
            )

        # If no scenes detected, return the whole video as one scene
        if not boundaries:
            duration_ns = int(video.duration.get_seconds() * 1_000_000_000)
            boundaries.append(
                SceneBoundary(start_ns=0, end_ns=duration_ns, fps=fps)
            )

        return boundaries
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_tools/test_scene.py -v`
Expected: Pure logic tests PASS, PySceneDetect tests PASS if installed or SKIP if not

**Step 5: Commit**

```bash
git add src/ave/tools/scene_pyscenedetect.py tests/test_tools/test_scene.py
git commit -m "feat(scene): PySceneDetect backend with content/adaptive/threshold/hash"
```

---

### Task 5: Keyframe Extraction Utility

**Files:**
- Modify: `src/ave/tools/scene.py` (add extract_keyframes)
- Add to: `tests/test_tools/test_scene.py`

**Step 1: Write failing tests**

```python
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
class TestExtractKeyframes:
    def test_extract_middle_keyframes(self, tmp_path):
        from ave.tools.scene import extract_keyframes, SceneBoundary
        from tests.fixtures.generate import generate_av_clip

        clip = tmp_path / "clip.mp4"
        generate_av_clip(clip, duration=4, width=320, height=240)

        boundaries = [
            SceneBoundary(start_ns=0, end_ns=2_000_000_000, fps=24.0),
            SceneBoundary(start_ns=2_000_000_000, end_ns=4_000_000_000, fps=24.0),
        ]

        output_dir = tmp_path / "keyframes"
        paths = extract_keyframes(clip, boundaries, output_dir, strategy="middle")

        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.suffix == ".jpg"

    def test_extract_first_keyframes(self, tmp_path):
        from ave.tools.scene import extract_keyframes, SceneBoundary
        from tests.fixtures.generate import generate_av_clip

        clip = tmp_path / "clip.mp4"
        generate_av_clip(clip, duration=2, width=320, height=240)

        boundaries = [
            SceneBoundary(start_ns=0, end_ns=2_000_000_000, fps=24.0),
        ]

        output_dir = tmp_path / "keyframes"
        paths = extract_keyframes(clip, boundaries, output_dir, strategy="first")

        assert len(paths) == 1
        assert paths[0].exists()
```

**Step 2: Run to verify failure**

**Step 3: Implement extract_keyframes**

Add to `src/ave/tools/scene.py`:

```python
import subprocess

def extract_keyframes(
    video_path: Path,
    boundaries: list[SceneBoundary],
    output_dir: Path,
    strategy: str = "middle",
) -> list[Path]:
    """Extract one keyframe per scene boundary using FFmpeg.

    This is an I/O utility, not pure logic.

    Args:
        video_path: Source video file.
        boundaries: Scene boundaries to extract from.
        output_dir: Directory to write keyframe images.
        strategy: "middle" (middle of scene) or "first" (first frame).

    Returns:
        List of paths to extracted keyframe images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, boundary in enumerate(boundaries):
        if strategy == "middle":
            seek_ns = (boundary.start_ns + boundary.end_ns) // 2
        elif strategy == "first":
            seek_ns = boundary.start_ns
        else:
            raise SceneError(f"Unknown strategy: {strategy}. Use 'middle' or 'first'.")

        seek_seconds = seek_ns / 1_000_000_000
        output_path = output_dir / f"keyframe_{i:04d}.jpg"

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{seek_seconds:.6f}",
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(output_path),
                ],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise SceneError(
                f"Keyframe extraction failed at {seek_seconds}s: {e.stderr.decode()}"
            ) from e

        paths.append(output_path)

    return paths
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_tools/test_scene.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/ave/tools/scene.py tests/test_tools/test_scene.py
git commit -m "feat(scene): keyframe extraction via FFmpeg"
```

---

### Task 6: SigLIP 2 Backend

**Files:**
- Create: `src/ave/tools/vision_siglip2.py`
- Add to: `tests/test_tools/test_vision.py`

**Step 1: Write failing tests**

```python
from tests.conftest import requires_vision


@requires_vision
class TestSigLIP2Backend:
    def test_embed_image_returns_list_float(self):
        import numpy as np
        from ave.tools.vision_siglip2 import SigLIP2Backend

        backend = SigLIP2Backend(model_name="google/siglip2-base-patch16-224")
        # Create a synthetic 224x224 RGB image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = backend.embed_image(image)
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
        assert len(result) > 0

    def test_embed_text_returns_list_float(self):
        from ave.tools.vision_siglip2 import SigLIP2Backend

        backend = SigLIP2Backend(model_name="google/siglip2-base-patch16-224")
        result = backend.embed_text("a person walking outdoors")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_embed_batch(self):
        import numpy as np
        from ave.tools.vision_siglip2 import SigLIP2Backend

        backend = SigLIP2Backend(model_name="google/siglip2-base-patch16-224")
        images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        results = backend.embed_batch(images)
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_image_text_similarity_makes_sense(self):
        import numpy as np
        from ave.tools.vision_siglip2 import SigLIP2Backend
        from ave.tools.vision import cosine_similarity

        backend = SigLIP2Backend(model_name="google/siglip2-base-patch16-224")

        # Red image
        red_image = np.zeros((224, 224, 3), dtype=np.uint8)
        red_image[:, :, 0] = 255

        red_embedding = backend.embed_image(red_image)
        red_text = backend.embed_text("a red image")
        blue_text = backend.embed_text("a blue image")

        sim_red = cosine_similarity(red_embedding, red_text)
        sim_blue = cosine_similarity(red_embedding, blue_text)

        # Red image should be more similar to "red" text than "blue" text
        assert sim_red > sim_blue
```

**Step 2: Run — should SKIP without onnxruntime**

**Step 3: Implement SigLIP 2 backend**

```python
# src/ave/tools/vision_siglip2.py
"""SigLIP 2 vision backend — image/text embeddings via ONNX + CUDA."""

from __future__ import annotations

import numpy as np

from ave.tools.vision import VisionError


class SigLIP2Backend:
    """Vision embedding backend using Google SigLIP 2.

    Uses HuggingFace transformers for model loading and ONNX Runtime
    for fast inference with CUDA acceleration.
    """

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        try:
            from transformers import AutoProcessor, AutoModel
        except ImportError:
            raise VisionError(
                "transformers not installed. "
                "Install with: pip install transformers"
            )

        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

    def embed_image(self, image: np.ndarray) -> list[float]:
        """Embed a single image (H, W, 3 uint8) into a vector."""
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image)
        inputs = self._processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)

        embedding = outputs[0].cpu().numpy()
        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    def embed_text(self, text: str) -> list[float]:
        """Embed a text string into a vector."""
        import torch

        inputs = self._processor(text=[text], return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)

        embedding = outputs[0].cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    def embed_batch(self, images: list[np.ndarray]) -> list[list[float]]:
        """Embed multiple images in one forward pass."""
        import torch
        from PIL import Image

        pil_images = [Image.fromarray(img) for img in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)

        embeddings = outputs.cpu().numpy()
        # L2 normalize each
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
        return embeddings.tolist()
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_tools/test_vision.py -v`
Expected: Pure logic tests PASS, SigLIP2 tests PASS if transformers installed or SKIP

**Step 5: Commit**

```bash
git add src/ave/tools/vision_siglip2.py tests/test_tools/test_vision.py
git commit -m "feat(vision): SigLIP 2 backend with image/text embedding"
```

---

### Task 7: Docker Integration

**Files:**
- Modify: `docker/Dockerfile`
- Modify: `docker/docker-compose.yml`

**Step 1: Add scene + vision deps to Dockerfile**

After the existing `pip install pywhispercpp` line, add:

```dockerfile
# Install scene detection and vision deps
RUN pip install --break-system-packages \
    "scenedetect[opencv]>=0.6" \
    "onnxruntime-gpu>=1.20" \
    "transformers>=4.49" \
    "numpy>=1.26" \
    "Pillow>=10.0" \
    "torch>=2.0"
```

**Step 2: Add vision model cache volume to docker-compose.yml**

The existing `whisper-models` volume pattern works. Add `vision-models` volume:

```yaml
volumes:
  whisper-models:
  vision-models:
```

Add to `ave` service volumes:

```yaml
- vision-models:/root/.cache/huggingface
```

**Step 3: Verify Dockerfile builds (if Docker available)**

Run: `docker compose -f docker/docker-compose.yml build`

**Step 4: Commit**

```bash
git add docker/Dockerfile docker/docker-compose.yml
git commit -m "feat(docker): add PySceneDetect + SigLIP 2 dependencies"
```

---

### Task 8: E2E Test — Scene Detection Pipeline

**Files:**
- Create: `tests/test_e2e/test_scene_e2e.py`

**Step 1: Write E2E test**

```python
# tests/test_e2e/test_scene_e2e.py
"""E2E tests for scene detection pipeline."""

import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg, requires_scenedetect


@requires_scenedetect
@requires_ffmpeg
@pytest.mark.slow
class TestSceneDetectionE2E:
    def test_detect_and_extract_keyframes(self, tmp_path):
        """Full pipeline: generate video with cuts → detect scenes → extract keyframes."""
        from tests.fixtures.generate import generate_color_bars
        from ave.tools.scene import SceneBoundary, extract_keyframes
        from ave.tools.scene_pyscenedetect import PySceneDetectBackend

        # Create video with a hard cut (two different clips concatenated)
        clip_a = tmp_path / "a.mp4"
        clip_b = tmp_path / "b.mp4"
        combined = tmp_path / "combined.mp4"

        generate_color_bars(clip_a, duration=2, width=320, height=240)
        # Different content for detectable cut
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             "color=c=red:size=320x240:rate=24:duration=2",
             "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", str(clip_b)],
            capture_output=True, check=True,
        )

        filelist = tmp_path / "filelist.txt"
        filelist.write_text(f"file '{clip_a}'\nfile '{clip_b}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(filelist), "-c", "copy", str(combined)],
            capture_output=True, check=True,
        )

        # Detect scenes
        backend = PySceneDetectBackend()
        scenes = backend.detect_scenes(combined, threshold=27.0)
        assert len(scenes) >= 1

        # Extract keyframes
        kf_dir = tmp_path / "keyframes"
        keyframes = extract_keyframes(combined, scenes, kf_dir, strategy="middle")
        assert len(keyframes) == len(scenes)
        for kf in keyframes:
            assert kf.exists()
            assert kf.stat().st_size > 0
```

**Step 2: Run (should pass if scenedetect + ffmpeg available, skip otherwise)**

Run: `.venv/bin/python -m pytest tests/test_e2e/test_scene_e2e.py -v`

**Step 3: Commit**

```bash
git add tests/test_e2e/test_scene_e2e.py
git commit -m "test(e2e): scene detection pipeline — detect + extract keyframes"
```

---

### Task 9: Final Lint, Format, Full Suite, Push + PR

**Step 1: Lint and format**

```bash
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format src/ tests/
```

**Step 2: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v -m "not slow" --tb=short
```

Expected: All existing tests pass + new pure logic tests pass. Backend tests skip if deps not installed.

**Step 3: Reinstall package**

```bash
.venv/bin/pip install -e . --no-deps
```

**Step 4: Push and create PR**

```bash
git push -u origin feature/vision-scene-detection
gh pr create --title "Vision embeddings (SigLIP 2) + scene detection (PySceneDetect)" --body "..."
```

**Step 5: Wait for CI, merge**
