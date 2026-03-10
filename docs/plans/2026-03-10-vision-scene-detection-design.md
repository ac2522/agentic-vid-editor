# Vision & Scene Detection — Design

## Overview
Modular vision embedding and scene detection tools for the agentic video editor. Protocol-based interfaces with swappable backends. SigLIP 2 for vision embeddings, PySceneDetect for scene boundary detection.

## Architecture Principle
Abstract interfaces with swappable backends — the agent calls `detect_scenes()` or `embed_frames()`, never imports a specific backend directly. Today SigLIP 2 + PySceneDetect, tomorrow Gemini + something else. Same pattern as `transcribe()` with whisper.cpp.

## Module Structure

```
src/ave/tools/
  vision.py              # VisionBackend Protocol + pure logic (similarity, tagging)
  vision_siglip2.py      # SigLIP 2 backend (ONNX + CUDA)
  scene.py               # SceneBackend Protocol + pure logic (boundary models)
  scene_pyscenedetect.py # PySceneDetect backend
```

## Protocols (type-annotation-only, not runtime_checkable)

```python
class VisionBackend(Protocol):
    def embed_image(self, image: np.ndarray) -> list[float]: ...
    def embed_text(self, text: str) -> list[float]: ...
    def embed_batch(self, images: list[np.ndarray]) -> list[list[float]]: ...

class SceneBackend(Protocol):
    def detect_scenes(self, video_path: Path, threshold: float,
                      detector: str = "content") -> list[SceneBoundary]: ...
```

## Data Models (Pydantic)

### SceneBoundary
- `start_ns: int` — authoritative timestamp
- `end_ns: int` — authoritative timestamp
- `fps: float` — for frame number derivation
- `start_frame` / `end_frame` as `@property` (derived from ns + fps)

### FrameEmbedding
- `frame_idx: int`
- `timestamp_ns: int`
- `embedding: list[float]` — JSON-serializable (not np.ndarray)
- `scene_id: str | None`
- `embedding_array` as `@property` → returns `np.ndarray`

### SceneTag
- `scene_id: str`
- `labels: dict[str, float]` — label → confidence score

### VisualAnalysis (top-level, like Transcript)
- `asset_id: str`
- `scenes: list[SceneBoundary]`
- `frame_embeddings: list[FrameEmbedding]`
- `tags: list[SceneTag]`
- `save_analysis()` / `load_analysis()` functions (matches save_transcript/load_transcript)

## Pure Logic Layer (no backend dependency)

### vision.py
- `cosine_similarity(a: list[float], b: list[float]) -> float`
- `similarity_search(query_embedding, frame_embeddings) -> list[SimilarityResult]`
  - SimilarityResult: `frame_embedding: FrameEmbedding, score: float, rank: int`
- `tag_frames(frame_embeddings, label_embeddings: dict[str, list[float]]) -> list[SceneTag]`
  - Zero-shot classification via cosine similarity
- `save_analysis(analysis: VisualAnalysis, path: Path)`
- `load_analysis(path: Path) -> VisualAnalysis`

### scene.py
- `detect_scenes(video_path, backend, threshold, detector)` → `list[SceneBoundary]`
- Metadata key constants: `AGENT_META_SCENE_ID = "agent:scene-id"` etc.

## I/O Utilities (not pure logic)
- `extract_keyframes(video_path, boundaries, strategy)` — FFmpeg-based frame extraction
  - Strategies: `"middle"` (middle of each scene) or `"first"` (first frame after cut)
  - Parallel to `extract_audio_for_transcription` in transcribe.py
- `embed_frames(video_path, timestamps_ns, backend)` → `list[FrameEmbedding]`

## Scene Detection Backend: PySceneDetect

- Wraps `ContentDetector` (jump cuts), `AdaptiveDetector` (camera movement), `ThresholdDetector` (fades), `HashDetector` (perceptual hash)
- Detector selectable via `detector: str` parameter
- Returns timestamps in nanoseconds for GES compatibility
- CPU-only OpenCV for v1 (GPU OpenCV as future optimization)
- Dependency: `scenedetect[opencv]>=0.6`

## Vision Backend: SigLIP 2

- ONNX Runtime with CUDA execution provider for fast inference
- Default model: `google/siglip2-base-patch16-224` (86M params, fastest)
- Configurable model size: base (86M), large (303M), So400m (400M), giant (1B)
- ONNX quantization: 3.7x size reduction, 2x speed boost
- Model cache: `~/.cache/ave/models/` (same pattern as whisper)
- Dependencies: `onnxruntime-gpu>=1.20`, `numpy>=1.26`, `transformers>=4.49`

## Integration with Existing Systems

### Timeline Metadata
- Scene boundaries stored as `agent:scene-id`, `agent:scene-start-ns`, `agent:scene-end-ns`
- Constants defined at module level in scene.py

### Asset Registry
- `AssetEntry.visual_analysis_path` already scaffolded — points to VisualAnalysis JSON

### Docker
- `pip install scenedetect[opencv] onnxruntime-gpu transformers` in Dockerfile
- Model cache volume (same whisper-models volume pattern)
- CPU OpenCV for v1 scene detection

### pyproject.toml Optional Dependencies
```toml
[project.optional-dependencies]
vision = ["onnxruntime-gpu>=1.20", "numpy>=1.26", "transformers>=4.49"]
scene = ["scenedetect[opencv]>=0.6"]
```

## Testing Strategy

### Test Markers
- `requires_scenedetect` — PySceneDetect available
- `requires_vision` — ONNX Runtime + SigLIP 2 model available

### Unit Tests (pure logic, no deps)
- Cosine similarity computation
- Search ranking with synthetic embeddings
- Tag assignment (zero-shot classification math)
- SceneBoundary frame number derivation from ns + fps
- VisualAnalysis save/load roundtrip

### Integration Tests
- PySceneDetect on synthetic video with known cuts (FFmpeg-generated)
- SigLIP 2 model produces correct-shape embeddings
- Keyframe extraction produces valid images

### E2E Tests (Docker)
- Ingest → detect scenes → embed keyframes → tag → store analysis → verify
