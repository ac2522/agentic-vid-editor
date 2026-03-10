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
        return max(self.labels, key=self.labels.get)  # type: ignore[arg-type]


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
    """Compute cosine similarity between two vectors.

    Uses pure math (no numpy) for portability.
    """
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
    """Search frame embeddings by cosine similarity to a query vector.

    Args:
        query: Query embedding vector.
        embeddings: Frame embeddings to search.
        top_k: If set, return only the top-k most similar results.

    Returns:
        List of SimilarityResult sorted by descending similarity score.
    """
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

    Args:
        embeddings: Frame embeddings with optional scene_id.
        label_embeddings: Mapping of label name to embedding vector.

    Returns:
        List of SceneTag, one per unique scene_id.
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
