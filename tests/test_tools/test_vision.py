"""Unit tests for vision tools — data models, pure math, and backends."""

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
from tests.conftest import requires_vision


class TestVisionError:
    def test_is_exception(self):
        err = VisionError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"


class TestSimilarityResultModel:
    def test_create(self):
        fe = FrameEmbedding(frame_idx=0, timestamp_ns=0, embedding=[1.0, 0.0])
        sr = SimilarityResult(frame_embedding=fe, score=0.95, rank=1)
        assert sr.score == 0.95
        assert sr.rank == 1


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

    @requires_vision
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


class TestVisualAnalysis:
    def test_save_load_roundtrip(self, tmp_path):
        from ave.tools.vision import save_analysis, load_analysis

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
        images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(3)]
        results = backend.embed_batch(images)
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_image_text_similarity_makes_sense(self):
        import numpy as np

        from ave.tools.vision import cosine_similarity
        from ave.tools.vision_siglip2 import SigLIP2Backend

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
