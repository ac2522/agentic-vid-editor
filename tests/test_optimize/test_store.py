"""Tests for artifact version store."""

from __future__ import annotations

from pathlib import Path

import pytest

from ave.optimize.artifacts import ArtifactKind, ContextArtifact
from ave.optimize.store import ArtifactStore


def _make_artifact(
    artifact_id: str = "role.editor.system_prompt",
    content: str = "You are a video editor.",
) -> ContextArtifact:
    return ContextArtifact(
        id=artifact_id,
        kind=ArtifactKind.SYSTEM_PROMPT,
        content=content,
        source_location="src/ave/agent/roles.py",
        metadata={},
    )


class TestArtifactStore:
    def test_save_returns_version_number(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        artifact = _make_artifact()
        version = store.save(artifact, score=0.75, campaign_id="2026-03-16_abc12345")
        assert version == 1

    def test_save_increments_version(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        artifact = _make_artifact()
        v1 = store.save(artifact, score=0.75, campaign_id="c1")
        v2 = store.save(
            _make_artifact(content="Improved prompt"), score=0.85, campaign_id="c2"
        )
        assert v1 == 1
        assert v2 == 2

    def test_load_best_returns_highest_scored(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        store.save(_make_artifact(content="v1 content"), score=0.75, campaign_id="c1")
        store.save(_make_artifact(content="v2 content"), score=0.90, campaign_id="c2")
        store.save(_make_artifact(content="v3 content"), score=0.80, campaign_id="c3")
        best = store.load_best("role.editor.system_prompt")
        assert best is not None
        assert best.content == "v2 content"

    def test_load_best_nonexistent_returns_none(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        assert store.load_best("nonexistent.artifact") is None

    def test_current_best_score(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        assert store.current_best_score("role.editor.system_prompt") is None
        store.save(_make_artifact(), score=0.75, campaign_id="c1")
        assert store.current_best_score("role.editor.system_prompt") == 0.75
        store.save(_make_artifact(content="better"), score=0.90, campaign_id="c2")
        assert store.current_best_score("role.editor.system_prompt") == 0.90

    def test_history_returns_all_versions(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        store.save(_make_artifact(content="v1"), score=0.7, campaign_id="c1")
        store.save(_make_artifact(content="v2"), score=0.8, campaign_id="c2")
        history = store.history("role.editor.system_prompt")
        assert len(history) == 2
        assert history[0]["version"] == 1
        assert history[0]["score"] == 0.7
        assert history[1]["version"] == 2
        assert history[1]["score"] == 0.8

    def test_history_nonexistent_returns_empty(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        assert store.history("nonexistent") == []

    def test_diff_between_versions(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        store.save(_make_artifact(content="original text"), score=0.7, campaign_id="c1")
        store.save(_make_artifact(content="improved text"), score=0.8, campaign_id="c2")
        diff_output = store.diff("role.editor.system_prompt", v1=1, v2=2)
        assert "original" in diff_output or "-original" in diff_output
        assert "improved" in diff_output or "+improved" in diff_output

    def test_artifact_id_with_dots_creates_valid_directories(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        artifact = _make_artifact(artifact_id="tool.trim_clip.description")
        store.save(artifact, score=0.8, campaign_id="c1")
        # Directory should exist
        artifact_dir = tmp_path / "optimized" / "artifacts" / "tool.trim_clip.description"
        assert artifact_dir.is_dir()

    def test_store_persists_across_instances(self, tmp_path: Path):
        store1 = ArtifactStore(tmp_path)
        store1.save(_make_artifact(content="persisted"), score=0.9, campaign_id="c1")

        store2 = ArtifactStore(tmp_path)
        best = store2.load_best("role.editor.system_prompt")
        assert best is not None
        assert best.content == "persisted"

    def test_campaign_log_written(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        store.save(_make_artifact(), score=0.8, campaign_id="2026-03-16_abc12345")
        campaigns_file = tmp_path / "optimized" / "campaigns.jsonl"
        assert campaigns_file.exists()

    def test_path_traversal_rejected(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        artifact = _make_artifact(artifact_id="../../etc/passwd")
        with pytest.raises(ValueError, match="Invalid artifact_id"):
            store.save(artifact, score=0.5, campaign_id="c1")

    def test_path_traversal_in_load_rejected(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        with pytest.raises(ValueError, match="Invalid artifact_id"):
            store.load_best("../../../etc/passwd")

    def test_diff_missing_version_raises(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        store.save(_make_artifact(content="v1"), score=0.7, campaign_id="c1")
        with pytest.raises(FileNotFoundError):
            store.diff("role.editor.system_prompt", v1=1, v2=99)
