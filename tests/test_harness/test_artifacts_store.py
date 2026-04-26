"""Tests for the harness artifacts store (rendered MP4 + judge traces)."""
import json
from pathlib import Path


def test_store_init_creates_directory(tmp_path):
    from ave.harness.artifacts.store import ArtifactStore
    ArtifactStore(root=tmp_path / "artifacts")
    assert (tmp_path / "artifacts").exists()


def test_store_write_render_copies_mp4(tmp_path):
    from ave.harness.artifacts.store import ArtifactStore
    src = tmp_path / "render.mp4"
    src.write_bytes(b"FAKE_MP4")
    store = ArtifactStore(root=tmp_path / "store")
    info = store.write_render(scenario_id="reel.test", run_id="r1", mp4_path=src)
    assert Path(info.mp4_path).exists()
    assert Path(info.mp4_path).read_bytes() == b"FAKE_MP4"


def test_store_write_trace_writes_json(tmp_path):
    from ave.harness.artifacts.store import ArtifactStore
    store = ArtifactStore(root=tmp_path / "store")
    info = store.write_trace(
        scenario_id="reel.test", run_id="r1",
        trace={"dimensions": [{"name": "framing", "score": 0.85}]},
    )
    data = json.loads(Path(info.trace_path).read_text())
    assert data["dimensions"][0]["score"] == 0.85


def test_store_prune_removes_old_artifacts(tmp_path):
    """Artifacts older than retention_days should be removed."""
    import os
    import time

    from ave.harness.artifacts.store import ArtifactStore
    store = ArtifactStore(root=tmp_path / "store")
    src = tmp_path / "x.mp4"
    src.write_bytes(b"x")
    info = store.write_render(scenario_id="s", run_id="r1", mp4_path=src)
    old_time = time.time() - (40 * 86400)
    os.utime(info.mp4_path, (old_time, old_time))
    pruned = store.prune(retention_days=30)
    assert pruned >= 1
    assert not Path(info.mp4_path).exists()


def test_store_prune_keeps_recent_artifacts(tmp_path):
    from ave.harness.artifacts.store import ArtifactStore
    store = ArtifactStore(root=tmp_path / "store")
    src = tmp_path / "x.mp4"
    src.write_bytes(b"x")
    info = store.write_render(scenario_id="s", run_id="r2", mp4_path=src)
    pruned = store.prune(retention_days=30)
    assert pruned == 0
    assert Path(info.mp4_path).exists()
