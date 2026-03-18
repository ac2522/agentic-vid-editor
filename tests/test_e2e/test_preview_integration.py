"""Cross-module integration tests for the Preview System (Phase 3).

Verifies that preview pipeline modules (segment, cache, frame, server) work
coherently together end-to-end.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from ave.preview.cache import SegmentCache, SegmentState
from ave.preview.frame import compute_frame_timecode, extract_frame
from ave.render.segment import (
    SegmentBoundary,
    compute_segment_boundaries,
    segment_filename,
)

from tests.conftest import FIXTURES_DIR, requires_ffmpeg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NS_PER_SEC = 1_000_000_000
TEST_VIDEO = FIXTURES_DIR / "av_clip_1080p24.mp4"


def _boundaries_as_tuples(
    boundaries: list[SegmentBoundary],
) -> list[tuple[int, int, int]]:
    """Convert SegmentBoundary list to (index, start_ns, end_ns) tuples."""
    return [(b.index, b.start_ns, b.end_ns) for b in boundaries]


# ---------------------------------------------------------------------------
# Preview Pipeline Integration (pure logic + FFmpeg, no GES/server)
# ---------------------------------------------------------------------------


class TestPreviewPipeline:
    """Integration tests that exercise segment + cache + frame modules together."""

    def test_segment_boundaries_to_cache_registration(self, tmp_path: Path) -> None:
        """Compute boundaries from a 20s timeline, register in cache,
        verify all segments start DIRTY with correct count."""
        duration_ns = 20 * NS_PER_SEC
        boundaries = compute_segment_boundaries(duration_ns, segment_duration_ns=5 * NS_PER_SEC)

        assert len(boundaries) == 4

        cache = SegmentCache(tmp_path / "cache", timeline_id="tl_20s")
        cache.register_segments(_boundaries_as_tuples(boundaries))

        counts = cache.segment_count()
        assert counts["total"] == 4
        assert counts["dirty"] == 4
        assert counts["clean"] == 0
        assert counts["rendering"] == 0

        for b in boundaries:
            assert cache.get_state(b.index) == SegmentState.DIRTY

    def test_segment_boundaries_filenames(self) -> None:
        """Compute boundaries then generate filenames; verify naming pattern."""
        duration_ns = 20 * NS_PER_SEC
        boundaries = compute_segment_boundaries(duration_ns, segment_duration_ns=5 * NS_PER_SEC)

        timeline_id = "proj42"
        filenames = [segment_filename(timeline_id, b.start_ns, b.end_ns) for b in boundaries]

        assert len(filenames) == 4
        for fn in filenames:
            assert fn.startswith(f"{timeline_id}_")
            assert fn.endswith(".mp4")

        # Verify specific expected filenames
        assert filenames[0] == f"{timeline_id}_0_{5 * NS_PER_SEC}.mp4"
        assert filenames[1] == f"{timeline_id}_{5 * NS_PER_SEC}_{10 * NS_PER_SEC}.mp4"

        # All filenames must be unique
        assert len(set(filenames)) == len(filenames)

    def test_cache_invalidation_after_edit(self, tmp_path: Path) -> None:
        """Register 4 segments, mark all CLEAN, simulate edit at 7s,
        verify only overlapping segment is DIRTY."""
        duration_ns = 20 * NS_PER_SEC
        boundaries = compute_segment_boundaries(duration_ns, segment_duration_ns=5 * NS_PER_SEC)

        cache = SegmentCache(tmp_path / "cache", timeline_id="edit_test")
        cache.register_segments(_boundaries_as_tuples(boundaries))

        # Transition all segments through DIRTY -> RENDERING -> CLEAN
        for b in boundaries:
            fp = tmp_path / "cache" / f"seg_{b.index}.mp4"
            fp.write_bytes(b"fake segment data")
            cache.mark_rendering(b.index)
            cache.mark_clean(b.index, fp)

        # Verify all clean
        assert cache.segment_count()["clean"] == 4

        # Simulate edit at 7s: invalidate range 5s-10s
        cache.invalidate_range(5 * NS_PER_SEC, 10 * NS_PER_SEC)

        # Segment 1 (5s-10s) should be DIRTY; others remain CLEAN
        assert cache.get_state(0) == SegmentState.CLEAN  # 0s-5s
        assert cache.get_state(1) == SegmentState.DIRTY  # 5s-10s
        assert cache.get_state(2) == SegmentState.CLEAN  # 10s-15s
        assert cache.get_state(3) == SegmentState.CLEAN  # 15s-20s

    def test_viewport_priority_ordering(self, tmp_path: Path) -> None:
        """Register 4 segments, set viewport to 8s-15s,
        dirty segments in viewport returned first."""
        duration_ns = 20 * NS_PER_SEC
        boundaries = compute_segment_boundaries(duration_ns, segment_duration_ns=5 * NS_PER_SEC)

        cache = SegmentCache(tmp_path / "cache", timeline_id="viewport_test")
        cache.register_segments(_boundaries_as_tuples(boundaries))

        # All segments are DIRTY initially
        dirty = cache.get_dirty_segments(
            viewport_start_ns=8 * NS_PER_SEC,
            viewport_end_ns=15 * NS_PER_SEC,
        )

        assert len(dirty) == 4

        # Segments overlapping 8s-15s are indices 1 (5-10s) and 2 (10-15s)
        viewport_indices = [s.index for s in dirty[:2]]
        assert 1 in viewport_indices
        assert 2 in viewport_indices

        # Non-viewport segments come after
        outside_indices = [s.index for s in dirty[2:]]
        assert 0 in outside_indices
        assert 3 in outside_indices

    @requires_ffmpeg
    def test_frame_extraction_at_segment_boundaries(self, tmp_path: Path) -> None:
        """For each segment boundary, extract a frame at the boundary timestamp,
        verify valid JPEG."""
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test fixture not found: {TEST_VIDEO}")

        # 5s video with 5s segments = 1 segment; use 1s segments for more boundaries
        duration_ns = 5 * NS_PER_SEC
        boundaries = compute_segment_boundaries(duration_ns, segment_duration_ns=1 * NS_PER_SEC)

        assert len(boundaries) == 5

        jpeg_magic = b"\xff\xd8\xff"

        for b in boundaries:
            # Extract frame at start of each segment
            frame_data = extract_frame(TEST_VIDEO, b.start_ns, width=320)
            assert len(frame_data) > 0
            assert frame_data[:3] == jpeg_magic, (
                f"Frame at {compute_frame_timecode(b.start_ns)} is not valid JPEG"
            )

    def test_cache_clean_with_real_file(self, tmp_path: Path) -> None:
        """Create a real temp file, register segment, mark clean with file,
        verify path, invalidate, verify file deleted."""
        cache = SegmentCache(tmp_path / "cache", timeline_id="file_test")
        cache.register_segments([(0, 0, 5 * NS_PER_SEC)])

        seg_file = tmp_path / "cache" / "seg_0.mp4"
        seg_file.write_bytes(b"segment video content")

        cache.mark_rendering(0)
        cache.mark_clean(0, seg_file)
        assert cache.get_state(0) == SegmentState.CLEAN
        assert cache.get_segment_path(0) == seg_file
        assert seg_file.exists()

        # Invalidate should delete file
        cache.invalidate_segment(0)
        assert cache.get_state(0) == SegmentState.DIRTY
        assert not seg_file.exists()

    def test_cache_state_persistence_roundtrip(self, tmp_path: Path) -> None:
        """Register segments, mark some clean, save state, load state,
        verify all states preserved."""
        cache_dir = tmp_path / "cache"
        cache = SegmentCache(cache_dir, timeline_id="persist_test")

        boundaries = _boundaries_as_tuples(
            compute_segment_boundaries(15 * NS_PER_SEC, segment_duration_ns=5 * NS_PER_SEC)
        )
        cache.register_segments(boundaries)

        # Mark segments 0 and 2 as CLEAN (DIRTY -> RENDERING -> CLEAN)
        for idx in (0, 2):
            fp = cache_dir / f"seg_{idx}.mp4"
            fp.write_bytes(b"data")
            cache.mark_rendering(idx)
            cache.mark_clean(idx, fp)

        # Save
        state_file = tmp_path / "cache_state.json"
        cache.save_state(state_file)

        # Load into fresh cache
        loaded = SegmentCache.load_state(state_file, cache_dir)

        assert loaded.get_state(0) == SegmentState.CLEAN
        assert loaded.get_state(1) == SegmentState.DIRTY
        assert loaded.get_state(2) == SegmentState.CLEAN
        assert loaded.segment_count() == cache.segment_count()

    def test_full_preview_pipeline_simulation(self, tmp_path: Path) -> None:
        """Simulate full flow: compute boundaries -> register in cache ->
        iterate dirty segments -> 'render' (create dummy file) -> mark clean ->
        verify all clean -> simulate edit -> verify affected segments dirty."""
        duration_ns = 20 * NS_PER_SEC
        boundaries = compute_segment_boundaries(duration_ns, segment_duration_ns=5 * NS_PER_SEC)

        cache_dir = tmp_path / "segments"
        cache = SegmentCache(cache_dir, timeline_id="full_flow")
        cache.register_segments(_boundaries_as_tuples(boundaries))

        # Step 1: All dirty
        assert cache.segment_count()["dirty"] == 4

        # Step 2: Iterate dirty, "render" each
        dirty_segs = cache.get_dirty_segments()
        for seg in dirty_segs:
            cache.mark_rendering(seg.index)
            assert cache.get_state(seg.index) == SegmentState.RENDERING

            # Create dummy rendered file
            fname = segment_filename("full_flow", seg.start_ns, seg.end_ns)
            fp = cache_dir / fname
            fp.write_bytes(b"rendered fmp4 data")
            cache.mark_clean(seg.index, fp)
            assert cache.get_state(seg.index) == SegmentState.CLEAN

        # Step 3: All clean
        counts = cache.segment_count()
        assert counts["clean"] == 4
        assert counts["dirty"] == 0

        # Step 4: Simulate edit at 12s (invalidate 10s-15s)
        cache.invalidate_range(10 * NS_PER_SEC, 15 * NS_PER_SEC)

        # Segment 2 (10s-15s) should be dirty, others clean
        assert cache.get_state(0) == SegmentState.CLEAN
        assert cache.get_state(1) == SegmentState.CLEAN
        assert cache.get_state(2) == SegmentState.DIRTY
        assert cache.get_state(3) == SegmentState.CLEAN

        # Verify the invalidated segment's file was deleted
        fname2 = segment_filename("full_flow", 10 * NS_PER_SEC, 15 * NS_PER_SEC)
        assert not (cache_dir / fname2).exists()


# ---------------------------------------------------------------------------
# Server Integration (requires aiohttp)
# ---------------------------------------------------------------------------


class TestServerIntegration:
    """Integration tests for the preview server with real cache and files."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_aiohttp(self) -> None:
        pytest.importorskip("aiohttp")

    def _make_cache_with_segments(self, cache_dir: Path, n_segments: int = 4) -> SegmentCache:
        """Create a SegmentCache with registered segments."""
        cache = SegmentCache(cache_dir, timeline_id="server_test")
        boundaries = _boundaries_as_tuples(
            compute_segment_boundaries(
                n_segments * 5 * NS_PER_SEC,
                segment_duration_ns=5 * NS_PER_SEC,
            )
        )
        cache.register_segments(boundaries)
        return cache

    @pytest.mark.asyncio
    async def test_server_with_real_cache(self, tmp_path: Path) -> None:
        """Create real SegmentCache with registered segments, create app,
        GET /api/status, verify counts match cache."""
        from aiohttp.test_utils import TestClient, TestServer

        from ave.preview.server import PreviewServer

        cache_dir = tmp_path / "cache"
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir(parents=True)

        cache = self._make_cache_with_segments(cache_dir)

        # Mark 2 segments clean (DIRTY -> RENDERING -> CLEAN)
        for idx in (0, 1):
            fp = cache_dir / f"seg_{idx}.mp4"
            fp.write_bytes(b"data")
            cache.mark_rendering(idx)
            cache.mark_clean(idx, fp)

        server = PreviewServer(cache, segments_dir)
        app = server.create_app()

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.status == 200
            data = await resp.json()
            expected = cache.segment_count()
            assert data["total"] == expected["total"]
            assert data["dirty"] == expected["dirty"]
            assert data["clean"] == expected["clean"]
            assert data["rendering"] == expected["rendering"]

    @requires_ffmpeg
    @pytest.mark.asyncio
    async def test_server_ws_frame_with_real_video(self, tmp_path: Path) -> None:
        """Create server with real test video, WS frame request,
        verify returned JPEG is valid (base64 decode, check magic bytes)."""
        if not TEST_VIDEO.exists():
            pytest.skip(f"Test fixture not found: {TEST_VIDEO}")

        from aiohttp.test_utils import TestClient, TestServer

        from ave.preview.server import PreviewServer

        cache_dir = tmp_path / "cache"
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir(parents=True)

        cache = self._make_cache_with_segments(cache_dir)
        server = PreviewServer(cache, segments_dir, video_path=TEST_VIDEO)
        app = server.create_app()

        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/ws")

            await ws.send_json({"type": "frame", "timestamp_ns": 1 * NS_PER_SEC})
            resp = await ws.receive_json()

            assert resp["type"] == "frame"
            assert resp["timestamp_ns"] == 1 * NS_PER_SEC
            assert resp["format"] == "jpeg"

            frame_bytes = base64.b64decode(resp["data"])
            assert len(frame_bytes) > 0
            assert frame_bytes[:3] == b"\xff\xd8\xff", "Response is not valid JPEG"

            await ws.close()

    @pytest.mark.asyncio
    async def test_server_segment_serving_with_real_file(self, tmp_path: Path) -> None:
        """Create real fMP4 file in segments_dir, GET /segments/filename,
        verify 200 with correct content."""
        from aiohttp.test_utils import TestClient, TestServer

        from ave.preview.server import PreviewServer

        cache_dir = tmp_path / "cache"
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir(parents=True)

        # Create a fake fMP4 file in the segments directory
        seg_content = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 100
        seg_name = "server_test_0_5000000000.mp4"
        (segments_dir / seg_name).write_bytes(seg_content)

        cache = self._make_cache_with_segments(cache_dir)
        server = PreviewServer(cache, segments_dir)
        app = server.create_app()

        async with TestClient(TestServer(app)) as client:
            resp = await client.get(f"/segments/{seg_name}")
            assert resp.status == 200
            body = await resp.read()
            assert body == seg_content
