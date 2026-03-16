"""Tests for the segment cache manager."""

from pathlib import Path

import pytest

from ave.preview.cache import CacheError, SegmentCache, SegmentState


class TestSegmentStateEnum:
    def test_segment_state_enum(self):
        """SegmentState has DIRTY, RENDERING, CLEAN values."""
        assert SegmentState.DIRTY.value == "dirty"
        assert SegmentState.RENDERING.value == "rendering"
        assert SegmentState.CLEAN.value == "clean"
        assert len(SegmentState) == 3


class TestCacheInit:
    def test_cache_init_creates_directory(self, tmp_path: Path):
        """SegmentCache(cache_dir) creates the directory if it doesn't exist."""
        cache_dir = tmp_path / "cache" / "segments"
        assert not cache_dir.exists()
        SegmentCache(cache_dir)
        assert cache_dir.is_dir()


class TestCacheRegisterSegments:
    def test_cache_register_segments(self, tmp_path: Path):
        """Register segments from boundaries list, all start DIRTY."""
        cache = SegmentCache(tmp_path / "cache")
        boundaries = [
            (0, 0, 1_000_000_000),
            (1, 1_000_000_000, 2_000_000_000),
            (2, 2_000_000_000, 3_000_000_000),
        ]
        cache.register_segments(boundaries)
        for idx, start, end in boundaries:
            assert cache.get_state(idx) == SegmentState.DIRTY


class TestCacheGetState:
    def test_cache_get_state(self, tmp_path: Path):
        """Get state of registered segment returns DIRTY initially."""
        cache = SegmentCache(tmp_path / "cache")
        cache.register_segments([(0, 0, 1_000_000_000)])
        assert cache.get_state(0) == SegmentState.DIRTY

    def test_cache_get_state_unknown_segment(self, tmp_path: Path):
        """Unknown segment raises CacheError."""
        cache = SegmentCache(tmp_path / "cache")
        with pytest.raises(CacheError):
            cache.get_state(99)


class TestCacheTransitions:
    def test_cache_mark_rendering(self, tmp_path: Path):
        """Transition DIRTY -> RENDERING."""
        cache = SegmentCache(tmp_path / "cache")
        cache.register_segments([(0, 0, 1_000_000_000)])
        cache.mark_rendering(0)
        assert cache.get_state(0) == SegmentState.RENDERING

    def test_cache_mark_clean(self, tmp_path: Path):
        """Transition RENDERING -> CLEAN (requires file path)."""
        cache_dir = tmp_path / "cache"
        cache = SegmentCache(cache_dir)
        cache.register_segments([(0, 0, 1_000_000_000)])
        cache.mark_rendering(0)

        rendered_file = cache_dir / "segment_0.mp4"
        rendered_file.write_bytes(b"fake video data")

        cache.mark_clean(0, rendered_file)
        assert cache.get_state(0) == SegmentState.CLEAN

    def test_cache_mark_clean_without_file_raises(self, tmp_path: Path):
        """Marking clean without actual file raises CacheError."""
        cache = SegmentCache(tmp_path / "cache")
        cache.register_segments([(0, 0, 1_000_000_000)])
        cache.mark_rendering(0)

        nonexistent = tmp_path / "cache" / "does_not_exist.mp4"
        with pytest.raises(CacheError):
            cache.mark_clean(0, nonexistent)


class TestCacheInvalidation:
    def test_cache_invalidate_segment(self, tmp_path: Path):
        """Mark specific segment DIRTY, removes cached file if exists."""
        cache_dir = tmp_path / "cache"
        cache = SegmentCache(cache_dir)
        cache.register_segments([(0, 0, 1_000_000_000)])
        cache.mark_rendering(0)

        rendered_file = cache_dir / "segment_0.mp4"
        rendered_file.write_bytes(b"fake video data")
        cache.mark_clean(0, rendered_file)
        assert rendered_file.exists()

        cache.invalidate_segment(0)
        assert cache.get_state(0) == SegmentState.DIRTY
        assert not rendered_file.exists()

    def test_cache_invalidate_range(self, tmp_path: Path):
        """Invalidate all segments overlapping a time range."""
        cache = SegmentCache(tmp_path / "cache")
        cache.register_segments([
            (0, 0, 1_000_000_000),
            (1, 1_000_000_000, 2_000_000_000),
            (2, 2_000_000_000, 3_000_000_000),
            (3, 3_000_000_000, 4_000_000_000),
        ])
        # Mark all rendering then clean (with dummy files)
        for i in range(4):
            cache.mark_rendering(i)
            f = tmp_path / "cache" / f"seg_{i}.mp4"
            f.write_bytes(b"data")
            cache.mark_clean(i, f)

        # Invalidate range that overlaps segments 1 and 2
        # seg1: [1e9, 2e9), seg2: [2e9, 3e9)
        cache.invalidate_range(1_500_000_000, 2_500_000_000)

        assert cache.get_state(0) == SegmentState.CLEAN
        assert cache.get_state(1) == SegmentState.DIRTY
        assert cache.get_state(2) == SegmentState.DIRTY
        assert cache.get_state(3) == SegmentState.CLEAN

    def test_cache_invalidate_all(self, tmp_path: Path):
        """Invalidate all segments."""
        cache = SegmentCache(tmp_path / "cache")
        cache.register_segments([
            (0, 0, 1_000_000_000),
            (1, 1_000_000_000, 2_000_000_000),
        ])
        cache.mark_rendering(0)
        cache.invalidate_all()
        assert cache.get_state(0) == SegmentState.DIRTY
        assert cache.get_state(1) == SegmentState.DIRTY


class TestCacheDirtySegments:
    def test_cache_get_dirty_segments(self, tmp_path: Path):
        """Returns list of DIRTY segments sorted by index (priority)."""
        cache = SegmentCache(tmp_path / "cache")
        cache.register_segments([
            (0, 0, 1_000_000_000),
            (1, 1_000_000_000, 2_000_000_000),
            (2, 2_000_000_000, 3_000_000_000),
        ])
        cache.mark_rendering(1)

        dirty = cache.get_dirty_segments()
        assert len(dirty) == 2
        assert dirty[0].index == 0
        assert dirty[1].index == 2

    def test_cache_get_dirty_segments_viewport_priority(self, tmp_path: Path):
        """Segments overlapping viewport range returned first."""
        cache = SegmentCache(tmp_path / "cache")
        cache.register_segments([
            (0, 0, 1_000_000_000),
            (1, 1_000_000_000, 2_000_000_000),
            (2, 2_000_000_000, 3_000_000_000),
            (3, 3_000_000_000, 4_000_000_000),
        ])

        # Viewport covers segments 2 and 3
        dirty = cache.get_dirty_segments(
            viewport_start_ns=2_000_000_000,
            viewport_end_ns=4_000_000_000,
        )
        assert len(dirty) == 4
        # Viewport segments first (2, 3), then the rest (0, 1)
        assert dirty[0].index == 2
        assert dirty[1].index == 3
        assert dirty[2].index == 0
        assert dirty[3].index == 1


class TestCacheSegmentPath:
    def test_cache_get_clean_segment_path(self, tmp_path: Path):
        """Returns file path for CLEAN segment."""
        cache_dir = tmp_path / "cache"
        cache = SegmentCache(cache_dir)
        cache.register_segments([(0, 0, 1_000_000_000)])
        cache.mark_rendering(0)

        rendered_file = cache_dir / "segment_0.mp4"
        rendered_file.write_bytes(b"data")
        cache.mark_clean(0, rendered_file)

        assert cache.get_segment_path(0) == rendered_file

    def test_cache_get_clean_segment_path_not_clean(self, tmp_path: Path):
        """Raises CacheError if segment not CLEAN."""
        cache = SegmentCache(tmp_path / "cache")
        cache.register_segments([(0, 0, 1_000_000_000)])
        with pytest.raises(CacheError):
            cache.get_segment_path(0)


class TestCacheSegmentCount:
    def test_cache_segment_count(self, tmp_path: Path):
        """Reports total and per-state counts."""
        cache_dir = tmp_path / "cache"
        cache = SegmentCache(cache_dir)
        cache.register_segments([
            (0, 0, 1_000_000_000),
            (1, 1_000_000_000, 2_000_000_000),
            (2, 2_000_000_000, 3_000_000_000),
        ])
        cache.mark_rendering(1)
        f = cache_dir / "seg_2.mp4"
        f.write_bytes(b"data")
        cache.mark_rendering(2)
        cache.mark_clean(2, f)

        counts = cache.segment_count()
        assert counts == {"total": 3, "dirty": 1, "rendering": 1, "clean": 1}


class TestCachePersistence:
    def test_cache_persistence_save_load(self, tmp_path: Path):
        """Save cache state to JSON, reload, state preserved."""
        cache_dir = tmp_path / "cache"
        cache = SegmentCache(cache_dir, timeline_id="tl_42")
        cache.register_segments([
            (0, 0, 1_000_000_000),
            (1, 1_000_000_000, 2_000_000_000),
            (2, 2_000_000_000, 3_000_000_000),
        ])
        cache.mark_rendering(1)
        f = cache_dir / "seg_2.mp4"
        f.write_bytes(b"data")
        cache.mark_rendering(2)
        cache.mark_clean(2, f)

        state_file = tmp_path / "cache_state.json"
        cache.save_state(state_file)
        assert state_file.exists()

        loaded = SegmentCache.load_state(state_file, cache_dir)
        assert loaded.get_state(0) == SegmentState.DIRTY
        assert loaded.get_state(1) == SegmentState.RENDERING
        assert loaded.get_state(2) == SegmentState.CLEAN
        assert loaded.get_segment_path(2) == f
        assert loaded.segment_count() == {"total": 3, "dirty": 1, "rendering": 1, "clean": 1}
