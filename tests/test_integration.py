"""Integration test: full pipeline from source media to rendered proxy."""

from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg, requires_ges


@requires_ges
@requires_ffmpeg
@pytest.mark.slow
class TestFullPipeline:
    def test_ingest_to_timeline_to_render(self, fixtures_dir: Path, tmp_project: Path):
        """End-to-end: ingest source → add to timeline → render proxy."""
        from ave.ingest.probe import probe_media
        from ave.ingest.registry import AssetRegistry
        from ave.ingest.transcoder import ingest
        from ave.project.timeline import Timeline
        from ave.render.proxy import render_proxy

        # 1. Prepare source
        source = fixtures_dir / "av_clip_1080p24.mp4"
        if not source.exists():
            from tests.fixtures.generate import generate_av_clip
            generate_av_clip(source)

        # 2. Ingest
        registry = AssetRegistry(tmp_project / "assets" / "registry.json")
        entry = ingest(
            source=source,
            project_dir=tmp_project,
            asset_id="clip_001",
            registry=registry,
            project_fps=24.0,
        )

        assert entry.working_path.exists(), "Working intermediate should exist"
        assert entry.proxy_path.exists(), "Proxy should exist"

        # 3. Verify probe of working intermediate
        working_info = probe_media(entry.working_path)
        assert working_info.has_video
        assert working_info.video.codec == "dnxhd"

        # 4. Build timeline
        tl = Timeline.create(tmp_project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=entry.working_path,
            layer=0,
            start_ns=0,
            duration_ns=2 * 1_000_000_000,  # 2 seconds
        )
        tl.set_clip_metadata(clip_id, "agent:edit-intent", "Integration test clip")
        tl.set_metadata("agent:project-name", "Integration Test")
        tl.save()

        # 5. Render proxy
        export = tmp_project / "exports" / "integration_test.mp4"
        render_proxy(tmp_project / "project.xges", export, height=480)

        assert export.exists(), "Rendered proxy should exist"
        assert export.stat().st_size > 0, "Rendered proxy should have content"

        # 6. Verify rendered output
        output_info = probe_media(export)
        assert output_info.has_video
        assert output_info.video.height == 480
        assert output_info.duration_seconds > 0

        # 7. Verify registry persisted
        registry2 = AssetRegistry(tmp_project / "assets" / "registry.json")
        registry2.load()
        assert registry2.count() == 1
        assert registry2.get("clip_001").asset_id == "clip_001"
