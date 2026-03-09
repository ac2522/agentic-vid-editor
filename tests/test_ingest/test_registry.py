"""Tests for asset registry — JSON metadata store for ingested media."""

from pathlib import Path

import pytest


class TestAssetRegistry:
    def test_create_empty_registry(self, tmp_project: Path):
        from ave.ingest.registry import AssetRegistry

        reg = AssetRegistry(tmp_project / "assets" / "registry.json")

        assert reg.count() == 0

    def test_add_asset(self, tmp_project: Path):
        from ave.ingest.registry import AssetRegistry, AssetEntry

        reg = AssetRegistry(tmp_project / "assets" / "registry.json")
        entry = AssetEntry(
            asset_id="clip_001",
            original_path=Path("/media/raw/take1.mov"),
            working_path=tmp_project / "assets" / "media" / "working" / "take1.mxf",
            proxy_path=tmp_project / "assets" / "media" / "proxy" / "take1.mp4",
            original_fps=59.94,
            conformed_fps=24.0,
            duration_seconds=120.5,
            width=3840,
            height=2160,
            codec="dnxhd",
            camera_color_space="V-Gamut",
            camera_transfer="V-Log",
            idt_reference="aces_vlog_to_ap1",
        )

        reg.add(entry)

        assert reg.count() == 1
        assert reg.get("clip_001") == entry

    def test_save_and_load(self, tmp_project: Path):
        from ave.ingest.registry import AssetRegistry, AssetEntry

        registry_path = tmp_project / "assets" / "registry.json"
        reg = AssetRegistry(registry_path)
        reg.add(
            AssetEntry(
                asset_id="clip_001",
                original_path=Path("/media/raw/take1.mov"),
                working_path=Path("working/take1.mxf"),
                proxy_path=Path("proxy/take1.mp4"),
                original_fps=24.0,
                conformed_fps=24.0,
                duration_seconds=60.0,
                width=1920,
                height=1080,
                codec="dnxhd",
                camera_color_space="V-Gamut",
                camera_transfer="V-Log",
                idt_reference="aces_vlog_to_ap1",
            )
        )
        reg.save()

        # Load into a new instance
        reg2 = AssetRegistry(registry_path)
        reg2.load()

        assert reg2.count() == 1
        loaded = reg2.get("clip_001")
        assert loaded.camera_color_space == "V-Gamut"
        assert loaded.idt_reference == "aces_vlog_to_ap1"

    def test_remove_asset(self, tmp_project: Path):
        from ave.ingest.registry import AssetRegistry, AssetEntry

        reg = AssetRegistry(tmp_project / "assets" / "registry.json")
        reg.add(
            AssetEntry(
                asset_id="clip_001",
                original_path=Path("/media/raw/take1.mov"),
                working_path=Path("working/take1.mxf"),
                proxy_path=Path("proxy/take1.mp4"),
                original_fps=24.0,
                conformed_fps=24.0,
                duration_seconds=60.0,
                width=1920,
                height=1080,
                codec="dnxhd",
                camera_color_space="Rec709",
                camera_transfer="bt709",
                idt_reference=None,
            )
        )

        reg.remove("clip_001")
        assert reg.count() == 0

    def test_get_nonexistent_raises(self, tmp_project: Path):
        from ave.ingest.registry import AssetRegistry

        reg = AssetRegistry(tmp_project / "assets" / "registry.json")

        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_list_all(self, tmp_project: Path):
        from ave.ingest.registry import AssetRegistry, AssetEntry

        reg = AssetRegistry(tmp_project / "assets" / "registry.json")
        for i in range(3):
            reg.add(
                AssetEntry(
                    asset_id=f"clip_{i:03d}",
                    original_path=Path(f"/media/raw/take{i}.mov"),
                    working_path=Path(f"working/take{i}.mxf"),
                    proxy_path=Path(f"proxy/take{i}.mp4"),
                    original_fps=24.0,
                    conformed_fps=24.0,
                    duration_seconds=60.0,
                    width=1920,
                    height=1080,
                    codec="dnxhd",
                    camera_color_space="V-Gamut",
                    camera_transfer="V-Log",
                    idt_reference="aces_vlog_to_ap1",
                )
            )

        entries = reg.list_all()
        assert len(entries) == 3
        assert {e.asset_id for e in entries} == {"clip_000", "clip_001", "clip_002"}
