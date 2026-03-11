"""Tests for render preset definitions and validation."""

from dataclasses import FrozenInstanceError

import pytest

from ave.render.presets import (
    PRESETS,
    PresetError,
    RenderPreset,
    get_preset,
    list_presets,
    validate_preset,
)

# ---------------------------------------------------------------------------
# Required preset names
# ---------------------------------------------------------------------------

REQUIRED_PRESETS = [
    "h264_web",
    "h265_archive",
    "prores_master",
    "dnxhr_master",
    "instagram_reel",
    "tiktok",
    "youtube_4k",
    "twitter_x",
]


class TestPresetsDict:
    """All 8+ presets exist in PRESETS dict."""

    @pytest.mark.parametrize("name", REQUIRED_PRESETS)
    def test_preset_exists(self, name: str) -> None:
        assert name in PRESETS

    def test_presets_count(self) -> None:
        assert len(PRESETS) >= 8

    def test_all_presets_are_render_preset_instances(self) -> None:
        for name, preset in PRESETS.items():
            assert isinstance(preset, RenderPreset), f"{name} is not a RenderPreset"


class TestListPresets:
    """list_presets() returns correct count with name/description."""

    def test_returns_list(self) -> None:
        result = list_presets()
        assert isinstance(result, list)

    def test_correct_count(self) -> None:
        result = list_presets()
        assert len(result) == len(PRESETS)

    def test_entries_have_name_and_description(self) -> None:
        for entry in list_presets():
            assert "name" in entry
            assert "description" in entry

    def test_all_names_present(self) -> None:
        names = {e["name"] for e in list_presets()}
        for name in REQUIRED_PRESETS:
            assert name in names


class TestGetPreset:
    """get_preset() returns correct presets or raises PresetError."""

    def test_get_h264_web(self) -> None:
        preset = get_preset("h264_web")
        assert preset.video_codec == "x264enc"
        assert preset.container == "mp4mux"
        assert preset.width == 1920
        assert preset.height == 1080

    def test_get_nonexistent_raises(self) -> None:
        with pytest.raises(PresetError):
            get_preset("nonexistent")


class TestValidatePreset:
    """validate_preset() validates presets correctly."""

    def test_valid_preset_returns_empty(self) -> None:
        preset = get_preset("h264_web")
        warnings = validate_preset(preset)
        assert warnings == []

    def test_all_builtin_presets_valid(self) -> None:
        for name, preset in PRESETS.items():
            warnings = validate_preset(preset)
            assert warnings == [], f"Preset {name} has warnings: {warnings}"

    def test_catches_mismatched_container_extension(self) -> None:
        bad_preset = RenderPreset(
            name="bad",
            description="bad preset",
            video_codec="x264enc",
            audio_codec="avenc_aac",
            container="mp4mux",
            file_extension=".mov",  # mismatch: mp4mux should use .mp4
            video_props={},
            audio_props={},
            width=1920,
            height=1080,
            fps=30.0,
        )
        warnings = validate_preset(bad_preset)
        assert len(warnings) > 0
        assert any("container" in w.lower() or "extension" in w.lower() for w in warnings)

    def test_catches_unknown_video_codec(self) -> None:
        bad_preset = RenderPreset(
            name="bad",
            description="bad preset",
            video_codec="unknown_codec",
            audio_codec="avenc_aac",
            container="mp4mux",
            file_extension=".mp4",
            video_props={},
            audio_props={},
            width=1920,
            height=1080,
            fps=30.0,
        )
        warnings = validate_preset(bad_preset)
        assert len(warnings) > 0

    def test_catches_zero_width(self) -> None:
        bad_preset = RenderPreset(
            name="bad",
            description="bad preset",
            video_codec="x264enc",
            audio_codec="avenc_aac",
            container="mp4mux",
            file_extension=".mp4",
            video_props={},
            audio_props={},
            width=0,
            height=1080,
            fps=30.0,
        )
        warnings = validate_preset(bad_preset)
        assert len(warnings) > 0

    def test_catches_negative_height(self) -> None:
        bad_preset = RenderPreset(
            name="bad",
            description="bad preset",
            video_codec="x264enc",
            audio_codec="avenc_aac",
            container="mp4mux",
            file_extension=".mp4",
            video_props={},
            audio_props={},
            width=1920,
            height=-100,
            fps=30.0,
        )
        warnings = validate_preset(bad_preset)
        assert len(warnings) > 0

    def test_catches_empty_name(self) -> None:
        bad_preset = RenderPreset(
            name="",
            description="bad preset",
            video_codec="x264enc",
            audio_codec="avenc_aac",
            container="mp4mux",
            file_extension=".mp4",
            video_props={},
            audio_props={},
            width=1920,
            height=1080,
            fps=30.0,
        )
        warnings = validate_preset(bad_preset)
        assert len(warnings) > 0


class TestSpecificPresets:
    """Test specific preset properties."""

    def test_instagram_reel_vertical(self) -> None:
        preset = get_preset("instagram_reel")
        assert preset.width == 1080
        assert preset.height == 1920

    def test_tiktok_vertical(self) -> None:
        preset = get_preset("tiktok")
        assert preset.width == 1080
        assert preset.height == 1920

    def test_prores_master_qtmux_mov(self) -> None:
        preset = get_preset("prores_master")
        assert preset.container == "qtmux"
        assert preset.file_extension == ".mov"

    def test_youtube_4k_dimensions(self) -> None:
        preset = get_preset("youtube_4k")
        assert preset.width == 3840
        assert preset.height == 2160

    def test_twitter_x_dimensions(self) -> None:
        preset = get_preset("twitter_x")
        assert preset.width == 1280
        assert preset.height == 720

    def test_all_presets_have_nonempty_name_and_description(self) -> None:
        for name, preset in PRESETS.items():
            assert preset.name, f"Preset {name} has empty name"
            assert preset.description, f"Preset {name} has empty description"


class TestRenderPresetFrozen:
    """RenderPreset is frozen (immutable)."""

    def test_frozen(self) -> None:
        preset = get_preset("h264_web")
        with pytest.raises(FrozenInstanceError):
            preset.name = "changed"  # type: ignore[misc]

    def test_frozen_width(self) -> None:
        preset = get_preset("h264_web")
        with pytest.raises(FrozenInstanceError):
            preset.width = 999  # type: ignore[misc]
