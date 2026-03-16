"""Render preset definitions and validation.

Pure logic layer: no GES dependency. Defines codec/container/resolution
presets that the GES execution layer (export.py) will consume.
"""

from __future__ import annotations

from dataclasses import dataclass


class PresetError(Exception):
    """Raised when a preset is not found or invalid."""


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderPreset:
    """Immutable render preset definition."""

    name: str
    description: str
    video_codec: str  # GStreamer element: "x264enc", "x265enc", etc.
    audio_codec: str  # "avenc_aac", "flacenc", "wavenc"
    container: str  # "mp4mux", "matroskamux", "qtmux", "mxfmux"
    file_extension: str  # ".mp4", ".mkv", ".mov", ".mxf"
    video_props: dict  # codec-specific: bitrate, profile, quality, etc.
    audio_props: dict  # sample-rate, bitrate, etc.
    width: int | None  # None = source resolution
    height: int | None
    fps: float | None  # None = source fps


# ---------------------------------------------------------------------------
# Known codecs and container/extension mappings
# ---------------------------------------------------------------------------

KNOWN_VIDEO_CODECS = {
    "x264enc",
    "x265enc",
    "avenc_prores_ks",
    "avenc_dnxhd",
    "vp9enc",
    "av1enc",
}

KNOWN_AUDIO_CODECS = {
    "avenc_aac",
    "flacenc",
    "wavenc",
    "opusenc",
    "lamemp3enc",
}

# Maps container element -> expected file extensions
CONTAINER_EXTENSIONS: dict[str, set[str]] = {
    "mp4mux": {".mp4", ".m4v"},
    "qtmux": {".mov", ".mp4"},
    "matroskamux": {".mkv", ".webm"},
    "mxfmux": {".mxf"},
    "webmmux": {".webm"},
    "oggmux": {".ogg", ".ogv"},
    "avimux": {".avi"},
}

# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, RenderPreset] = {
    "h264_web": RenderPreset(
        name="h264_web",
        description="H.264 1080p for web (YouTube/Vimeo)",
        video_codec="x264enc",
        audio_codec="avenc_aac",
        container="mp4mux",
        file_extension=".mp4",
        video_props={"bitrate": 8000, "profile": "high", "bframes": 2},
        audio_props={"bitrate": 192000, "rate": 48000},
        width=1920,
        height=1080,
        fps=None,
    ),
    "h265_archive": RenderPreset(
        name="h265_archive",
        description="H.265 high-quality archive (source resolution, CRF)",
        video_codec="x265enc",
        audio_codec="avenc_aac",
        container="mp4mux",
        file_extension=".mp4",
        video_props={"crf": 18, "tune": "ssim"},
        audio_props={"bitrate": 256000, "rate": 48000},
        width=None,
        height=None,
        fps=None,
    ),
    "prores_master": RenderPreset(
        name="prores_master",
        description="ProRes 422 HQ master (source resolution)",
        video_codec="avenc_prores_ks",
        audio_codec="avenc_aac",
        container="qtmux",
        file_extension=".mov",
        video_props={"profile": 3},  # 3 = ProRes 422 HQ
        audio_props={"bitrate": 256000, "rate": 48000},
        width=None,
        height=None,
        fps=None,
    ),
    "dnxhr_master": RenderPreset(
        name="dnxhr_master",
        description="DNxHR HQX master (source resolution)",
        video_codec="avenc_dnxhd",
        audio_codec="avenc_aac",
        container="mxfmux",
        file_extension=".mxf",
        video_props={"profile": "dnxhr_hqx"},
        audio_props={"bitrate": 256000, "rate": 48000},
        width=None,
        height=None,
        fps=None,
    ),
    "instagram_reel": RenderPreset(
        name="instagram_reel",
        description="Instagram Reel — H.264 1080x1920 vertical, 30fps",
        video_codec="x264enc",
        audio_codec="avenc_aac",
        container="mp4mux",
        file_extension=".mp4",
        video_props={"bitrate": 5000, "profile": "high", "bframes": 2},
        audio_props={"bitrate": 128000, "rate": 44100},
        width=1080,
        height=1920,
        fps=30.0,
    ),
    "tiktok": RenderPreset(
        name="tiktok",
        description="TikTok — H.264 1080x1920 vertical, 30fps",
        video_codec="x264enc",
        audio_codec="avenc_aac",
        container="mp4mux",
        file_extension=".mp4",
        video_props={"bitrate": 6000, "profile": "high", "bframes": 2},
        audio_props={"bitrate": 128000, "rate": 44100},
        width=1080,
        height=1920,
        fps=30.0,
    ),
    "youtube_4k": RenderPreset(
        name="youtube_4k",
        description="YouTube 4K — H.265 3840x2160, high bitrate",
        video_codec="x265enc",
        audio_codec="avenc_aac",
        container="mp4mux",
        file_extension=".mp4",
        video_props={"bitrate": 40000, "tune": "ssim"},
        audio_props={"bitrate": 256000, "rate": 48000},
        width=3840,
        height=2160,
        fps=None,
    ),
    "twitter_x": RenderPreset(
        name="twitter_x",
        description="Twitter/X — H.264 1280x720, mp4",
        video_codec="x264enc",
        audio_codec="avenc_aac",
        container="mp4mux",
        file_extension=".mp4",
        video_props={"bitrate": 5000, "profile": "main"},
        audio_props={"bitrate": 128000, "rate": 44100},
        width=1280,
        height=720,
        fps=None,
    ),
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def list_presets() -> list[dict]:
    """Return [{name, description}, ...] for all presets."""
    return [{"name": preset.name, "description": preset.description} for preset in PRESETS.values()]


def get_preset(name: str) -> RenderPreset:
    """Get a preset by name.

    Raises:
        PresetError: If the preset name is not found.
    """
    if name not in PRESETS:
        raise PresetError(f"Unknown preset: {name!r}")
    return PRESETS[name]


def validate_preset(preset: RenderPreset) -> list[str]:
    """Check a preset for issues.

    Returns a list of warning strings. An empty list means the preset is valid.
    """
    warnings: list[str] = []

    # Name and description must be non-empty
    if not preset.name:
        warnings.append("Preset name is empty")
    if not preset.description:
        warnings.append("Preset description is empty")

    # Video codec must be known
    if preset.video_codec not in KNOWN_VIDEO_CODECS:
        warnings.append(f"Unknown video codec: {preset.video_codec!r}")

    # Audio codec must be known
    if preset.audio_codec not in KNOWN_AUDIO_CODECS:
        warnings.append(f"Unknown audio codec: {preset.audio_codec!r}")

    # Container/extension must match
    if preset.container in CONTAINER_EXTENSIONS:
        valid_exts = CONTAINER_EXTENSIONS[preset.container]
        if preset.file_extension not in valid_exts:
            warnings.append(
                f"Container {preset.container!r} does not match "
                f"extension {preset.file_extension!r} "
                f"(expected one of {sorted(valid_exts)})"
            )
    else:
        warnings.append(f"Unknown container: {preset.container!r}")

    # Dimensions must be positive if specified
    if preset.width is not None and preset.width <= 0:
        warnings.append(f"Width must be positive, got {preset.width}")
    if preset.height is not None and preset.height <= 0:
        warnings.append(f"Height must be positive, got {preset.height}")

    # FPS must be positive if specified
    if preset.fps is not None and preset.fps <= 0:
        warnings.append(f"FPS must be positive, got {preset.fps}")

    return warnings
