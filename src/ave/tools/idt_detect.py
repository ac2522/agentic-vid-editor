"""IDT auto-detection from ffprobe color metadata.

Maps camera color metadata (color_space, color_transfer, color_primaries)
reported by ffprobe to OCIO Input Device Transform (IDT) names.

Pure logic — no Docker, GPU, or external dependencies required.
"""

from __future__ import annotations

from ave.ingest.probe import VideoStream
from ave.ingest.registry import AssetEntry


# ---------------------------------------------------------------------------
# Camera metadata -> OCIO IDT mapping table
# ---------------------------------------------------------------------------
# Keys: (color_space, color_transfer, color_primaries) as reported by ffprobe.
# Values: OCIO IDT color space name, or None for passthrough (no IDT needed).
#
# ffprobe reports these values from the container/codec metadata.  The exact
# strings depend on how the camera or encoder wrote them.  We map the most
# common representations.

IDT_MAP: dict[tuple[str | None, str | None, str | None], str | None] = {
    # -----------------------------------------------------------------------
    # Sony S-Log3 / S-Gamut3.Cine
    # -----------------------------------------------------------------------
    ("bt2020nc", "arib-std-b67", "bt2020"): "Sony S-Log3 S-Gamut3.Cine",

    # -----------------------------------------------------------------------
    # Canon C-Log2 / Cinema Gamut
    # -----------------------------------------------------------------------
    ("bt2020nc", "bt2020-10", "bt2020"): "Canon C-Log2 Cinema Gamut",

    # -----------------------------------------------------------------------
    # RED Log3G10 / REDWideGamutRGB
    # -----------------------------------------------------------------------
    ("bt2020nc", "smpte428", "bt2020"): "RED Log3G10 REDWideGamutRGB",

    # -----------------------------------------------------------------------
    # ARRI LogC3 / ALEXA Wide Gamut (ALEXA classic)
    # -----------------------------------------------------------------------
    ("bt2020nc", "bt2020-12", "bt2020"): "ARRI LogC3 AWG",

    # -----------------------------------------------------------------------
    # ARRI LogC4 / AWG4 (ALEXA 35)
    # -----------------------------------------------------------------------
    ("bt2020nc", "unknown", "bt2020"): "ARRI LogC4 AWG4",

    # -----------------------------------------------------------------------
    # Panasonic V-Log / V-Gamut
    # -----------------------------------------------------------------------
    ("bt2020nc", "arib-std-b67", "smpte431"): "Panasonic V-Log V-Gamut",

    # -----------------------------------------------------------------------
    # Blackmagic Film Gen 5
    # -----------------------------------------------------------------------
    ("bt2020nc", "smpte428", "smpte431"): "Blackmagic Film Gen 5",

    # -----------------------------------------------------------------------
    # Fujifilm F-Log2 / F-Gamut
    # -----------------------------------------------------------------------
    ("bt2020nc", "arib-std-b67", "smpte432"): "Fujifilm F-Log2 F-Gamut",

    # -----------------------------------------------------------------------
    # Nikon N-Log
    # -----------------------------------------------------------------------
    ("bt2020nc", "bt2020-10", "smpte432"): "Nikon N-Log",

    # -----------------------------------------------------------------------
    # Rec.709 — standard broadcast / web (no IDT needed)
    # -----------------------------------------------------------------------
    ("bt709", "bt709", "bt709"): None,

    # -----------------------------------------------------------------------
    # Rec.2020 PQ — HDR passthrough (no IDT needed)
    # -----------------------------------------------------------------------
    ("bt2020nc", "smpte2084", "bt2020"): None,

    # -----------------------------------------------------------------------
    # Rec.2020 HLG — HDR passthrough (no IDT needed)
    # -----------------------------------------------------------------------
    ("bt2020nc", "arib-std-b67", "bt709"): None,
}


# ---------------------------------------------------------------------------
# Detection function
# ---------------------------------------------------------------------------


def detect_idt(video_stream: VideoStream) -> str | None:
    """Detect OCIO IDT name from ffprobe color metadata.

    Looks up the combination of (color_space, color_transfer, color_primaries)
    in the IDT_MAP.

    Args:
        video_stream: A probed VideoStream with color metadata fields.

    Returns:
        OCIO IDT name string if a known camera profile is matched,
        or None if no match is found or the profile is a passthrough.
    """
    key = (
        video_stream.color_space,
        video_stream.color_transfer,
        video_stream.color_primaries,
    )

    if key not in IDT_MAP:
        return None

    return IDT_MAP[key]


# ---------------------------------------------------------------------------
# Auto-populate on ingest
# ---------------------------------------------------------------------------


def auto_detect_and_set_idt(
    entry: AssetEntry,
    video_stream: VideoStream,
) -> AssetEntry:
    """If IDT detected, return new entry with camera_color_space and idt_reference set.

    If not detected (unknown metadata or passthrough), returns entry unchanged.

    This is a pure function: the original entry is never mutated.

    Args:
        entry: The asset entry to potentially update.
        video_stream: Probed video stream with color metadata.

    Returns:
        A new AssetEntry with IDT fields populated, or the original entry
        if no IDT was detected.
    """
    idt_name = detect_idt(video_stream)

    if idt_name is None:
        return entry

    return entry.model_copy(
        update={
            "camera_color_space": idt_name,
            "camera_transfer": video_stream.color_transfer or entry.camera_transfer,
            "idt_reference": idt_name,
        }
    )
