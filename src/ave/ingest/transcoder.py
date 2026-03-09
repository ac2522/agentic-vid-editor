"""Ingest transcoder — any format to DNxHR HQX + H.264 proxy."""

import subprocess
from pathlib import Path

from ave.ingest.probe import probe_media
from ave.ingest.registry import AssetEntry, AssetRegistry
from ave.utils import fps_close


class TranscodeError(Exception):
    """Raised when transcoding fails."""


def transcode_to_working(
    source: Path,
    output: Path,
    codec: str = "dnxhd",
    profile: str = "dnxhr_hqx",
    target_fps: float | None = None,
) -> None:
    """Transcode source to working intermediate (DNxHR HQX in MXF or ProRes in MOV).

    Camera log encoding is PRESERVED — no color space conversion.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(source)]

    if target_fps is not None:
        cmd.extend(["-r", str(target_fps)])

    if codec == "dnxhd":
        cmd.extend(
            [
                "-c:v",
                "dnxhd",
                "-profile:v",
                profile,
                "-pix_fmt",
                "yuv422p10le",
            ]
        )
    elif codec == "prores":
        cmd.extend(
            [
                "-c:v",
                "prores_ks",
                "-profile:v",
                "3",  # HQ
                "-pix_fmt",
                "yuv422p10le",
            ]
        )
    else:
        raise TranscodeError(f"Unsupported codec: {codec}")

    # Copy audio
    cmd.extend(["-c:a", "pcm_s16le"])
    cmd.append(str(output))

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise TranscodeError(f"Working transcode failed: {e.stderr.decode()}") from e


def transcode_to_proxy(
    source: Path,
    output: Path,
    height: int = 480,
    target_fps: float | None = None,
) -> None:
    """Transcode source to lightweight H.264 proxy."""
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(source)]

    if target_fps is not None:
        cmd.extend(["-r", str(target_fps)])

    cmd.extend(
        [
            "-vf",
            f"scale=-2:{height}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output),
        ]
    )

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise TranscodeError(f"Proxy transcode failed: {e.stderr.decode()}") from e


def ingest(
    source: Path,
    project_dir: Path,
    asset_id: str,
    registry: AssetRegistry,
    project_fps: float = 24.0,
    codec: str = "dnxhd",
    profile: str = "dnxhr_hqx",
) -> AssetEntry:
    """Full ingest pipeline: probe → transcode working + proxy → register.

    Camera log encoding is preserved in the working intermediate.
    IDT reference is stored in registry for non-destructive application at render time.
    """
    # 1. Probe source
    info = probe_media(source)
    if not info.has_video:
        raise TranscodeError(f"Source has no video stream: {source}")

    # 2. Determine output paths
    suffix = ".mxf" if codec == "dnxhd" else ".mov"
    working_path = project_dir / "assets" / "media" / "working" / f"{asset_id}{suffix}"
    proxy_path = project_dir / "assets" / "media" / "proxy" / f"{asset_id}.mp4"

    # 3. Transcode to working intermediate (preserving camera log)
    needs_conform = not fps_close(info.video.fps, project_fps)
    transcode_to_working(
        source,
        working_path,
        codec=codec,
        profile=profile,
        target_fps=project_fps if needs_conform else None,
    )

    # 4. Transcode to proxy
    transcode_to_proxy(
        source,
        proxy_path,
        height=480,
        target_fps=project_fps if needs_conform else None,
    )

    # 5. Register
    entry = AssetEntry(
        asset_id=asset_id,
        original_path=source,
        working_path=working_path,
        proxy_path=proxy_path,
        original_fps=info.video.fps,
        conformed_fps=project_fps,
        duration_seconds=info.duration_seconds,
        width=info.video.width,
        height=info.video.height,
        codec=codec,
        camera_color_space=info.video.color_space or "unknown",
        camera_transfer=info.video.color_transfer or "unknown",
        idt_reference=None,  # User sets this based on camera
    )
    registry.add(entry)
    registry.save()

    return entry
