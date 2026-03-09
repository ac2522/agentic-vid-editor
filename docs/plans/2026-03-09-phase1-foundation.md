# Phase 1: Foundation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the foundational infrastructure — Docker container, ingest pipeline, GES timeline interface, and proxy rendering — so that all subsequent phases have a tested, working base.

**Architecture:** GES/XGES is the project format. GStreamer >= 1.28.1 is the rendering engine. CUDA+OpenGL is the GPU pipeline. DNxHR HQX in MXF is the default intermediate codec. Camera log encoding is preserved; IDTs are applied non-destructively at render time. All code is Python with GObject Introspection bindings for GES/GStreamer. libplacebo and OCIO GStreamer elements are C prototypes.

**Tech Stack:** Python 3.12, GES (via PyGObject), GStreamer 1.28.1, FFmpeg 8.0, Docker + NVIDIA Container Toolkit, pytest, pydantic

**Reference:** Architecture at `docs/plans/2026-03-08-architecture-design-v2.md`

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/ave/__init__.py`
- Create: `src/ave/ingest/__init__.py`
- Create: `src/ave/project/__init__.py`
- Create: `src/ave/render/__init__.py`
- Create: `src/ave/tools/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `CLAUDE.md`
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "ave"
version = "0.1.0"
description = "Agentic Video Editor — agent-driven video editing framework"
requires-python = ">=3.12"
dependencies = [
    "PyGObject>=3.50.0",
    "pydantic>=2.10",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=6.0",
    "ruff>=0.9.0",
]

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: marks tests that require NVIDIA GPU (deselect with '-m \"not gpu\"')",
    "slow: marks tests that take >10 seconds (deselect with '-m \"not slow\"')",
]

[tool.ruff]
target-version = "py312"
line-length = 100
```

**Step 2: Create package structure**

```python
# src/ave/__init__.py
"""Agentic Video Editor — agent-driven video editing framework."""

__version__ = "0.1.0"
```

```python
# src/ave/ingest/__init__.py
```

```python
# src/ave/project/__init__.py
```

```python
# src/ave/render/__init__.py
```

```python
# src/ave/tools/__init__.py
```

```python
# tests/__init__.py
```

**Step 3: Create tests/conftest.py**

```python
"""Shared test fixtures for ave test suite."""

import os
import subprocess
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _ges_available() -> bool:
    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GES", "1.0")
        from gi.repository import Gst, GES
        Gst.init(None)
        GES.init()
        return True
    except (ImportError, ValueError):
        return False


def _gpu_available() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, check=True, text=True
        )
        return "NVIDIA" in result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# Skip markers
requires_ffmpeg = pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg not found")
requires_ges = pytest.mark.skipif(not _ges_available(), reason="GES not available")
requires_gpu = pytest.mark.skipif(not _gpu_available(), reason="NVIDIA GPU not available")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to test fixtures directory. Generates fixtures if missing."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory structure."""
    project = tmp_path / "test_project"
    (project / "assets" / "media" / "working").mkdir(parents=True)
    (project / "assets" / "media" / "proxy").mkdir(parents=True)
    (project / "cache" / "segments").mkdir(parents=True)
    (project / "cache" / "thumbnails").mkdir(parents=True)
    (project / "luts").mkdir(parents=True)
    (project / "transcriptions").mkdir(parents=True)
    (project / "exports").mkdir(parents=True)
    return project
```

**Step 4: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
*.egg
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
tests/fixtures/*.mp4
tests/fixtures/*.mxf
tests/fixtures/*.mov
tests/fixtures/*.wav
tests/fixtures/*.ts
*.xges
!tests/fixtures/*.xges
.env
```

**Step 5: Create CLAUDE.md**

```markdown
# Agentic Video Editor (ave)

## Project Overview
Agent-driven video editing framework. GES/XGES project format, GStreamer rendering engine, CUDA+OpenGL GPU pipeline.

## Architecture
See `docs/plans/2026-03-08-architecture-design-v2.md` for full architecture.

## Development
- Python 3.12+ with PyGObject for GES/GStreamer bindings
- Run inside Docker container (see `docker/Dockerfile`) for GStreamer >= 1.28.1
- Package: `src/ave/` — install with `pip install -e ".[dev]"`
- Tests: `pytest` — use `-m "not gpu"` to skip GPU tests outside container

## Key Conventions
- All tools are pure functions: explicit inputs, result output, no side effects beyond output path
- Camera log encoding preserved in intermediates; IDTs applied non-destructively at render time
- Default intermediate: DNxHR HQX in MXF. ProRes 422 HQ in MOV as option.
- GES metadata uses `agent:` prefix for agent-specific keys
- TDD: write failing test first, then implement

## Test Markers
- `@pytest.mark.gpu` — requires NVIDIA GPU
- `@pytest.mark.slow` — takes >10 seconds
- `@requires_ffmpeg` — requires FFmpeg binary
- `@requires_ges` — requires GES/GStreamer Python bindings
```

**Step 6: Install package and verify**

Run: `cd /home/zaia/Development/agentic-vid-editor && pip install -e ".[dev]"`
Expected: Successful installation, `import ave` works

**Step 7: Run empty test suite**

Run: `cd /home/zaia/Development/agentic-vid-editor && pytest -v`
Expected: "no tests ran" or collected 0 items (no errors)

**Step 8: Commit**

```bash
git init
git add pyproject.toml src/ tests/ CLAUDE.md .gitignore
git commit -m "feat: project scaffolding with ave package structure and test harness"
```

---

## Task 2: Docker Development Container

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/docker-compose.yml`
- Create: `docker/entrypoint.sh`

**Step 1: Create docker/Dockerfile**

This builds GStreamer 1.28.1 from source with nvcodec, GES, and OpenGL support.

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV GST_VERSION=1.28.1

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-venv python3-gi python3-gi-cairo \
    gir1.2-glib-2.0 gir1.2-gst-plugins-base-1.0 \
    meson ninja-build cmake pkg-config flex bison nasm \
    libglib2.0-dev libgudev-1.0-dev libmount-dev \
    libgl-dev libegl-dev libgles-dev libdrm-dev libgbm-dev \
    libx11-dev libxext-dev libxv-dev libxi-dev libxfixes-dev \
    libwayland-dev libpulse-dev libasound2-dev \
    libopus-dev libvpx-dev libx264-dev libx265-dev \
    libfdk-aac-dev libmp3lame-dev libvorbis-dev libtheora-dev \
    libsrt-gnutls-dev libsoup-3.0-dev libjson-glib-dev \
    libcairo2-dev libpango1.0-dev librsvg2-dev \
    libavcodec-dev libavformat-dev libavutil-dev libavfilter-dev libswscale-dev \
    libplacebo-dev libplacebo254 \
    libopencolorio-dev \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Build GStreamer 1.28.1 monorepo (includes GES, nvcodec, GL plugins)
WORKDIR /build
RUN git clone --branch ${GST_VERSION} --depth 1 \
    https://gitlab.freedesktop.org/gstreamer/gstreamer.git

WORKDIR /build/gstreamer
RUN meson setup build \
    --prefix=/usr/local \
    --buildtype=release \
    -Dgpl=enabled \
    -Dgst-plugins-bad:nvcodec=enabled \
    -Dgst-plugins-base:gl=enabled \
    -Dgst-plugins-base:gl_api=opengl,gles2 \
    -Dgst-plugins-base:gl_platform=egl,glx \
    -Dges=enabled \
    -Dgst-editing-services:pygi-overrides-dir=/usr/lib/python3/dist-packages/gi/overrides \
    -Dintrospection=enabled \
    -Dpython=enabled \
    -Dgst-plugins-rs=disabled \
    -Ddevtools=disabled \
    -Ddoc=disabled \
    -Dexamples=disabled \
    -Dtests=disabled \
    && ninja -C build -j$(nproc) \
    && ninja -C build install

# Update library paths
RUN ldconfig

# --- Runtime stage ---
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-gi python3-gi-cairo \
    gir1.2-glib-2.0 \
    libgl1 libegl1 libgles2 libdrm2 libgbm1 \
    libx11-6 libxext6 libxv1 \
    libpulse0 libasound2t64 \
    libopus0 libvpx9 libx264-164 libx265-209 \
    libfdk-aac2 libmp3lame0 libvorbis0a libtheora0 \
    libsrt1.5-gnutls libsoup-3.0-0 libjson-glib-1.0-0 \
    libcairo2 libpango-1.0-0 librsvg2-2 \
    libplacebo254 \
    libopencolorio2.3 \
    ffmpeg \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy built GStreamer from builder
COPY --from=builder /usr/local /usr/local
RUN ldconfig

# Set up GStreamer environment
ENV GST_PLUGIN_PATH=/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0
ENV GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0
ENV LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV GST_GL_API=opengl3

# Install ave package
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --break-system-packages -e ".[dev]"

# Copy test code
COPY tests/ tests/
COPY CLAUDE.md .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["pytest", "-v"]
```

**Step 2: Create docker/entrypoint.sh**

```bash
#!/bin/bash
set -e

# Verify GStreamer installation
echo "=== GStreamer Version ==="
gst-inspect-1.0 --version

# Verify GES is available
python3 -c "
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GES', '1.0')
from gi.repository import Gst, GES
Gst.init(None)
GES.init()
print(f'GES initialized: Gst {Gst.version_string()}')
" || echo "WARNING: GES not available"

# Verify nvcodec if GPU present
if command -v nvidia-smi &> /dev/null; then
    echo "=== NVIDIA GPU ==="
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    gst-inspect-1.0 nvcodec 2>/dev/null | head -3 || echo "nvcodec plugin not found"
fi

exec "$@"
```

**Step 3: Create docker/docker-compose.yml**

```yaml
services:
  ave:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - DISPLAY=${DISPLAY:-}
    volumes:
      - ../src:/app/src
      - ../tests:/app/tests
      - ../docs:/app/docs
      - /tmp/.X11-unix:/tmp/.X11-unix:ro
    working_dir: /app
    # Override default CMD for interactive use
    command: bash
    stdin_open: true
    tty: true

  test:
    extends: ave
    command: pytest -v -m "not slow"
```

**Step 4: Verify Dockerfile syntax**

Run: `cd /home/zaia/Development/agentic-vid-editor && docker compose -f docker/docker-compose.yml config`
Expected: Valid compose config output (no syntax errors)

**Step 5: Commit**

```bash
git add docker/
git commit -m "feat: Docker container with GStreamer 1.28.1, nvcodec, GES, OpenGL"
```

---

## Task 3: Test Fixture Generation

**Files:**
- Create: `tests/fixtures/generate.py`
- Test: `tests/test_fixtures.py`

**Step 1: Write the test for fixture generation**

```python
# tests/test_fixtures.py
"""Tests that verify test fixtures can be generated and are valid."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg


@requires_ffmpeg
class TestFixtureGeneration:
    def test_generate_color_bars_h264(self, fixtures_dir: Path):
        """Generate 5-second 1080p24 color bars in H.264."""
        output = fixtures_dir / "color_bars_1080p24.mp4"
        if output.exists():
            output.unlink()

        from tests.fixtures.generate import generate_color_bars

        generate_color_bars(output, duration=5, width=1920, height=1080, fps=24)

        assert output.exists()
        probe = _probe(output)
        video = _video_stream(probe)
        assert video["width"] == 1920
        assert video["height"] == 1080
        assert video["codec_name"] == "h264"
        assert float(probe["format"]["duration"]) == pytest.approx(5.0, abs=0.5)

    def test_generate_color_bars_30fps(self, fixtures_dir: Path):
        """Generate 3-second 720p30 color bars for mixed-fps testing."""
        output = fixtures_dir / "color_bars_720p30.mp4"
        if output.exists():
            output.unlink()

        from tests.fixtures.generate import generate_color_bars

        generate_color_bars(output, duration=3, width=1280, height=720, fps=30)

        assert output.exists()
        probe = _probe(output)
        video = _video_stream(probe)
        assert video["width"] == 1280
        assert video["height"] == 720

    def test_generate_test_tone(self, fixtures_dir: Path):
        """Generate 5-second 1kHz test tone."""
        output = fixtures_dir / "test_tone_1khz.wav"
        if output.exists():
            output.unlink()

        from tests.fixtures.generate import generate_test_tone

        generate_test_tone(output, frequency=1000, duration=5)

        assert output.exists()
        probe = _probe(output)
        audio = _audio_stream(probe)
        assert audio["codec_name"] == "pcm_s16le"
        assert int(audio["sample_rate"]) == 48000

    def test_generate_av_clip(self, fixtures_dir: Path):
        """Generate clip with both video and audio."""
        output = fixtures_dir / "av_clip_1080p24.mp4"
        if output.exists():
            output.unlink()

        from tests.fixtures.generate import generate_av_clip

        generate_av_clip(output, duration=5, width=1920, height=1080, fps=24)

        assert output.exists()
        probe = _probe(output)
        streams = probe["streams"]
        codec_types = {s["codec_type"] for s in streams}
        assert "video" in codec_types
        assert "audio" in codec_types


def _probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _video_stream(probe: dict) -> dict:
    return next(s for s in probe["streams"] if s["codec_type"] == "video")


def _audio_stream(probe: dict) -> dict:
    return next(s for s in probe["streams"] if s["codec_type"] == "audio")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_fixtures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.fixtures.generate'`

**Step 3: Write fixture generation module**

```python
# tests/fixtures/generate.py
"""Generate test media fixtures using FFmpeg.

All fixtures are synthetic (color bars, test tones) so tests don't depend
on real media files. Generated files are git-ignored.
"""

import subprocess
from pathlib import Path


def generate_color_bars(
    output: Path,
    duration: int = 5,
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
) -> None:
    """Generate SMPTE color bars as H.264 MP4."""
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            str(output),
        ],
        capture_output=True, check=True,
    )


def generate_test_tone(
    output: Path,
    frequency: int = 1000,
    duration: int = 5,
    sample_rate: int = 48000,
) -> None:
    """Generate sine wave test tone as WAV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency={frequency}:sample_rate={sample_rate}:duration={duration}",
            "-c:a", "pcm_s16le",
            str(output),
        ],
        capture_output=True, check=True,
    )


def generate_av_clip(
    output: Path,
    duration: int = 5,
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
    audio_freq: int = 440,
) -> None:
    """Generate clip with SMPTE color bars + sine tone."""
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}",
            "-f", "lavfi",
            "-i", f"sine=frequency={audio_freq}:sample_rate=48000:duration={duration}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            str(output),
        ],
        capture_output=True, check=True,
    )
```

Also create `tests/fixtures/__init__.py`:

```python
# tests/fixtures/__init__.py
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fixtures.py -v`
Expected: 4 passed (or skipped if no FFmpeg)

**Step 5: Commit**

```bash
git add tests/fixtures/ tests/test_fixtures.py
git commit -m "feat: test fixture generation — color bars, test tones, AV clips"
```

---

## Task 4: Media Probe Utility

**Files:**
- Create: `src/ave/ingest/probe.py`
- Test: `tests/test_ingest/test_probe.py`

**Step 1: Write failing tests**

```python
# tests/test_ingest/__init__.py
```

```python
# tests/test_ingest/test_probe.py
"""Tests for media probe utility."""

from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg


@requires_ffmpeg
class TestProbe:
    @pytest.fixture(autouse=True)
    def _setup_fixture(self, fixtures_dir: Path):
        """Ensure color bars fixture exists."""
        self.fixture = fixtures_dir / "color_bars_1080p24.mp4"
        if not self.fixture.exists():
            from tests.fixtures.generate import generate_color_bars
            generate_color_bars(self.fixture)

    def test_probe_returns_media_info(self):
        from ave.ingest.probe import probe_media

        info = probe_media(self.fixture)

        assert info.path == self.fixture
        assert info.duration_seconds > 0
        assert info.has_video
        assert info.video is not None

    def test_probe_video_stream(self):
        from ave.ingest.probe import probe_media

        info = probe_media(self.fixture)

        assert info.video.width == 1920
        assert info.video.height == 1080
        assert info.video.codec == "h264"
        assert info.video.fps == pytest.approx(24.0, abs=0.1)
        assert info.video.pix_fmt == "yuv420p"

    def test_probe_audio_stream(self, fixtures_dir: Path):
        av_clip = fixtures_dir / "av_clip_1080p24.mp4"
        if not av_clip.exists():
            from tests.fixtures.generate import generate_av_clip
            generate_av_clip(av_clip)

        from ave.ingest.probe import probe_media

        info = probe_media(av_clip)

        assert info.has_audio
        assert info.audio is not None
        assert info.audio.sample_rate == 48000
        assert info.audio.channels >= 1

    def test_probe_nonexistent_file(self):
        from ave.ingest.probe import probe_media, ProbeError

        with pytest.raises(ProbeError, match="does not exist"):
            probe_media(Path("/nonexistent/file.mp4"))

    def test_probe_invalid_file(self, tmp_path: Path):
        bad_file = tmp_path / "not_media.txt"
        bad_file.write_text("this is not media")

        from ave.ingest.probe import probe_media, ProbeError

        with pytest.raises(ProbeError, match="ffprobe failed"):
            probe_media(bad_file)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest/test_probe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ave.ingest.probe'`

**Step 3: Write probe implementation**

```python
# src/ave/ingest/probe.py
"""Media probe utility wrapping ffprobe."""

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class ProbeError(Exception):
    """Raised when media probing fails."""


@dataclass(frozen=True)
class VideoStream:
    width: int
    height: int
    codec: str
    pix_fmt: str
    fps: float
    bit_depth: int
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    duration_seconds: float


@dataclass(frozen=True)
class AudioStream:
    codec: str
    sample_rate: int
    channels: int
    channel_layout: str | None
    duration_seconds: float


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    format_name: str
    duration_seconds: float
    size_bytes: int
    video: VideoStream | None
    audio: AudioStream | None

    @property
    def has_video(self) -> bool:
        return self.video is not None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None


def probe_media(path: Path) -> MediaInfo:
    """Probe a media file and return structured metadata.

    Raises ProbeError if the file doesn't exist or can't be probed.
    """
    if not path.exists():
        raise ProbeError(f"File does not exist: {path}")

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise ProbeError(f"ffprobe failed for {path}: {e.stderr}") from e

    data = json.loads(result.stdout)

    if "format" not in data:
        raise ProbeError(f"ffprobe returned no format info for {path}")

    fmt = data["format"]
    streams = data.get("streams", [])

    video = _parse_video_stream(streams)
    audio = _parse_audio_stream(streams)

    return MediaInfo(
        path=path,
        format_name=fmt.get("format_name", "unknown"),
        duration_seconds=float(fmt.get("duration", 0)),
        size_bytes=int(fmt.get("size", 0)),
        video=video,
        audio=audio,
    )


def _parse_fps(stream: dict) -> float:
    """Parse frame rate from stream, trying r_frame_rate then avg_frame_rate."""
    for key in ("r_frame_rate", "avg_frame_rate"):
        raw = stream.get(key, "0/1")
        if "/" in raw:
            frac = Fraction(raw)
            if frac > 0:
                return float(frac)
    return 0.0


def _parse_video_stream(streams: list[dict]) -> VideoStream | None:
    for s in streams:
        if s.get("codec_type") == "video":
            return VideoStream(
                width=int(s.get("width", 0)),
                height=int(s.get("height", 0)),
                codec=s.get("codec_name", "unknown"),
                pix_fmt=s.get("pix_fmt", "unknown"),
                fps=_parse_fps(s),
                bit_depth=int(s.get("bits_per_raw_sample", 8) or 8),
                color_space=s.get("color_space"),
                color_transfer=s.get("color_transfer"),
                color_primaries=s.get("color_primaries"),
                duration_seconds=float(s.get("duration", 0) or 0),
            )
    return None


def _parse_audio_stream(streams: list[dict]) -> AudioStream | None:
    for s in streams:
        if s.get("codec_type") == "audio":
            return AudioStream(
                codec=s.get("codec_name", "unknown"),
                sample_rate=int(s.get("sample_rate", 0)),
                channels=int(s.get("channels", 0)),
                channel_layout=s.get("channel_layout"),
                duration_seconds=float(s.get("duration", 0) or 0),
            )
    return None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest/test_probe.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add src/ave/ingest/probe.py tests/test_ingest/
git commit -m "feat: media probe utility wrapping ffprobe"
```

---

## Task 5: Asset Registry

**Files:**
- Create: `src/ave/ingest/registry.py`
- Test: `tests/test_ingest/test_registry.py`

**Step 1: Write failing tests**

```python
# tests/test_ingest/test_registry.py
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
        reg.add(AssetEntry(
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
        ))
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
        reg.add(AssetEntry(
            asset_id="clip_001",
            original_path=Path("/media/raw/take1.mov"),
            working_path=Path("working/take1.mxf"),
            proxy_path=Path("proxy/take1.mp4"),
            original_fps=24.0, conformed_fps=24.0,
            duration_seconds=60.0, width=1920, height=1080,
            codec="dnxhd", camera_color_space="Rec709",
            camera_transfer="bt709", idt_reference=None,
        ))

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
            reg.add(AssetEntry(
                asset_id=f"clip_{i:03d}",
                original_path=Path(f"/media/raw/take{i}.mov"),
                working_path=Path(f"working/take{i}.mxf"),
                proxy_path=Path(f"proxy/take{i}.mp4"),
                original_fps=24.0, conformed_fps=24.0,
                duration_seconds=60.0, width=1920, height=1080,
                codec="dnxhd", camera_color_space="V-Gamut",
                camera_transfer="V-Log", idt_reference="aces_vlog_to_ap1",
            ))

        entries = reg.list_all()
        assert len(entries) == 3
        assert {e.asset_id for e in entries} == {"clip_000", "clip_001", "clip_002"}
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write registry implementation**

```python
# src/ave/ingest/registry.py
"""Asset registry — JSON metadata store for ingested media."""

import json
from pathlib import Path

from pydantic import BaseModel


class AssetEntry(BaseModel):
    """Metadata for a single ingested media asset."""

    asset_id: str
    original_path: Path
    working_path: Path
    proxy_path: Path
    original_fps: float
    conformed_fps: float
    duration_seconds: float
    width: int
    height: int
    codec: str
    camera_color_space: str
    camera_transfer: str
    idt_reference: str | None = None
    transcription_path: Path | None = None
    visual_analysis_path: Path | None = None


class AssetRegistry:
    """JSON-backed registry of ingested media assets."""

    def __init__(self, path: Path):
        self._path = path
        self._entries: dict[str, AssetEntry] = {}
        if path.exists():
            self.load()

    def add(self, entry: AssetEntry) -> None:
        self._entries[entry.asset_id] = entry

    def get(self, asset_id: str) -> AssetEntry:
        return self._entries[asset_id]

    def remove(self, asset_id: str) -> None:
        del self._entries[asset_id]

    def list_all(self) -> list[AssetEntry]:
        return list(self._entries.values())

    def count(self) -> int:
        return len(self._entries)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.model_dump(mode="json") for k, v in self._entries.items()}
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def load(self) -> None:
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text())
        self._entries = {k: AssetEntry(**v) for k, v in raw.items()}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest/test_registry.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add src/ave/ingest/registry.py tests/test_ingest/test_registry.py
git commit -m "feat: asset registry — JSON metadata store for ingested media"
```

---

## Task 6: Ingest Transcoder

**Files:**
- Create: `src/ave/ingest/transcoder.py`
- Test: `tests/test_ingest/test_transcoder.py`

**Step 1: Write failing tests**

```python
# tests/test_ingest/test_transcoder.py
"""Tests for ingest transcoder — any format to DNxHR HQX + proxy."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg


def _probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _video_stream(probe: dict) -> dict:
    return next(s for s in probe["streams"] if s["codec_type"] == "video")


@requires_ffmpeg
class TestTranscoder:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.source = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.source.exists():
            from tests.fixtures.generate import generate_av_clip
            generate_av_clip(self.source)
        self.project = tmp_project

    def test_transcode_to_dnxhr(self):
        from ave.ingest.transcoder import transcode_to_working

        output = self.project / "assets" / "media" / "working" / "clip.mxf"
        transcode_to_working(self.source, output, codec="dnxhd", profile="dnxhr_hqx")

        assert output.exists()
        probe = _probe(output)
        video = _video_stream(probe)
        assert video["codec_name"] == "dnxhd"
        assert video["width"] == 1920
        assert video["height"] == 1080

    def test_transcode_to_proxy(self):
        from ave.ingest.transcoder import transcode_to_proxy

        output = self.project / "assets" / "media" / "proxy" / "clip.mp4"
        transcode_to_proxy(self.source, output, height=480)

        assert output.exists()
        probe = _probe(output)
        video = _video_stream(probe)
        assert video["codec_name"] == "h264"
        assert video["height"] == 480

    def test_transcode_with_fps_conforming(self, fixtures_dir: Path):
        """30fps source conformed to 24fps project."""
        source_30 = fixtures_dir / "color_bars_720p30.mp4"
        if not source_30.exists():
            from tests.fixtures.generate import generate_color_bars
            generate_color_bars(source_30, duration=3, width=1280, height=720, fps=30)

        from ave.ingest.transcoder import transcode_to_working
        from fractions import Fraction

        output = self.project / "assets" / "media" / "working" / "conformed.mxf"
        transcode_to_working(
            source_30, output, codec="dnxhd", profile="dnxhr_hqx", target_fps=24,
        )

        assert output.exists()
        probe = _probe(output)
        video = _video_stream(probe)
        fps_str = video.get("r_frame_rate", "0/1")
        fps = float(Fraction(fps_str))
        assert fps == pytest.approx(24.0, abs=0.1)

    @pytest.mark.slow
    def test_full_ingest(self):
        """Full ingest: probe, transcode working + proxy, register."""
        from ave.ingest.transcoder import ingest
        from ave.ingest.registry import AssetRegistry

        registry = AssetRegistry(self.project / "assets" / "registry.json")

        entry = ingest(
            source=self.source,
            project_dir=self.project,
            asset_id="clip_001",
            registry=registry,
            project_fps=24.0,
        )

        assert entry.asset_id == "clip_001"
        assert entry.working_path.exists()
        assert entry.proxy_path.exists()
        assert registry.count() == 1
        assert registry.get("clip_001") == entry
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest/test_transcoder.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write transcoder implementation**

```python
# src/ave/ingest/transcoder.py
"""Ingest transcoder — any format to DNxHR HQX + H.264 proxy."""

import subprocess
from pathlib import Path

from ave.ingest.probe import MediaInfo, probe_media
from ave.ingest.registry import AssetEntry, AssetRegistry


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
        cmd.extend([
            "-c:v", "dnxhd",
            "-profile:v", profile,
            "-pix_fmt", "yuv422p10le",
        ])
    elif codec == "prores":
        cmd.extend([
            "-c:v", "prores_ks",
            "-profile:v", "3",  # HQ
            "-pix_fmt", "yuv422p10le",
        ])
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

    cmd.extend([
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output),
    ])

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
    needs_conform = info.video.fps != project_fps
    transcode_to_working(
        source, working_path,
        codec=codec, profile=profile,
        target_fps=project_fps if needs_conform else None,
    )

    # 4. Transcode to proxy
    transcode_to_proxy(
        source, proxy_path, height=480,
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest/test_transcoder.py -v`
Expected: 4 passed (test_full_ingest may be slow)

**Step 5: Commit**

```bash
git add src/ave/ingest/transcoder.py tests/test_ingest/test_transcoder.py
git commit -m "feat: ingest transcoder — DNxHR HQX working + H.264 proxy + registry"
```

---

## Task 7: GES Timeline Interface

**Files:**
- Create: `src/ave/project/timeline.py`
- Test: `tests/test_project/__init__.py`
- Test: `tests/test_project/test_timeline.py`

**Step 1: Write failing tests**

```python
# tests/test_project/__init__.py
```

```python
# tests/test_project/test_timeline.py
"""Tests for GES timeline interface."""

from pathlib import Path

import pytest

from tests.conftest import requires_ges, requires_ffmpeg


@requires_ges
class TestTimelineCreate:
    def test_create_empty_timeline(self, tmp_project: Path):
        from ave.project.timeline import Timeline

        tl = Timeline.create(tmp_project / "project.xges", fps=24.0)

        assert tl.fps == 24.0
        assert tl.duration_ns == 0
        assert tl.clip_count == 0

    def test_save_and_load(self, tmp_project: Path):
        from ave.project.timeline import Timeline

        xges_path = tmp_project / "project.xges"
        tl = Timeline.create(xges_path, fps=24.0)
        tl.set_metadata("agent:project-name", "Test Project")
        tl.save()

        assert xges_path.exists()

        tl2 = Timeline.load(xges_path)
        assert tl2.fps == 24.0
        assert tl2.get_metadata("agent:project-name") == "Test Project"


@requires_ges
@requires_ffmpeg
class TestTimelineClips:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip
            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_add_clip(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)

        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )

        assert clip_id is not None
        assert tl.clip_count == 1
        assert tl.duration_ns > 0

    def test_add_clip_with_inpoint(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)

        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            inpoint_ns=1_000_000_000,  # start 1s into clip
            duration_ns=2 * 1_000_000_000,
        )

        assert clip_id is not None
        assert tl.clip_count == 1

    def test_remove_clip(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )

        tl.remove_clip(clip_id)
        assert tl.clip_count == 0

    def test_add_multiple_clips_sequentially(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        dur = 2 * 1_000_000_000

        tl.add_clip(media_path=self.clip_path, layer=0, start_ns=0, duration_ns=dur)
        tl.add_clip(media_path=self.clip_path, layer=0, start_ns=dur, duration_ns=dur)

        assert tl.clip_count == 2

    def test_clip_metadata(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )

        tl.set_clip_metadata(clip_id, "agent:edit-intent", "Opening shot")
        value = tl.get_clip_metadata(clip_id, "agent:edit-intent")
        assert value == "Opening shot"

    def test_save_load_with_clips(self):
        from ave.project.timeline import Timeline

        xges_path = self.project / "project.xges"
        tl = Timeline.create(xges_path, fps=24.0)
        tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )
        tl.save()

        tl2 = Timeline.load(xges_path)
        assert tl2.clip_count == 1


@requires_ges
@requires_ffmpeg
class TestTimelineEffects:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip
            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_add_effect_to_clip(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )

        effect_id = tl.add_effect(clip_id, "videobalance")
        assert effect_id is not None

    def test_set_effect_property(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )
        effect_id = tl.add_effect(clip_id, "videobalance")

        tl.set_effect_property(clip_id, effect_id, "saturation", 0.5)
        value = tl.get_effect_property(clip_id, effect_id, "saturation")
        assert value == pytest.approx(0.5, abs=0.01)

    def test_remove_effect(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )
        effect_id = tl.add_effect(clip_id, "videobalance")
        tl.remove_effect(clip_id, effect_id)

        # No exception means success — clip has no effects
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project/test_timeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write timeline implementation**

```python
# src/ave/project/timeline.py
"""GES timeline interface for agent-friendly timeline manipulation."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GES", "1.0")

from gi.repository import GES, GLib, Gst  # noqa: E402

# Initialize GStreamer and GES once
Gst.init(None)
GES.init()


class TimelineError(Exception):
    """Raised when timeline operations fail."""


class Timeline:
    """High-level wrapper around a GES timeline."""

    def __init__(self, ges_timeline: GES.Timeline, path: Path, fps: float):
        self._timeline = ges_timeline
        self._path = path
        self._fps = fps
        self._clips: dict[str, GES.Clip] = {}
        self._next_clip_id = 0

    @classmethod
    def create(cls, path: Path, fps: float = 24.0) -> Timeline:
        """Create a new empty timeline with audio and video tracks."""
        timeline = GES.Timeline.new_audio_video()
        if timeline is None:
            raise TimelineError("Failed to create GES timeline")

        # Add initial layer
        timeline.append_layer()

        tl = cls(timeline, path, fps)
        return tl

    @classmethod
    def load(cls, path: Path) -> Timeline:
        """Load a timeline from an XGES file."""
        if not path.exists():
            raise TimelineError(f"XGES file not found: {path}")

        timeline = GES.Timeline.new()
        uri = _path_to_uri(path)

        project = GES.Project.new(uri)
        timeline = project.extract()

        if timeline is None:
            raise TimelineError(f"Failed to load timeline from {path}")

        # Detect fps from video track restriction caps
        fps = 24.0  # default
        for track in timeline.get_tracks():
            if track.get_property("track-type") == GES.TrackType.VIDEO:
                caps = track.get_restriction_caps()
                if caps and caps.get_size() > 0:
                    structure = caps.get_structure(0)
                    ok, num, den = structure.get_fraction("framerate")
                    if ok and den > 0:
                        fps = num / den
                break

        tl = cls(timeline, path, fps)

        # Re-index existing clips
        for layer in timeline.get_layers():
            for clip in layer.get_clips():
                clip_id = f"clip_{tl._next_clip_id:04d}"
                tl._clips[clip_id] = clip
                tl._next_clip_id += 1

        return tl

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def duration_ns(self) -> int:
        return self._timeline.get_duration()

    @property
    def clip_count(self) -> int:
        count = 0
        for layer in self._timeline.get_layers():
            count += len(layer.get_clips())
        return count

    def add_clip(
        self,
        media_path: Path,
        layer: int = 0,
        start_ns: int = 0,
        duration_ns: int | None = None,
        inpoint_ns: int = 0,
    ) -> str:
        """Add a media clip to the timeline. Returns clip ID."""
        uri = _path_to_uri(media_path)

        asset = GES.UriClipAsset.request_sync(uri)
        if asset is None:
            raise TimelineError(f"Failed to load asset: {media_path}")

        if duration_ns is None:
            duration_ns = asset.get_duration()

        # Ensure layer exists
        layers = self._timeline.get_layers()
        while len(layers) <= layer:
            self._timeline.append_layer()
            layers = self._timeline.get_layers()

        target_layer = layers[layer]
        clip = target_layer.add_asset(
            asset, start_ns, inpoint_ns, duration_ns, GES.TrackType.UNKNOWN,
        )

        if clip is None:
            raise TimelineError(f"Failed to add clip at start={start_ns}")

        clip_id = f"clip_{self._next_clip_id:04d}"
        self._clips[clip_id] = clip
        self._next_clip_id += 1

        return clip_id

    def remove_clip(self, clip_id: str) -> None:
        """Remove a clip from the timeline."""
        clip = self._get_clip(clip_id)
        layer = clip.get_layer()
        if layer is None:
            raise TimelineError(f"Clip {clip_id} has no layer")
        layer.remove_clip(clip)
        del self._clips[clip_id]

    def add_effect(self, clip_id: str, element_description: str) -> str:
        """Add a GStreamer effect to a clip. Returns effect ID."""
        clip = self._get_clip(clip_id)
        effect = GES.Effect.new(element_description)
        if effect is None:
            raise TimelineError(f"Failed to create effect: {element_description}")

        if not clip.add(effect):
            raise TimelineError(f"Failed to add effect to {clip_id}")

        effect_id = f"{clip_id}_fx_{element_description.split()[0]}"
        return effect_id

    def remove_effect(self, clip_id: str, effect_id: str) -> None:
        """Remove an effect from a clip."""
        clip = self._get_clip(clip_id)
        element_name = effect_id.split("_fx_")[-1]

        for child in clip.get_children(False):
            if isinstance(child, GES.Effect):
                desc = child.get_property("bin-description")
                if desc and desc.split()[0] == element_name:
                    clip.remove(child)
                    return

        raise TimelineError(f"Effect {effect_id} not found on {clip_id}")

    def set_effect_property(
        self, clip_id: str, effect_id: str, prop_name: str, value: object
    ) -> None:
        """Set a property on an effect."""
        clip = self._get_clip(clip_id)
        element_name = effect_id.split("_fx_")[-1]

        for child in clip.get_children(False):
            if isinstance(child, GES.Effect):
                desc = child.get_property("bin-description")
                if desc and desc.split()[0] == element_name:
                    child.set_child_property(prop_name, value)
                    return

        raise TimelineError(f"Effect {effect_id} not found on {clip_id}")

    def get_effect_property(
        self, clip_id: str, effect_id: str, prop_name: str
    ) -> object:
        """Get a property from an effect."""
        clip = self._get_clip(clip_id)
        element_name = effect_id.split("_fx_")[-1]

        for child in clip.get_children(False):
            if isinstance(child, GES.Effect):
                desc = child.get_property("bin-description")
                if desc and desc.split()[0] == element_name:
                    ok, value = child.get_child_property(prop_name)
                    if ok:
                        return value
                    raise TimelineError(f"Property {prop_name} not found")

        raise TimelineError(f"Effect {effect_id} not found on {clip_id}")

    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata on the timeline."""
        self._timeline.set_meta(key, value)

    def get_metadata(self, key: str) -> str | None:
        """Get metadata from the timeline."""
        return self._timeline.get_meta(key)

    def set_clip_metadata(self, clip_id: str, key: str, value: str) -> None:
        """Set metadata on a clip."""
        clip = self._get_clip(clip_id)
        clip.set_meta(key, value)

    def get_clip_metadata(self, clip_id: str, key: str) -> str | None:
        """Get metadata from a clip."""
        clip = self._get_clip(clip_id)
        return clip.get_meta(key)

    def save(self) -> None:
        """Save timeline to XGES file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        uri = _path_to_uri(self._path)
        if not self._timeline.save_to_uri(uri, None, True):
            raise TimelineError(f"Failed to save timeline to {self._path}")

    def _get_clip(self, clip_id: str) -> GES.Clip:
        if clip_id not in self._clips:
            raise TimelineError(f"Clip not found: {clip_id}")
        return self._clips[clip_id]


def _path_to_uri(path: Path) -> str:
    """Convert a Path to a file URI."""
    abs_path = str(path.resolve())
    return "file://" + quote(abs_path, safe="/")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_project/test_timeline.py -v`
Expected: All tests pass (or skip if GES not available)

**Step 5: Commit**

```bash
git add src/ave/project/timeline.py tests/test_project/
git commit -m "feat: GES timeline interface — clips, effects, metadata, save/load"
```

---

## Task 8: Proxy Render via GES Pipeline

**Files:**
- Create: `src/ave/render/proxy.py`
- Test: `tests/test_render/__init__.py`
- Test: `tests/test_render/test_proxy.py`

**Step 1: Write failing tests**

```python
# tests/test_render/__init__.py
```

```python
# tests/test_render/test_proxy.py
"""Tests for proxy rendering via GES pipeline."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg, requires_ges


def _probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _video_stream(probe: dict) -> dict:
    return next(s for s in probe["streams"] if s["codec_type"] == "video")


@requires_ges
@requires_ffmpeg
class TestProxyRender:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip
            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_render_single_clip(self):
        from ave.project.timeline import Timeline
        from ave.render.proxy import render_proxy

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )
        tl.save()

        output = self.project / "exports" / "proxy.mp4"
        render_proxy(self.project / "project.xges", output, height=480)

        assert output.exists()
        probe = _probe(output)
        video = _video_stream(probe)
        assert video["height"] == 480
        assert video["codec_name"] == "h264"
        assert float(probe["format"]["duration"]) > 0

    def test_render_creates_valid_output(self):
        from ave.project.timeline import Timeline
        from ave.render.proxy import render_proxy

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=1 * 1_000_000_000,
        )
        tl.save()

        output = self.project / "exports" / "test_render.mp4"
        render_proxy(self.project / "project.xges", output)

        assert output.exists()
        assert output.stat().st_size > 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_render/test_proxy.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write proxy render implementation**

```python
# src/ave/render/proxy.py
"""Proxy rendering via GES pipeline."""

from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GES", "1.0")
gi.require_version("GstPbutils", "1.0")

from gi.repository import GES, GLib, Gst, GstPbutils  # noqa: E402

from ave.project.timeline import _path_to_uri  # noqa: E402

Gst.init(None)
GES.init()


class RenderError(Exception):
    """Raised when rendering fails."""


def render_proxy(
    xges_path: Path,
    output_path: Path,
    height: int = 480,
    video_bitrate: int = 2_000_000,
) -> None:
    """Render an XGES timeline to an H.264 proxy MP4.

    Uses GES.Pipeline for native GStreamer rendering.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    uri = _path_to_uri(xges_path)
    project = GES.Project.new(uri)
    timeline = project.extract()

    if timeline is None:
        raise RenderError(f"Failed to load timeline from {xges_path}")

    pipeline = GES.Pipeline()
    pipeline.set_timeline(timeline)

    # Build encoding profile: H.264 video + AAC audio in MP4
    container_profile = GstPbutils.EncodingContainerProfile.new(
        "mp4", None,
        Gst.Caps.from_string("video/quicktime,variant=iso"),
        None,
    )

    video_caps = Gst.Caps.from_string(
        f"video/x-h264,height={height}"
    )
    video_profile = GstPbutils.EncodingVideoProfile.new(
        video_caps, None,
        Gst.Caps.from_string(f"video/x-raw,height={height}"),
        0,
    )
    container_profile.add_profile(video_profile)

    audio_caps = Gst.Caps.from_string("audio/mpeg,mpegversion=4")
    audio_profile = GstPbutils.EncodingAudioProfile.new(
        audio_caps, None, None, 0,
    )
    container_profile.add_profile(audio_profile)

    output_uri = _path_to_uri(output_path)
    pipeline.set_render_settings(output_uri, container_profile)
    pipeline.set_mode(GES.PipelineFlags.RENDER)

    # Run the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    bus = pipeline.get_bus()
    while True:
        msg = bus.timed_pop_filtered(
            Gst.CLOCK_TIME_NONE,
            Gst.MessageType.EOS | Gst.MessageType.ERROR,
        )
        if msg.type == Gst.MessageType.EOS:
            break
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            pipeline.set_state(Gst.State.NULL)
            raise RenderError(f"Render failed: {err.message}\n{debug}")

    pipeline.set_state(Gst.State.NULL)

    if not output_path.exists():
        raise RenderError(f"Render completed but output not found: {output_path}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_render/test_proxy.py -v`
Expected: All tests pass (or skip if GES not available)

**Step 5: Commit**

```bash
git add src/ave/render/proxy.py tests/test_render/
git commit -m "feat: proxy render via GES pipeline — H.264 MP4 output"
```

---

## Task 9: Integration Test — Full Pipeline

**Files:**
- Test: `tests/test_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_integration.py
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
```

**Step 2: Run the integration test**

Run: `pytest tests/test_integration.py -v -m slow`
Expected: 1 passed (or skipped without GES/FFmpeg)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration test — full ingest → timeline → render pipeline"
```

---

## Task 10: libplacebo GStreamer Element Prototype (C)

> This is a C GStreamer plugin. It runs inside the Docker container. This task outlines the architecture and minimal prototype; full production implementation is Phase 3.

**Files:**
- Create: `plugins/gst-libplacebo/meson.build`
- Create: `plugins/gst-libplacebo/gstplacebofilter.h`
- Create: `plugins/gst-libplacebo/gstplacebofilter.c`
- Create: `plugins/gst-libplacebo/plugin.c`

**Step 1: Create meson.build**

```meson
# plugins/gst-libplacebo/meson.build
project('gst-libplacebo', 'c',
  version : '0.1.0',
  default_options : ['warning_level=2'])

gst_dep = dependency('gstreamer-1.0', version : '>= 1.28.1')
gst_gl_dep = dependency('gstreamer-gl-1.0', version : '>= 1.28.1')
gst_video_dep = dependency('gstreamer-video-1.0', version : '>= 1.28.1')
placebo_dep = dependency('libplacebo', version : '>= 7.349')

gstplacebo = shared_library('gstplacebo',
  'gstplacebofilter.c',
  'plugin.c',
  dependencies : [gst_dep, gst_gl_dep, gst_video_dep, placebo_dep],
  install : true,
  install_dir : join_paths(get_option('libdir'), 'gstreamer-1.0'),
)
```

**Step 2: Create header**

```c
// plugins/gst-libplacebo/gstplacebofilter.h
#ifndef __GST_PLACEBO_FILTER_H__
#define __GST_PLACEBO_FILTER_H__

#include <gst/gl/gstglfilter.h>
#include <libplacebo/opengl.h>
#include <libplacebo/renderer.h>
#include <libplacebo/log.h>

G_BEGIN_DECLS

#define GST_TYPE_PLACEBO_FILTER (gst_placebo_filter_get_type())
G_DECLARE_FINAL_TYPE(GstPlaceboFilter, gst_placebo_filter, GST, PLACEBO_FILTER, GstGLFilter)

struct _GstPlaceboFilter {
  GstGLFilter parent;

  /* libplacebo state */
  pl_log pl_log;
  pl_opengl pl_gl;
  pl_renderer pl_renderer;
  pl_tex src_tex;
  pl_tex dst_tex;

  /* Properties */
  gchar *lut_path;
};

G_END_DECLS

#endif /* __GST_PLACEBO_FILTER_H__ */
```

**Step 3: Create minimal filter implementation**

This prototype implements a passthrough filter with optional 3D LUT application. Full color management, tone mapping, and scaling are Phase 3.

```c
// plugins/gst-libplacebo/gstplacebofilter.c
/*
 * GstPlaceboFilter — libplacebo OpenGL backend filter for GStreamer.
 *
 * Phase 1 prototype: passthrough + basic 3D LUT application.
 * Phase 3: full color management, tone mapping, HDR, scaling.
 *
 * Reference: FFmpeg vf_libplacebo.c (Vulkan-only, 1845 lines).
 * This element uses the OpenGL backend instead.
 */

#include "gstplacebofilter.h"
#include <gst/gl/gstglfuncs.h>

GST_DEBUG_CATEGORY_STATIC(gst_placebo_filter_debug);
#define GST_CAT_DEFAULT gst_placebo_filter_debug

enum {
  PROP_0,
  PROP_LUT_PATH,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:GLMemory),format=RGBA"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:GLMemory),format=RGBA"));

G_DEFINE_TYPE(GstPlaceboFilter, gst_placebo_filter, GST_TYPE_GL_FILTER);

static void gst_placebo_filter_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(object);
  switch (prop_id) {
    case PROP_LUT_PATH:
      g_free(self->lut_path);
      self->lut_path = g_value_dup_string(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void gst_placebo_filter_get_property(GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(object);
  switch (prop_id) {
    case PROP_LUT_PATH:
      g_value_set_string(value, self->lut_path);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static gboolean gst_placebo_filter_gl_start(GstGLFilter *filter) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(filter);

  self->pl_log = pl_log_create(PL_API_VER, pl_log_params(
      .log_cb = NULL,
      .log_level = PL_LOG_WARN,
  ));

  self->pl_gl = pl_opengl_create(self->pl_log, pl_opengl_params(
      .allow_software = false,
  ));

  if (!self->pl_gl) {
    GST_ERROR_OBJECT(self, "Failed to create libplacebo OpenGL context");
    return FALSE;
  }

  self->pl_renderer = pl_renderer_create(self->pl_log, self->pl_gl->gpu);

  GST_INFO_OBJECT(self, "libplacebo OpenGL backend initialized (GPU: %s)",
      self->pl_gl->gpu->glsl.version ? "yes" : "no");

  return TRUE;
}

static void gst_placebo_filter_gl_stop(GstGLFilter *filter) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(filter);

  pl_tex_destroy(self->pl_gl->gpu, &self->src_tex);
  pl_tex_destroy(self->pl_gl->gpu, &self->dst_tex);
  pl_renderer_destroy(&self->pl_renderer);
  pl_opengl_destroy(&self->pl_gl);
  pl_log_destroy(&self->pl_log);
}

static gboolean gst_placebo_filter_filter_texture(GstGLFilter *filter,
    GstGLMemory *input, GstGLMemory *output) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(filter);

  /* Phase 1: passthrough — copy input to output via libplacebo renderer.
   * Phase 3 will add: LUT application, tone mapping, color management. */

  GstVideoInfo *in_info = &GST_GL_FILTER(self)->in_info;
  int w = GST_VIDEO_INFO_WIDTH(in_info);
  int h = GST_VIDEO_INFO_HEIGHT(in_info);

  /* Wrap GStreamer GL textures as libplacebo textures */
  struct pl_opengl_wrap_params src_wrap = {
      .width = w, .height = h,
      .texture = gst_gl_memory_get_texture_id(input),
      .target = GL_TEXTURE_2D,
      .iformat = GL_RGBA8,
  };
  struct pl_opengl_wrap_params dst_wrap = {
      .width = w, .height = h,
      .texture = gst_gl_memory_get_texture_id(output),
      .target = GL_TEXTURE_2D,
      .iformat = GL_RGBA8,
  };

  pl_tex src = pl_opengl_wrap(self->pl_gl->gpu, &src_wrap);
  pl_tex dst = pl_opengl_wrap(self->pl_gl->gpu, &dst_wrap);

  if (!src || !dst) {
    GST_ERROR_OBJECT(self, "Failed to wrap GL textures");
    if (src) pl_tex_destroy(self->pl_gl->gpu, &src);
    if (dst) pl_tex_destroy(self->pl_gl->gpu, &dst);
    return FALSE;
  }

  /* Render: passthrough for now */
  struct pl_frame img = {
      .num_planes = 1,
      .planes = {{ .texture = src,
                    .components = 4,
                    .component_mapping = {0, 1, 2, 3} }},
      .repr = pl_color_repr_sdtv,
      .color = pl_color_space_srgb,
  };

  struct pl_frame target = {
      .num_planes = 1,
      .planes = {{ .texture = dst,
                    .components = 4,
                    .component_mapping = {0, 1, 2, 3} }},
      .repr = pl_color_repr_sdtv,
      .color = pl_color_space_srgb,
  };

  struct pl_render_params params = pl_render_default_params;

  gboolean ok = pl_render_image(self->pl_renderer, &img, &target, &params);

  pl_tex_destroy(self->pl_gl->gpu, &src);
  pl_tex_destroy(self->pl_gl->gpu, &dst);

  return ok;
}

static void gst_placebo_filter_finalize(GObject *object) {
  GstPlaceboFilter *self = GST_PLACEBO_FILTER(object);
  g_free(self->lut_path);
  G_OBJECT_CLASS(gst_placebo_filter_parent_class)->finalize(object);
}

static void gst_placebo_filter_class_init(GstPlaceboFilterClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GstGLFilterClass *filter_class = GST_GL_FILTER_CLASS(klass);

  gobject_class->set_property = gst_placebo_filter_set_property;
  gobject_class->get_property = gst_placebo_filter_get_property;
  gobject_class->finalize = gst_placebo_filter_finalize;

  g_object_class_install_property(gobject_class, PROP_LUT_PATH,
      g_param_spec_string("lut-path", "LUT Path",
          "Path to a .cube 3D LUT file", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata(element_class,
      "libplacebo Filter", "Filter/Video",
      "Color processing via libplacebo OpenGL backend",
      "Agentic Video Editor <ave@example.com>");

  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);

  filter_class->gl_start = gst_placebo_filter_gl_start;
  filter_class->gl_stop = gst_placebo_filter_gl_stop;
  filter_class->filter_texture = gst_placebo_filter_filter_texture;

  GST_DEBUG_CATEGORY_INIT(gst_placebo_filter_debug, "placebofilter", 0,
      "libplacebo color filter");
}

static void gst_placebo_filter_init(GstPlaceboFilter *self) {
  self->lut_path = NULL;
  self->pl_log = NULL;
  self->pl_gl = NULL;
  self->pl_renderer = NULL;
  self->src_tex = NULL;
  self->dst_tex = NULL;
}
```

**Step 4: Create plugin registration**

```c
// plugins/gst-libplacebo/plugin.c
#include <gst/gst.h>
#include "gstplacebofilter.h"

static gboolean plugin_init(GstPlugin *plugin) {
  return gst_element_register(plugin, "placebofilter", GST_RANK_NONE,
      GST_TYPE_PLACEBO_FILTER);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    placebo,
    "libplacebo color processing filter (OpenGL backend)",
    plugin_init,
    "0.1.0",
    "LGPL",
    "ave",
    "https://github.com/agentic-vid-editor"
)
```

**Step 5: Build and test inside Docker container**

Run:
```bash
docker compose -f docker/docker-compose.yml run ave bash -c "\
  cd /app/plugins/gst-libplacebo && \
  meson setup build && \
  ninja -C build && \
  GST_PLUGIN_PATH=/app/plugins/gst-libplacebo/build \
  gst-inspect-1.0 placebofilter"
```
Expected: Plugin info printed with "libplacebo Filter" description

**Step 6: Commit**

```bash
git add plugins/gst-libplacebo/
git commit -m "feat: libplacebo GStreamer element prototype (OpenGL backend, passthrough)"
```

---

## Task 11: OCIO GStreamer Element Prototype (C)

> Similar to Task 10, this is a C GStreamer plugin. Minimal prototype for Phase 1; production implementation in Phase 3.

**Files:**
- Create: `plugins/gst-ocio/meson.build`
- Create: `plugins/gst-ocio/gstociofilter.h`
- Create: `plugins/gst-ocio/gstociofilter.c`
- Create: `plugins/gst-ocio/plugin.c`

**Step 1: Create meson.build**

```meson
# plugins/gst-ocio/meson.build
project('gst-ocio', 'c', 'cpp',
  version : '0.1.0',
  default_options : ['warning_level=2'])

gst_dep = dependency('gstreamer-1.0', version : '>= 1.28.1')
gst_gl_dep = dependency('gstreamer-gl-1.0', version : '>= 1.28.1')
gst_video_dep = dependency('gstreamer-video-1.0', version : '>= 1.28.1')
ocio_dep = dependency('OpenColorIO', version : '>= 2.3')

gstocio = shared_library('gstocio',
  'gstociofilter.c',
  'plugin.c',
  dependencies : [gst_dep, gst_gl_dep, gst_video_dep, ocio_dep],
  install : true,
  install_dir : join_paths(get_option('libdir'), 'gstreamer-1.0'),
)
```

**Step 2: Create header**

```c
// plugins/gst-ocio/gstociofilter.h
#ifndef __GST_OCIO_FILTER_H__
#define __GST_OCIO_FILTER_H__

#include <gst/gl/gstglfilter.h>

G_BEGIN_DECLS

#define GST_TYPE_OCIO_FILTER (gst_ocio_filter_get_type())
G_DECLARE_FINAL_TYPE(GstOCIOFilter, gst_ocio_filter, GST, OCIO_FILTER, GstGLFilter)

struct _GstOCIOFilter {
  GstGLFilter parent;

  /* Properties */
  gchar *config_path;
  gchar *src_colorspace;
  gchar *dst_colorspace;

  /* OCIO state (opaque, managed in .cpp helper) */
  gpointer ocio_processor;

  /* GL state */
  guint gl_program;
  guint lut3d_tex;
  gint lut3d_size;
};

G_END_DECLS

#endif /* __GST_OCIO_FILTER_H__ */
```

**Step 3: Create minimal implementation outline**

The full OCIO element requires C++ for the OCIO API. For the prototype, create a stub that loads an OCIO config and validates the transform exists.

```c
// plugins/gst-ocio/gstociofilter.c
/*
 * GstOCIOFilter — OpenColorIO transform filter for GStreamer.
 *
 * Phase 1 prototype: validates OCIO config + transform exists, passthrough.
 * Phase 3: full GPU shader generation via GpuShaderDesc, LUT texture upload.
 */

#include "gstociofilter.h"

GST_DEBUG_CATEGORY_STATIC(gst_ocio_filter_debug);
#define GST_CAT_DEFAULT gst_ocio_filter_debug

enum {
  PROP_0,
  PROP_CONFIG_PATH,
  PROP_SRC_COLORSPACE,
  PROP_DST_COLORSPACE,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:GLMemory),format=RGBA"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:GLMemory),format=RGBA"));

G_DEFINE_TYPE(GstOCIOFilter, gst_ocio_filter, GST_TYPE_GL_FILTER);

static void gst_ocio_filter_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec) {
  GstOCIOFilter *self = GST_OCIO_FILTER(object);
  switch (prop_id) {
    case PROP_CONFIG_PATH:
      g_free(self->config_path);
      self->config_path = g_value_dup_string(value);
      break;
    case PROP_SRC_COLORSPACE:
      g_free(self->src_colorspace);
      self->src_colorspace = g_value_dup_string(value);
      break;
    case PROP_DST_COLORSPACE:
      g_free(self->dst_colorspace);
      self->dst_colorspace = g_value_dup_string(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void gst_ocio_filter_get_property(GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec) {
  GstOCIOFilter *self = GST_OCIO_FILTER(object);
  switch (prop_id) {
    case PROP_CONFIG_PATH:
      g_value_set_string(value, self->config_path);
      break;
    case PROP_SRC_COLORSPACE:
      g_value_set_string(value, self->src_colorspace);
      break;
    case PROP_DST_COLORSPACE:
      g_value_set_string(value, self->dst_colorspace);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static gboolean gst_ocio_filter_gl_start(GstGLFilter *filter) {
  GstOCIOFilter *self = GST_OCIO_FILTER(filter);
  GST_INFO_OBJECT(self, "OCIO filter started (config=%s, %s -> %s)",
      self->config_path ? self->config_path : "(default)",
      self->src_colorspace ? self->src_colorspace : "(none)",
      self->dst_colorspace ? self->dst_colorspace : "(none)");
  /* Phase 3: Load OCIO config, create processor, generate GPU shader,
   * compile GL program, upload LUT textures */
  return TRUE;
}

static void gst_ocio_filter_gl_stop(GstGLFilter *filter) {
  /* Phase 3: Clean up GL program, LUT textures, OCIO processor */
}

static gboolean gst_ocio_filter_filter_texture(GstGLFilter *filter,
    GstGLMemory *input, GstGLMemory *output) {
  /* Phase 1: passthrough. Phase 3: apply OCIO shader. */
  const GstGLFuncs *gl = filter->context->gl_vtable;
  guint in_tex = gst_gl_memory_get_texture_id(input);
  guint out_tex = gst_gl_memory_get_texture_id(output);

  /* Simple copy via FBO for prototype */
  guint fbo;
  gl->GenFramebuffers(1, &fbo);
  gl->BindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
  gl->FramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
      GL_TEXTURE_2D, in_tex, 0);

  GstVideoInfo *info = &filter->out_info;
  int w = GST_VIDEO_INFO_WIDTH(info);
  int h = GST_VIDEO_INFO_HEIGHT(info);

  gl->BindTexture(GL_TEXTURE_2D, out_tex);
  gl->CopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, w, h);

  gl->BindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  gl->DeleteFramebuffers(1, &fbo);

  return TRUE;
}

static void gst_ocio_filter_finalize(GObject *object) {
  GstOCIOFilter *self = GST_OCIO_FILTER(object);
  g_free(self->config_path);
  g_free(self->src_colorspace);
  g_free(self->dst_colorspace);
  G_OBJECT_CLASS(gst_ocio_filter_parent_class)->finalize(object);
}

static void gst_ocio_filter_class_init(GstOCIOFilterClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GstGLFilterClass *filter_class = GST_GL_FILTER_CLASS(klass);

  gobject_class->set_property = gst_ocio_filter_set_property;
  gobject_class->get_property = gst_ocio_filter_get_property;
  gobject_class->finalize = gst_ocio_filter_finalize;

  g_object_class_install_property(gobject_class, PROP_CONFIG_PATH,
      g_param_spec_string("config-path", "OCIO Config",
          "Path to OpenColorIO config", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_SRC_COLORSPACE,
      g_param_spec_string("src-colorspace", "Source Color Space",
          "Source OCIO color space name", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_DST_COLORSPACE,
      g_param_spec_string("dst-colorspace", "Destination Color Space",
          "Destination OCIO color space name", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata(element_class,
      "OCIO Color Transform", "Filter/Video",
      "OpenColorIO color space transform via OpenGL",
      "Agentic Video Editor <ave@example.com>");

  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);

  filter_class->gl_start = gst_ocio_filter_gl_start;
  filter_class->gl_stop = gst_ocio_filter_gl_stop;
  filter_class->filter_texture = gst_ocio_filter_filter_texture;

  GST_DEBUG_CATEGORY_INIT(gst_ocio_filter_debug, "ociofilter", 0,
      "OpenColorIO color transform filter");
}

static void gst_ocio_filter_init(GstOCIOFilter *self) {
  self->config_path = NULL;
  self->src_colorspace = NULL;
  self->dst_colorspace = NULL;
  self->ocio_processor = NULL;
  self->gl_program = 0;
  self->lut3d_tex = 0;
  self->lut3d_size = 0;
}
```

**Step 4: Create plugin registration**

```c
// plugins/gst-ocio/plugin.c
#include <gst/gst.h>
#include "gstociofilter.h"

static gboolean plugin_init(GstPlugin *plugin) {
  return gst_element_register(plugin, "ociofilter", GST_RANK_NONE,
      GST_TYPE_OCIO_FILTER);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    ocio,
    "OpenColorIO color transform filter",
    plugin_init,
    "0.1.0",
    "LGPL",
    "ave",
    "https://github.com/agentic-vid-editor"
)
```

**Step 5: Build and test inside Docker container**

Run:
```bash
docker compose -f docker/docker-compose.yml run ave bash -c "\
  cd /app/plugins/gst-ocio && \
  meson setup build && \
  ninja -C build && \
  GST_PLUGIN_PATH=/app/plugins/gst-ocio/build \
  gst-inspect-1.0 ociofilter"
```
Expected: Plugin info printed with "OCIO Color Transform" description

**Step 6: Commit**

```bash
git add plugins/gst-ocio/
git commit -m "feat: OCIO GStreamer element prototype (passthrough, OpenGL)"
```

---

## Phase 1 Completion Checklist

After all tasks are complete, verify:

- [ ] `pip install -e ".[dev]"` works
- [ ] `pytest -v -m "not gpu and not slow"` — all tests pass
- [ ] `pytest -v` inside Docker container — all tests pass including GES tests
- [ ] `docker compose -f docker/docker-compose.yml run test` — passes
- [ ] Ingest: H.264 source → DNxHR HQX working + H.264 proxy + asset registry entry
- [ ] Timeline: create, add clips, add effects, set keyframes, save/load XGES
- [ ] Render: GES timeline → 480p H.264 proxy
- [ ] Integration: full pipeline (ingest → timeline → render) passes
- [ ] libplacebo plugin: `gst-inspect-1.0 placebofilter` works inside Docker
- [ ] OCIO plugin: `gst-inspect-1.0 ociofilter` works inside Docker

---

## What Comes Next

Phase 1 establishes the foundation. Subsequent phases (separate plans):

- **Phase 2: Core Editing** — trim, split, concatenate, transitions, speed change, audio, transcription
- **Phase 3: Color & Compositing** — production libplacebo/OCIO, non-destructive IDT pipeline, GLSL grading, compositing, motion graphics
- **Phase 4: Preview & Polish** — segment cache, LL-HLS server, WebSocket scrubbing, timeline UI, OTIO export
- **Phase 5: Agent Layer** — tool registry, Tool RAG, visual analysis, verification loop, workflow prompts, conversational editing
