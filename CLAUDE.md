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
