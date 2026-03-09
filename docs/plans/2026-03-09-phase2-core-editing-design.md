# Phase 2: Core Editing — Design

## Overview
Timeline editing operations, audio tools, and transcription. All tools follow pure-function pattern with separated logic/execution layers for testability.

## Architecture: Three-Layer Testing

1. **Pure logic layer** (`src/ave/tools/`) — compute parameters, validate inputs. No GES. Testable everywhere.
2. **GES execution layer** (`src/ave/project/`) — apply computed params to GES timeline. Needs Docker.
3. **E2E tests** — full ingest → edit → render → probe output verification.

## Modules

### Edit Tools (`src/ave/tools/edit.py`)
- `trim(clip_id, in_ns, out_ns)` → trim clip to subrange
- `split(clip_id, position_ns)` → split clip at position, returns two clip IDs
- `concatenate(clip_ids, layer)` → place clips sequentially on timeline
- Pure logic: parameter validation, duration calculation, overlap detection

### Transitions (`src/ave/tools/transitions.py`)
- `add_transition(clip_a_id, clip_b_id, type, duration_ns)` → crossfade/wipe between clips
- Supported types: crossfade, fade-to-black, SMPTE wipes (via GES)
- Pure logic: transition type validation, duration bounds checking, overlap calculation

### Speed Change (`src/ave/tools/speed.py`)
- `set_speed(clip_id, rate)` → constant speed change (0.1x - 100x)
- Pure logic: rate validation, output duration calculation, audio pitch preservation flag

### Audio Tools (`src/ave/tools/audio.py`)
- `set_volume(clip_id, level_db)` → set clip volume in dB
- `add_fade(clip_id, fade_in_ns, fade_out_ns)` → audio fades
- `normalize(clip_id, target_lufs)` → loudness normalization
- Pure logic: dB/linear conversion, LUFS calculation, fade curve generation

### Transcription (`src/ave/tools/transcribe.py`)
- `transcribe(audio_path)` → word-level timestamped transcript as JSON
- Engine: `whispercpp` Python bindings (GGML models, CUDA acceleration, CPU fallback)
- Model: whisper large-v3-turbo GGML quantized
- Output format: `{"segments": [{"start": float, "end": float, "text": str, "words": [...]}]}`
- Model management: auto-download to `~/.cache/ave/models/`

## Transcription Engine: whispercpp (GGML)
- C++ core with Python bindings (`pywhispercpp` or `whispercpp`)
- GGML quantized models (Q5_0/Q8_0) — fast, small, CUDA-accelerated
- CPU fallback automatic when no CUDA
- No Python ML framework dependency (no torch/transformers)
- ~6x realtime on CPU, ~50x+ realtime on CUDA

## Docker Integration
- All Phase 2 deps added to existing `docker/Dockerfile`
- whispercpp built from source in Docker with CUDA support
- Model files mounted as volume (not baked into image)
- `docker-compose.yml` extended with model cache volume

## Testing Strategy
- Unit tests: pure logic, no external deps, run locally
- Integration tests: GES operations, require Docker (`@requires_ges`)
- E2E tests: full pipeline round-trips, require Docker + FFmpeg
- Transcription tests: require whispercpp (`@requires_whisper` marker)
- Target: 80-100+ tests total
