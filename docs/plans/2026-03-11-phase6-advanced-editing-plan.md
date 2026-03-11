# Phase 6: Advanced Editing Features — Implementation Plan

> Date: 2026-03-11
> Status: Ready for implementation
> Dependencies: Phase 4 (agent tools) complete. Phase 5 (GPU color) can run in parallel.

---

## Current State Inventory

### What's DONE (do NOT rebuild)

| Component | File | Status |
|-----------|------|--------|
| Compositing pure logic | `src/ave/tools/compositing.py` | 100% — BlendMode enum (7 modes), LayerParams, BlendFuncParams, compute_layer_params, compute_blend_params |
| Motion graphics pure logic | `src/ave/tools/motion_graphics.py` | 100% — TextOverlayParams, LowerThirdParams, compute_text_overlay, compute_position_coords, compute_lower_third, compute_title_card |
| Scene detection models | `src/ave/tools/scene.py` | 100% — SceneBoundary, SceneBackend protocol, extract_keyframes, metadata constants |
| Scene detection backend | `src/ave/tools/scene_pyscenedetect.py` | 100% — PySceneDetectBackend with content/adaptive/threshold/hash detectors |
| Vision embeddings | `src/ave/tools/vision.py` | 100% — FrameEmbedding, SceneTag, cosine_similarity, similarity_search, tag_frames |
| Vision backend (SigLIP 2) | `src/ave/tools/vision_siglip2.py` | 100% — SigLIP2Backend with embed_image, embed_text, embed_batch |
| Transcript editing | `src/ave/tools/transcript_edit.py` | 100% — find_filler_words, compute_filler_removal_cuts, find_word_range, compute_text_cut, compute_text_keep, search_transcript, compute_cuts_to_edit_ops |
| Compositing unit tests | `tests/test_tools/test_compositing.py` | 100% |
| Motion graphics unit tests | `tests/test_tools/test_motion_graphics.py` | 100% |
| Scene detection unit tests | `tests/test_tools/test_scene.py` | 100% |
| Vision unit tests | `tests/test_tools/test_vision.py` | 100% |

### What's REMAINING

1. **Compositing GES ops** — No `compositing_ops.py` exists. Pure logic layer is done but there's no bridge to GES timeline operations (layer ordering, blend mode application via glvideomixer or compositor element).
2. **Motion graphics GES ops** — No `motion_graphics_ops.py` exists. Pure logic computes params but nothing applies `textoverlay` or generates GES effects.
3. **Scene-to-timeline workflow** — Scene detection produces SceneBoundary objects but there's no function to automatically create a rough cut from detected scenes (selecting best shots, ordering by visual content, etc).
4. **Shot classification integration** — SigLIP 2 backend can classify frames but there's no pipeline connecting: ingest → frame extraction → embedding → classification → metadata storage in registry.
5. **OTIO import/export** — 0% complete. No OpenTimelineIO integration exists.
6. **Multi-format render presets** — 0% complete. Rendering is hardcoded H.264/AAC in proxy.py and fMP4 in segment.py.

---

## Subtask Breakdown

### Subtask 6-1: Compositing GES Operations Layer

**Agent type:** Single agent, Python + GES
**Estimated scope:** ~200 LOC Python

**Context files to read:**
- `src/ave/tools/compositing.py` — BlendMode, LayerParams, BlendFuncParams, compute_layer_params, compute_blend_params
- `src/ave/tools/color_ops.py` — reference pattern for GES ops layer (how effects are added to timeline)
- `src/ave/project/timeline.py` — Timeline class API (add_clip, add_effect, get_clip, etc.)
- `src/ave/project/operations.py` — existing GES operations

**What to implement:**

Create `src/ave/tools/compositing_ops.py`:

1. **`apply_layer_compositing(timeline, layers: list[dict]) -> list[str]`**
   - Validates via `compute_layer_params()`
   - For each layer: moves clip to correct GES layer (track), sets alpha property
   - Returns list of effect_ids

2. **`apply_blend_mode(timeline, clip_id: str, blend_mode: BlendMode) -> str`**
   - Gets blend params via `compute_blend_params()`
   - If `requires_shader` (from Phase 5-7): adds `glshader` effect with GLSL blend shader
   - If GL blend functions suffice: sets blend properties on the compositor pad
   - Returns effect_id

3. **`set_clip_position(timeline, clip_id: str, x: int, y: int, width: int | None, height: int | None) -> None`**
   - Sets position and optional scale on a clip's compositor pad
   - Uses GES `videomixerpad` properties (xpos, ypos, width, height)

4. **`set_clip_alpha(timeline, clip_id: str, alpha: float) -> None`**
   - Sets alpha on the clip's compositor pad

**Architecture note:** Use `compositor` element (CPU), NOT `glvideomixer` (has crash bugs #728, #786). The compositor element supports `xpos`, `ypos`, `width`, `height`, `alpha` pad properties.

**Test plan (TDD):**
- Test file: `tests/test_tools/test_compositing_ops.py`
- Pure logic tests (mock Timeline):
  - `apply_layer_compositing` calls `compute_layer_params` and modifies timeline
  - `apply_blend_mode` with MULTIPLY returns correct effect_id
  - `set_clip_position` updates clip pad properties
  - Invalid blend mode raises CompositingError
- GES integration tests (`@requires_ges`):
  - Two clips on different layers with alpha blending → render → output exists
  - Position offset → render → verify frame analysis shows offset content

**Acceptance criteria:**
- All 7 blend modes supported (5 via GL blend, 2 via shader)
- Layer ordering maps correctly to GES tracks
- Alpha and position properties work via compositor pad
- Uses CPU `compositor`, not `glvideomixer`

---

### Subtask 6-2: Motion Graphics GES Operations Layer

**Agent type:** Single agent, Python + GES
**Estimated scope:** ~180 LOC Python

**Context files to read:**
- `src/ave/tools/motion_graphics.py` — TextOverlayParams, LowerThirdParams, all compute_* functions
- `src/ave/tools/color_ops.py` — GES effect pattern reference
- `src/ave/project/timeline.py` — Timeline API

**What to implement:**

Create `src/ave/tools/motion_graphics_ops.py`:

1. **`apply_text_overlay(timeline, clip_id: str, params: TextOverlayParams) -> str`**
   - Adds `GES.Effect.new("textoverlay")` to the clip
   - Sets properties: `text`, `font-desc` (Pango format: "Arial 36"), `halignment`, `valignment`, `color` (GStreamer ARGB uint32), `shaded-background`
   - Maps TextPosition → pango halignment/valignment properties
   - Returns effect_id

2. **`apply_lower_third(timeline, clip_id: str, params: LowerThirdParams) -> list[str]`**
   - Applies two textoverlay effects (name + title) at different vertical offsets
   - Optionally applies a semi-transparent rectangle via `gdkpixbufoverlay` or custom GL shader for the background bar
   - Returns list of effect_ids

3. **`apply_title_card(timeline, start_ns: int, duration_ns: int, params: TextOverlayParams) -> str`**
   - Creates a `GES.TitleClip` (GES built-in title source) with centered text
   - Adds to the timeline at the specified position
   - Returns clip_id

4. **Position mapping helper:**
   ```python
   _HALIGN_MAP = {
       TextPosition.TOP_LEFT: "left", TextPosition.TOP_CENTER: "center", ...
   }
   _VALIGN_MAP = {
       TextPosition.TOP_LEFT: "top", TextPosition.BOTTOM_LEFT: "bottom", ...
   }
   ```

5. **Color conversion helper:**
   ```python
   def _rgba_to_argb_uint32(rgba: tuple[int, int, int, int]) -> int:
       """Convert (R, G, B, A) tuple to GStreamer ARGB uint32."""
   ```

**Test plan (TDD):**
- Test file: `tests/test_tools/test_motion_graphics_ops.py`
- Pure logic tests (mock Timeline):
  - `apply_text_overlay` creates textoverlay effect with correct properties
  - `_rgba_to_argb_uint32((255, 0, 0, 128))` returns correct uint32
  - Position mapping covers all 7 TextPosition values
- GES integration tests (`@requires_ges`):
  - Add text overlay to clip → render → output exists and has text
  - Lower third template → render → verify output
  - Title card → verify clip duration matches params

**Acceptance criteria:**
- textoverlay element works with all TextPosition values
- Color conversion RGBA → ARGB uint32 is correct
- Lower third creates two text layers + background
- Title card uses GES.TitleClip

---

### Subtask 6-3: Scene-to-Timeline Rough Cut Workflow

**Agent type:** Single agent, Python
**Estimated scope:** ~200 LOC Python

**Context files to read:**
- `src/ave/tools/scene.py` — SceneBoundary model, metadata constants
- `src/ave/tools/scene_pyscenedetect.py` — PySceneDetectBackend.detect_scenes()
- `src/ave/tools/vision.py` — SceneTag, similarity_search, tag_frames
- `src/ave/tools/edit.py` — TrimParams, compute_trim
- `src/ave/project/timeline.py` — Timeline.add_clip()

**What to implement:**

Create `src/ave/tools/rough_cut.py` (pure logic) and `src/ave/tools/rough_cut_ops.py` (GES execution):

**Pure logic (`rough_cut.py`):**

1. **`RoughCutParams` dataclass:**
   ```python
   @dataclass(frozen=True)
   class RoughCutParams:
       scenes: list[SceneBoundary]
       selected_indices: list[int]  # which scenes to include
       order: str  # "chronological" | "custom"
       gap_ns: int  # gap between clips (default 0)
   ```

2. **`select_scenes_by_tags(scenes: list[SceneBoundary], tags: list[SceneTag], include_labels: set[str], exclude_labels: set[str]) -> list[int]`**
   - Filter scenes by their classification labels
   - Example: include={"close-up", "medium"}, exclude={"wide"} → select only close-up and medium shots
   - Returns list of scene indices to include

3. **`select_scenes_by_duration(scenes: list[SceneBoundary], min_duration_ns: int, max_duration_ns: int) -> list[int]`**
   - Filter by duration range (remove very short/long scenes)

4. **`compute_rough_cut_timeline(params: RoughCutParams) -> list[dict]`**
   - Produces a list of clip placement dicts: `[{source_path, start_ns, end_ns, timeline_position_ns}, ...]`
   - Respects ordering and gaps

**GES execution (`rough_cut_ops.py`):**

5. **`apply_rough_cut(timeline, source_path: Path, params: RoughCutParams) -> list[str]`**
   - Creates clips on the timeline from the rough cut plan
   - Returns list of clip_ids

**Test plan (TDD):**
- Test file: `tests/test_tools/test_rough_cut.py`
- Test: 10 scenes, select by tags → correct indices
- Test: Filter by duration → removes short scenes
- Test: Chronological order → clips placed in order
- Test: Custom order with reordering → correct placement
- Test: Gap between clips → positions offset correctly
- All pure logic — no GES required

**Acceptance criteria:**
- Scene selection by label classification works
- Scene selection by duration range works
- Rough cut produces correctly ordered clip placements
- Zero-gap and custom-gap both work

---

### Subtask 6-4: Shot Classification Integration Pipeline

**Agent type:** Single agent, Python
**Estimated scope:** ~180 LOC Python

**Context files to read:**
- `src/ave/tools/vision.py` — VisionBackend protocol, FrameEmbedding, tag_frames, save_analysis
- `src/ave/tools/vision_siglip2.py` — SigLIP2Backend
- `src/ave/tools/scene.py` — SceneBoundary, extract_keyframes
- `src/ave/ingest/registry.py` — AssetEntry.visual_analysis_path
- `src/ave/ingest/probe.py` — probe_media

**What to implement:**

Create `src/ave/tools/classify.py`:

1. **Standard shot type labels:**
   ```python
   SHOT_LABELS = [
       "extreme wide shot", "wide shot", "medium wide shot",
       "medium shot", "medium close-up", "close-up",
       "extreme close-up", "over-the-shoulder", "two-shot",
       "insert shot", "aerial shot", "point-of-view",
   ]
   ```

2. **`classify_video(video_path: Path, scenes: list[SceneBoundary], backend: VisionBackend, output_dir: Path, labels: list[str] | None = None) -> VisualAnalysis`**
   - For each scene: extract middle keyframe via `extract_keyframes()`
   - Load keyframe images as numpy arrays
   - Batch embed frames via `backend.embed_batch()`
   - Embed label texts via `backend.embed_text()`
   - Run `tag_frames()` for zero-shot classification
   - Return `VisualAnalysis` with all embeddings and tags
   - Save to JSON via `save_analysis()`

3. **`classify_and_register(asset_id: str, registry: AssetRegistry, backend: VisionBackend) -> VisualAnalysis`**
   - Loads asset entry from registry
   - Runs scene detection (if not already done)
   - Runs classification
   - Updates `entry.visual_analysis_path` in registry
   - Returns analysis

4. **Frame loading helper:**
   ```python
   def _load_keyframe_as_array(path: Path) -> np.ndarray:
       """Load JPEG keyframe as (H, W, 3) uint8 numpy array."""
   ```

**Test plan (TDD):**
- Test file: `tests/test_tools/test_classify.py`
- Test: `SHOT_LABELS` has at least 10 entries
- Test: `classify_video` with mock backend → produces VisualAnalysis with correct scene count
- Test: Each scene gets a SceneTag with top_label
- Test: `classify_and_register` updates registry entry
- Mock VisionBackend for unit tests; real SigLIP2 tests with `@pytest.mark.slow` + `@pytest.mark.gpu`

**Acceptance criteria:**
- Full pipeline: video → scenes → keyframes → embeddings → classification → registry
- Standard shot labels cover common cinematographic shot types
- Works with mock backend (fast tests) and real SigLIP2 (slow tests)
- Analysis saved to JSON and path stored in AssetEntry

---

### Subtask 6-5: OTIO Export (XGES → OTIO)

**Agent type:** Single agent, Python + OTIO
**Estimated scope:** ~250 LOC Python

**Context files to read:**
- `src/ave/project/timeline.py` — Timeline class (clips, effects, layers)
- `src/ave/project/operations.py` — GES operations
- OTIO documentation (use web search for `opentimelineio` Python API)

**What to implement:**

Create `src/ave/interchange/otio_export.py`:

1. **`export_to_otio(timeline, output_path: Path, format: str = "otio") -> Path`**
   - Converts AVE Timeline → OTIO Timeline
   - Maps: GES clips → OTIO clips, GES layers → OTIO tracks, clip positions → time ranges
   - Supported output formats: `.otio` (native), `.fcpxml` (Final Cut Pro), `.aaf` (Avid/Resolve)
   - Returns output path

2. **Mapping functions:**
   ```python
   def _ges_clip_to_otio_clip(clip_data: dict) -> otio.schema.Clip
   def _ges_layer_to_otio_track(layer_data: dict, clips: list) -> otio.schema.Track
   def _ns_to_rational_time(ns: int, fps: float) -> otio.opentime.RationalTime
   def _ns_range_to_time_range(start_ns: int, duration_ns: int, fps: float) -> otio.opentime.TimeRange
   ```

3. **Effect metadata preservation:**
   - Store AVE-specific effect metadata in OTIO metadata dict
   - Color effects → OTIO metadata (not directly translatable to other NLEs)
   - Transitions → OTIO transitions (cross dissolve, wipe)

4. **Media reference handling:**
   - Map original file paths to OTIO ExternalReference
   - Handle proxy vs original paths (export should reference originals)

**Dependencies:** `pip install opentimelineio` (add to pyproject.toml optional deps: `interchange = ["opentimelineio>=0.17"]`)

**Test plan (TDD):**
- Test file: `tests/test_interchange/test_otio_export.py`
- Test: Simple timeline (2 clips) → OTIO file → read back → verify clip count, durations
- Test: Multi-layer timeline → OTIO → verify track count
- Test: Timeline with transition → OTIO → verify transition exists
- Test: Export to `.fcpxml` format → valid XML output
- Test: Nanosecond ↔ RationalTime conversion accuracy
- Tests require `opentimelineio` optional dep

**Acceptance criteria:**
- Round-trip: AVE timeline → OTIO → read back preserves clip positions and durations
- Supports OTIO native, FCPXML, and AAF output formats
- Effect metadata preserved in OTIO metadata dict
- Media references point to original files

---

### Subtask 6-6: OTIO Import (OTIO → XGES)

**Agent type:** Single agent, Python + OTIO + GES
**Estimated scope:** ~250 LOC Python

**Context files to read:**
- `src/ave/project/timeline.py` — Timeline.add_clip() API
- `src/ave/interchange/otio_export.py` (from S6-5) — mapping patterns to reverse
- OTIO documentation

**What to implement:**

Create `src/ave/interchange/otio_import.py`:

1. **`import_from_otio(otio_path: Path, timeline) -> dict`**
   - Reads OTIO file (any supported format: .otio, .fcpxml, .aaf, .edl)
   - Maps: OTIO tracks → GES layers, OTIO clips → GES clips
   - Returns summary dict: `{clip_count, track_count, duration_ns, warnings: list[str]}`

2. **Mapping functions:**
   ```python
   def _otio_clip_to_ges_params(clip: otio.schema.Clip) -> dict
   def _otio_track_to_layer_index(track: otio.schema.Track, track_idx: int) -> int
   def _rational_time_to_ns(rt: otio.opentime.RationalTime) -> int
   ```

3. **Unsupported feature handling:**
   - Effects not in AVE's toolkit → add warning, skip effect
   - Nested compositions → flatten with warning
   - Generator clips (color bars, etc.) → skip with warning
   - Return all warnings in the summary dict

4. **Media relinking:**
   - Check if referenced media files exist at original paths
   - If not found: add warning, keep clip with broken reference
   - Provide `relink_media(import_result, media_dir: Path)` helper

**Test plan (TDD):**
- Test file: `tests/test_interchange/test_otio_import.py`
- Test: Simple .otio file → import → timeline has correct clip count
- Test: Multi-track .otio → correct layer mapping
- Test: .otio with unsupported effect → warning in summary, no crash
- Test: Missing media → warning, not error
- Test: Round-trip: export → import → compare clip positions
- Tests require `opentimelineio` optional dep

**Acceptance criteria:**
- Imports .otio, .fcpxml, .aaf, .edl formats
- Unsupported features produce warnings, not errors
- Media relinking helper for missing files
- Round-trip (export → import) preserves timeline structure

---

### Subtask 6-7: Multi-Format Render Presets

**Agent type:** Single agent, Python
**Estimated scope:** ~200 LOC Python

**Context files to read:**
- `src/ave/render/proxy.py` — current hardcoded H.264 render
- `src/ave/render/segment.py` — fMP4 segment render
- `src/ave/project/timeline.py` — Timeline class

**What to implement:**

Create `src/ave/render/presets.py` (pure logic) and update `src/ave/render/export.py` (GES execution):

**Pure logic (`presets.py`):**

1. **`RenderPreset` dataclass:**
   ```python
   @dataclass(frozen=True)
   class RenderPreset:
       name: str
       description: str
       video_codec: str          # "x264enc", "x265enc", "nvh264enc", "avenc_prores_ks", "avenc_dnxhd"
       audio_codec: str          # "avenc_aac", "flacenc", "wavenc"
       container: str            # "mp4mux", "matroskamux", "qtmux", "mxfmux"
       file_extension: str       # ".mp4", ".mkv", ".mov", ".mxf"
       video_props: dict         # codec-specific: bitrate, profile, quality, etc.
       audio_props: dict         # sample-rate, bitrate, etc.
       width: int | None         # None = source resolution
       height: int | None
       fps: float | None         # None = source fps
   ```

2. **Built-in presets:**
   ```python
   PRESETS = {
       "h264_web": RenderPreset(
           name="H.264 Web", description="Web delivery (YouTube/Vimeo)",
           video_codec="x264enc", audio_codec="avenc_aac", container="mp4mux",
           file_extension=".mp4",
           video_props={"bitrate": 8000, "pass": 1, "speed-preset": "medium"},
           audio_props={"bitrate": 192000},
           width=1920, height=1080, fps=None,
       ),
       "h265_archive": RenderPreset(...),  # H.265 high quality archive
       "prores_master": RenderPreset(...), # ProRes 422 HQ in MOV
       "dnxhr_master": RenderPreset(...),  # DNxHR HQX in MXF
       "instagram_reel": RenderPreset(...), # 1080x1920 vertical, H.264
       "tiktok": RenderPreset(...),         # 1080x1920 vertical, H.264, 60s max
       "youtube_4k": RenderPreset(...),     # 3840x2160, H.265
       "twitter_x": RenderPreset(...),      # 1280x720, H.264, <140s
   }
   ```

3. **`list_presets() -> list[dict]`** — return name + description for each preset
4. **`get_preset(name: str) -> RenderPreset`** — return full preset
5. **`validate_preset(preset: RenderPreset) -> list[str]`** — check for issues (codec availability, resolution constraints)

**GES execution (`export.py`):**

6. **`render_with_preset(xges_path: Path, output_path: Path, preset: RenderPreset) -> Path`**
   - Builds GES pipeline with the preset's codec, container, and properties
   - Handles resolution scaling if preset specifies width/height
   - Returns output path

**Test plan (TDD):**
- Test file: `tests/test_render/test_presets.py`
- Test: All 8+ presets have valid codecs and containers
- Test: `list_presets()` returns correct count
- Test: `get_preset("h264_web")` returns preset with correct properties
- Test: `get_preset("nonexistent")` raises error
- Test: `validate_preset()` catches invalid codec names
- Test: Instagram preset has 1080x1920 dimensions
- Pure logic tests — no GES required

- Test file: `tests/test_render/test_export.py` (`@requires_ges`)
- Test: Render with h264_web preset → output is valid H.264 MP4
- Test: Render with prores_master preset → output is valid ProRes MOV

**Acceptance criteria:**
- At least 8 presets covering: web, archive, master, social media
- Social media presets have correct dimensions and duration constraints
- Presets are extensible (user can create custom RenderPreset)
- GES render pipeline respects all preset properties

---

### Subtask 6-8: Agent Tool Registration for New Features

**Agent type:** Single agent, Python
**Estimated scope:** ~150 LOC Python

**Context files to read:**
- `src/ave/agent/tools/editing.py` — pattern for tool registration
- `src/ave/agent/tools/color.py` — pattern for tool registration
- `src/ave/agent/registry.py` — ToolRegistry, `@registry.tool()` decorator
- `src/ave/agent/session.py` — EditingSession._load_all_tools()

**What to implement:**

1. **Create `src/ave/agent/tools/compositing.py`:**
   - Register tools: `set_layer_order`, `apply_blend_mode`, `set_clip_position`, `set_clip_alpha`
   - All use lazy imports from `ave.tools.compositing_ops`
   - Prerequisites: `timeline_loaded`, `clip_exists`

2. **Create `src/ave/agent/tools/motion_graphics.py`:**
   - Register tools: `add_text_overlay`, `add_lower_third`, `add_title_card`
   - Lazy imports from `ave.tools.motion_graphics_ops`
   - Prerequisites: `timeline_loaded`

3. **Create `src/ave/agent/tools/scene.py`:**
   - Register tools: `detect_scenes`, `classify_shots`, `create_rough_cut`
   - Lazy imports from `ave.tools.scene_pyscenedetect`, `ave.tools.classify`, `ave.tools.rough_cut_ops`
   - `detect_scenes` provides `scenes_detected`
   - `classify_shots` requires `scenes_detected`, provides `shots_classified`
   - `create_rough_cut` requires `scenes_detected`

4. **Create `src/ave/agent/tools/interchange.py`:**
   - Register tools: `export_otio`, `import_otio`
   - Lazy imports from `ave.interchange.otio_export`, `ave.interchange.otio_import`
   - `export_otio` requires `timeline_loaded`

5. **Update `src/ave/agent/session.py`:**
   - Add imports and registration calls for all new tool modules in `_load_all_tools()`

**Test plan (TDD):**
- Test file: `tests/test_agent/test_new_domain_tools.py`
- Test: All new tools registered and searchable
- Test: `search_tools("blend")` finds compositing tools
- Test: `search_tools("text")` finds motion graphics tools
- Test: `search_tools("scene")` finds scene detection tools
- Test: `search_tools("export")` finds interchange tools
- Test: Prerequisites correctly enforce ordering (e.g., classify requires scenes_detected)
- Pure logic tests — no GES required

**Acceptance criteria:**
- All new features accessible via agent tool discovery
- Progressive disclosure works (search_tools → get_tool_schema → call_tool)
- Prerequisite chains correctly model workflow dependencies
- Session.to_dict() reflects updated tool count

---

## Batch Execution Plan

```
Batch 1 (parallel — no dependencies between them):
├── S6-1: Compositing GES ops          [Python + GES]
├── S6-2: Motion graphics GES ops      [Python + GES]
├── S6-3: Rough cut workflow            [Python, pure logic]
├── S6-4: Shot classification pipeline  [Python, vision deps]
└── S6-7: Render presets                [Python, pure logic + GES]

Batch 2 (parallel — depends on OTIO understanding, no cross-deps):
├── S6-5: OTIO export                  [Python + OTIO]
└── S6-6: OTIO import                  [Python + OTIO + GES]

Batch 3 (depends on all Batch 1 + 2):
└── S6-8: Agent tool registration      [Python, wires everything together]
```

**Batch 1** subtasks all work on different files in different domains. Run all 5 in parallel.

**Batch 2** OTIO import/export share conceptual patterns but work on different files. Can run in parallel. Placed in Batch 2 because agents should understand OTIO before both tasks.

**Batch 3** registers all new tools in the agent framework. Must wait for all implementation to be complete so it can wire up the correct imports and prerequisites.

---

## Integration Test Plan

After all subtasks complete:

1. **Compositing E2E** (`@requires_ges`):
   - Two clips, different layers, multiply blend → render → verify output

2. **Motion graphics E2E** (`@requires_ges`):
   - Add lower third to clip → render → verify text appears in output frame

3. **Scene-to-edit pipeline** (pure logic + `@requires_ffmpeg`):
   - Detect scenes → classify shots → select close-ups → compute rough cut → verify clip placements

4. **OTIO round-trip** (requires `opentimelineio`):
   - Create AVE timeline → export .otio → import .otio → compare structure

5. **Agent workflow E2E** (pure logic):
   - EditingSession → load project → detect_scenes → classify_shots → create_rough_cut → add_text_overlay → export_otio
   - Verify full prerequisite chain works end-to-end

6. **Render presets E2E** (`@requires_ges`):
   - Render same timeline with h264_web and prores_master presets → probe both outputs → verify codecs/containers

---

## Risk Factors

| Risk | Mitigation |
|------|------------|
| `glvideomixer` crash bugs (#728, #786) | Use CPU `compositor` element exclusively |
| `textoverlay` limited formatting | Sufficient for v1; evaluate `qml6glsink` in Phase 7 |
| OTIO adapter quality varies by format | Test all formats; document known limitations |
| SigLIP2 model size (~800MB) | Cache model in Docker image; lazy download on first use |
| Social media specs change frequently | Presets are data — easy to update without code changes |

---

## pyproject.toml Changes Required

```toml
[project.optional-dependencies]
interchange = ["opentimelineio>=0.17"]
vision = ["transformers>=4.49", "torch>=2.0", "numpy", "Pillow"]
scene = ["scenedetect[opencv]>=0.6"]
```

These are already partially present in the Dockerfile but should be formalized as optional extras.
