# Phase 8: Web UI + Integration — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based chat interface where non-technical users edit video through natural language, with a read-only timeline visualization, preview player, and asset browser.

**Architecture:** Single aiohttp server with chat WebSocket (Anthropic Python SDK streaming), REST API for timeline/assets, preview sub-app (existing), and vanilla HTML/CSS/JS frontend. Pure Python TimelineModel (no GES dependency) for timeline state. Session recovery via server-side session store.

**Tech Stack:** Python 3.12+, aiohttp >=3.9, anthropic >=0.43, vanilla JS (no framework), HTML5 Canvas for timeline.

**Spec:** `docs/superpowers/specs/2026-03-14-phase8-web-ui-design.md`

---

## File Structure

### New files to create

| File | Responsibility |
|------|---------------|
| `src/ave/web/__init__.py` | Package init |
| `src/ave/web/timeline_model.py` | Pure Python timeline state + XGES XML parser |
| `src/ave/web/app.py` | aiohttp app factory, route mounting, lifecycle |
| `src/ave/web/api.py` | REST endpoints: /api/timeline, /api/assets |
| `src/ave/web/chat.py` | Chat WebSocket handler + Anthropic SDK agentic loop |
| `src/ave/web/client/index.html` | Single-page layout (3-panel) |
| `src/ave/web/client/styles.css` | Dark theme |
| `src/ave/web/client/chat.js` | Chat WebSocket client, streaming display |
| `src/ave/web/client/timeline.js` | Canvas timeline renderer (read-only) |
| `src/ave/web/client/preview.js` | Preview player (refactored from existing) |
| `src/ave/web/client/assets.js` | Asset browser panel |
| `tests/test_web/__init__.py` | Test package init |
| `tests/test_web/test_timeline_model.py` | TimelineModel unit tests |
| `tests/test_web/test_api.py` | REST API unit tests |
| `tests/test_web/test_chat_protocol.py` | Chat message protocol tests |
| `tests/test_web/test_web_integration.py` | aiohttp integration tests |
| `tests/test_web/test_chat_integration.py` | Chat workflow with mock Anthropic client |

### Files to modify

| File | Change |
|------|--------|
| `pyproject.toml:11-28` | Add `web` optional dependency group with `anthropic>=0.43` |
| `pyproject.toml:37-46` | Add `web` pytest marker |
| `tests/conftest.py` | Add `requires_aiohttp` and `requires_anthropic` markers |

---

## Chunk 1: TimelineModel — Pure Python Timeline State

### Task 1: TimelineModel data classes and serialization

**Files:**
- Create: `src/ave/web/__init__.py`
- Create: `src/ave/web/timeline_model.py`
- Create: `tests/test_web/__init__.py`
- Create: `tests/test_web/test_timeline_model.py`

- [ ] **Step 1: Write failing test — empty timeline serialization**

Create `tests/test_web/__init__.py` (empty) and `tests/test_web/test_timeline_model.py`:

```python
"""Tests for TimelineModel — pure Python timeline state."""

from ave.web.timeline_model import TimelineModel


class TestTimelineModelEmpty:
    """Empty timeline produces correct JSON structure."""

    def test_empty_timeline_to_dict(self):
        model = TimelineModel()
        result = model.to_dict()
        assert result == {"layers": [], "duration_ns": 0, "fps": 24.0}

    def test_empty_timeline_duration(self):
        model = TimelineModel()
        assert model.duration_ns == 0

    def test_custom_fps(self):
        model = TimelineModel(fps=30.0)
        assert model.to_dict()["fps"] == 30.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_web/test_timeline_model.py::TestTimelineModelEmpty -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ave.web'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/web/__init__.py` (empty).

Create `src/ave/web/timeline_model.py`:

```python
"""Pure Python timeline state model — no GES dependency.

Provides a queryable timeline state that can be populated from:
1. XGES XML parsing (xml.etree.ElementTree)
2. Tool call results (clip add/remove/update)

Serializes to JSON for the /api/timeline REST endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClipState:
    """A clip on the timeline."""

    id: str
    name: str
    layer_index: int
    start_ns: int
    duration_ns: int
    in_point_ns: int = 0
    effects: list[str] = field(default_factory=list)
    media_type: str = "video"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "start_ns": self.start_ns,
            "duration_ns": self.duration_ns,
            "in_point_ns": self.in_point_ns,
            "effects": list(self.effects),
            "type": self.media_type,
        }


@dataclass
class LayerState:
    """A layer containing clips."""

    index: int
    clips: list[ClipState] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "clips": [c.to_dict() for c in self.clips],
        }


class TimelineModel:
    """Pure Python timeline state. No GES dependency."""

    def __init__(self, fps: float = 24.0) -> None:
        self._layers: list[LayerState] = []
        self._fps = fps
        self._xges_path: Path | None = None

    @property
    def duration_ns(self) -> int:
        """Total timeline duration (max clip end across all layers)."""
        max_end = 0
        for layer in self._layers:
            for clip in layer.clips:
                end = clip.start_ns + clip.duration_ns
                if end > max_end:
                    max_end = end
        return max_end

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict for /api/timeline."""
        return {
            "duration_ns": self.duration_ns,
            "fps": self._fps,
            "layers": [layer.to_dict() for layer in self._layers],
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_web/test_timeline_model.py::TestTimelineModelEmpty -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/ave/web/__init__.py src/ave/web/timeline_model.py \
  tests/test_web/__init__.py tests/test_web/test_timeline_model.py
git commit -m "feat(web): add TimelineModel with empty state serialization"
```

### Task 2: TimelineModel clip operations

**Files:**
- Modify: `src/ave/web/timeline_model.py`
- Modify: `tests/test_web/test_timeline_model.py`

- [ ] **Step 1: Write failing tests — clip add/remove/update**

Append to `tests/test_web/test_timeline_model.py`:

```python
from ave.web.timeline_model import ClipState


class TestTimelineModelClips:
    """Clip manipulation operations."""

    def _make_clip(self, id="clip_001", start=0, duration=5_000_000_000):
        return ClipState(
            id=id, name="test.mxf", layer_index=0,
            start_ns=start, duration_ns=duration,
        )

    def test_add_clip_creates_layer(self):
        model = TimelineModel()
        clip = self._make_clip()
        model.add_clip(clip)
        result = model.to_dict()
        assert len(result["layers"]) == 1
        assert len(result["layers"][0]["clips"]) == 1
        assert result["layers"][0]["clips"][0]["id"] == "clip_001"

    def test_add_clip_duration_updated(self):
        model = TimelineModel()
        model.add_clip(self._make_clip(duration=10_000_000_000))
        assert model.duration_ns == 10_000_000_000

    def test_add_multiple_clips_same_layer(self):
        model = TimelineModel()
        model.add_clip(self._make_clip(id="c1", start=0))
        model.add_clip(self._make_clip(id="c2", start=5_000_000_000))
        result = model.to_dict()
        assert len(result["layers"][0]["clips"]) == 2

    def test_add_clips_different_layers(self):
        model = TimelineModel()
        c1 = self._make_clip(id="c1")
        c1.layer_index = 0
        c2 = self._make_clip(id="c2")
        c2.layer_index = 1
        model.add_clip(c1)
        model.add_clip(c2)
        result = model.to_dict()
        assert len(result["layers"]) == 2

    def test_remove_clip(self):
        model = TimelineModel()
        model.add_clip(self._make_clip(id="c1"))
        model.remove_clip("c1")
        result = model.to_dict()
        assert len(result["layers"][0]["clips"]) == 0

    def test_remove_nonexistent_clip_raises(self):
        model = TimelineModel()
        import pytest
        with pytest.raises(KeyError):
            model.remove_clip("nonexistent")

    def test_update_clip(self):
        model = TimelineModel()
        model.add_clip(self._make_clip(id="c1", start=0, duration=5_000_000_000))
        model.update_clip("c1", start_ns=1_000_000_000, duration_ns=3_000_000_000)
        result = model.to_dict()
        clip = result["layers"][0]["clips"][0]
        assert clip["start_ns"] == 1_000_000_000
        assert clip["duration_ns"] == 3_000_000_000

    def test_update_nonexistent_clip_raises(self):
        model = TimelineModel()
        import pytest
        with pytest.raises(KeyError):
            model.update_clip("nonexistent", start_ns=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_web/test_timeline_model.py::TestTimelineModelClips -v`
Expected: FAIL — `AttributeError: 'TimelineModel' object has no attribute 'add_clip'`

- [ ] **Step 3: Implement clip operations**

Add to `TimelineModel` in `src/ave/web/timeline_model.py`:

```python
    def _get_or_create_layer(self, index: int) -> LayerState:
        """Get layer by index, creating it and any intermediate layers if needed."""
        while len(self._layers) <= index:
            self._layers.append(LayerState(index=len(self._layers)))
        return self._layers[index]

    def _find_clip(self, clip_id: str) -> tuple[LayerState, int]:
        """Find clip by ID. Returns (layer, clip_index). Raises KeyError if not found."""
        for layer in self._layers:
            for i, clip in enumerate(layer.clips):
                if clip.id == clip_id:
                    return layer, i
        raise KeyError(f"Clip not found: {clip_id}")

    def add_clip(self, clip: ClipState) -> None:
        """Add a clip to the specified layer."""
        layer = self._get_or_create_layer(clip.layer_index)
        layer.clips.append(clip)

    def remove_clip(self, clip_id: str) -> None:
        """Remove a clip by ID. Raises KeyError if not found."""
        layer, idx = self._find_clip(clip_id)
        layer.clips.pop(idx)

    def update_clip(self, clip_id: str, **kwargs) -> None:
        """Update clip attributes. Raises KeyError if not found."""
        layer, idx = self._find_clip(clip_id)
        clip = layer.clips[idx]
        for key, value in kwargs.items():
            if not hasattr(clip, key):
                raise AttributeError(f"ClipState has no attribute '{key}'")
            setattr(clip, key, value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_web/test_timeline_model.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/ave/web/timeline_model.py tests/test_web/test_timeline_model.py
git commit -m "feat(web): add TimelineModel clip add/remove/update operations"
```

### Task 3: XGES XML parser

**Files:**
- Modify: `src/ave/web/timeline_model.py`
- Modify: `tests/test_web/test_timeline_model.py`

- [ ] **Step 1: Write failing tests — XGES parsing**

Append to `tests/test_web/test_timeline_model.py`:

```python
class TestTimelineModelXGES:
    """Load timeline state from XGES XML."""

    MINIMAL_XGES = """\
<ges version='0.7'>
  <project properties='properties;' metadatas='metadatas;'>
    <ressources>
      <asset id='file:///media/clip1.mxf' extractable-type-name='GESUriClip' />
    </ressources>
    <timeline properties='properties, framerate=(fraction)24/1;'
              metadatas='metadatas;'>
      <track caps='video/x-raw(ANY)' track-type='4' track-id='0' />
      <layer priority='0'>
        <clip id='0' asset-id='file:///media/clip1.mxf'
              type-name='GESUriClip' layer-priority='0'
              start='0' duration='5000000000' inpoint='1000000000'
              rate='0' track-types='6'
              metadatas='metadatas, agent:clip-id=(string)clip_001;'>
        </clip>
      </layer>
    </timeline>
  </project>
</ges>
"""

    MULTI_LAYER_XGES = """\
<ges version='0.7'>
  <project properties='properties;' metadatas='metadatas;'>
    <ressources>
      <asset id='file:///media/clip1.mxf' extractable-type-name='GESUriClip' />
      <asset id='file:///media/clip2.mxf' extractable-type-name='GESUriClip' />
    </ressources>
    <timeline properties='properties, framerate=(fraction)30/1;'
              metadatas='metadatas;'>
      <track caps='video/x-raw(ANY)' track-type='4' track-id='0' />
      <layer priority='0'>
        <clip id='0' asset-id='file:///media/clip1.mxf'
              type-name='GESUriClip' layer-priority='0'
              start='0' duration='5000000000' inpoint='0'
              rate='0' track-types='6'
              metadatas='metadatas, agent:clip-id=(string)c1;'>
        </clip>
      </layer>
      <layer priority='1'>
        <clip id='1' asset-id='file:///media/clip2.mxf'
              type-name='GESUriClip' layer-priority='1'
              start='2000000000' duration='3000000000' inpoint='0'
              rate='0' track-types='6'
              metadatas='metadatas, agent:clip-id=(string)c2;'>
        </clip>
      </layer>
    </timeline>
  </project>
</ges>
"""

    def test_load_from_xges_string(self):
        model = TimelineModel()
        model.load_from_xges_string(self.MINIMAL_XGES)
        result = model.to_dict()
        assert len(result["layers"]) == 1
        assert len(result["layers"][0]["clips"]) == 1
        clip = result["layers"][0]["clips"][0]
        assert clip["id"] == "clip_001"
        assert clip["start_ns"] == 0
        assert clip["duration_ns"] == 5_000_000_000
        assert clip["in_point_ns"] == 1_000_000_000

    def test_load_fps_from_xges(self):
        model = TimelineModel()
        model.load_from_xges_string(self.MULTI_LAYER_XGES)
        assert model.to_dict()["fps"] == 30.0

    def test_load_multi_layer_xges(self):
        model = TimelineModel()
        model.load_from_xges_string(self.MULTI_LAYER_XGES)
        result = model.to_dict()
        assert len(result["layers"]) == 2
        assert result["layers"][0]["clips"][0]["id"] == "c1"
        assert result["layers"][1]["clips"][0]["id"] == "c2"

    def test_load_duration_calculated(self):
        model = TimelineModel()
        model.load_from_xges_string(self.MULTI_LAYER_XGES)
        # clip2: start=2s + duration=3s = 5s
        assert model.duration_ns == 5_000_000_000

    def test_load_from_xges_file(self, tmp_path):
        xges_file = tmp_path / "test.xges"
        xges_file.write_text(self.MINIMAL_XGES)
        model = TimelineModel()
        model.load_from_xges(xges_file)
        assert len(model.to_dict()["layers"]) == 1

    def test_reload_clears_previous_state(self):
        model = TimelineModel()
        model.load_from_xges_string(self.MINIMAL_XGES)
        model.load_from_xges_string(self.MINIMAL_XGES)
        # Should not duplicate — reload is idempotent
        assert len(model.to_dict()["layers"][0]["clips"]) == 1

    def test_clip_name_from_asset_id(self):
        model = TimelineModel()
        model.load_from_xges_string(self.MINIMAL_XGES)
        clip = model.to_dict()["layers"][0]["clips"][0]
        assert clip["name"] == "clip1.mxf"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_web/test_timeline_model.py::TestTimelineModelXGES -v`
Expected: FAIL — `AttributeError: 'TimelineModel' object has no attribute 'load_from_xges_string'`

- [ ] **Step 3: Implement XGES parser**

Add to `src/ave/web/timeline_model.py`:

```python
import re
import xml.etree.ElementTree as ET
```

Add methods to `TimelineModel`:

```python
    def load_from_xges(self, xges_path: Path) -> None:
        """Parse XGES XML file to populate timeline state."""
        self._xges_path = xges_path
        xml_str = xges_path.read_text()
        self.load_from_xges_string(xml_str)

    def load_from_xges_string(self, xml_str: str) -> None:
        """Parse XGES XML string to populate timeline state."""
        self._layers.clear()

        root = ET.fromstring(xml_str)
        timeline_el = root.find(".//timeline")
        if timeline_el is None:
            return

        # Extract FPS from timeline properties
        props = timeline_el.get("properties", "")
        fps_match = re.search(r"framerate=\(fraction\)(\d+)/(\d+)", props)
        if fps_match:
            self._fps = int(fps_match.group(1)) / int(fps_match.group(2))

        # Parse layers and clips
        for layer_el in timeline_el.findall("layer"):
            priority = int(layer_el.get("priority", "0"))

            for clip_el in layer_el.findall("clip"):
                clip_id = self._extract_clip_id(clip_el)
                asset_id = clip_el.get("asset-id", "")
                name = asset_id.rsplit("/", 1)[-1] if "/" in asset_id else asset_id

                clip = ClipState(
                    id=clip_id,
                    name=name,
                    layer_index=priority,
                    start_ns=int(clip_el.get("start", "0")),
                    duration_ns=int(clip_el.get("duration", "0")),
                    in_point_ns=int(clip_el.get("inpoint", "0")),
                )

                # Detect media type from track-types (4=video, 2=audio, 6=both)
                track_types = int(clip_el.get("track-types", "6"))
                if track_types == 4:
                    clip.media_type = "video"
                elif track_types == 2:
                    clip.media_type = "audio"
                else:
                    clip.media_type = "av"

                # Extract effects
                for effect_el in clip_el.findall(".//effect"):
                    asset = effect_el.get("asset-id", "")
                    clip.effects.append(asset)

                self.add_clip(clip)

    def reload_from_xges(self) -> None:
        """Re-parse the stored XGES path. No-op if no path set."""
        if self._xges_path and self._xges_path.exists():
            self.load_from_xges(self._xges_path)

    @staticmethod
    def _extract_clip_id(clip_el: ET.Element) -> str:
        """Extract agent:clip-id from clip metadata, or generate fallback."""
        metadatas = clip_el.get("metadatas", "")
        match = re.search(r"agent:clip-id=\(string\)([^;,]+)", metadatas)
        if match:
            return match.group(1)
        return f"clip_{clip_el.get('id', '0')}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_web/test_timeline_model.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/ave/web/timeline_model.py tests/test_web/test_timeline_model.py
git commit -m "feat(web): add XGES XML parser for TimelineModel"
```

---

## Chunk 2: REST API + App Factory

### Task 4: REST API endpoints

**Files:**
- Create: `src/ave/web/api.py`
- Create: `tests/test_web/test_api.py`

- [ ] **Step 1: Write failing tests — timeline and asset endpoints**

Create `tests/test_web/test_api.py`:

```python
"""Tests for REST API endpoint logic."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from ave.web.api import get_timeline_response, get_assets_response
from ave.web.timeline_model import TimelineModel, ClipState


class TestTimelineEndpoint:
    """Timeline REST endpoint logic."""

    def test_empty_timeline(self):
        model = TimelineModel()
        result = get_timeline_response(model)
        assert result["layers"] == []
        assert result["duration_ns"] == 0
        assert result["fps"] == 24.0

    def test_timeline_with_clips(self):
        model = TimelineModel()
        model.add_clip(ClipState(
            id="c1", name="test.mxf", layer_index=0,
            start_ns=0, duration_ns=5_000_000_000,
        ))
        result = get_timeline_response(model)
        assert len(result["layers"]) == 1
        assert result["layers"][0]["clips"][0]["id"] == "c1"
        assert result["duration_ns"] == 5_000_000_000


class TestAssetsEndpoint:
    """Asset listing endpoint logic."""

    def test_empty_registry(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        registry_path.write_text("[]")
        result = get_assets_response(registry_path)
        assert result == {"assets": []}

    def test_registry_with_assets(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        assets = [
            {
                "asset_id": "a1",
                "original_path": "/media/clip.mp4",
                "working_path": "/media/working/clip.mxf",
                "proxy_path": "/media/proxy/clip.mp4",
                "original_fps": 24.0,
                "conformed_fps": 24.0,
                "duration_seconds": 10.0,
                "width": 1920,
                "height": 1080,
                "codec": "h264",
                "camera_color_space": "bt709",
                "camera_transfer": "bt709",
                "idt_reference": None,
                "transcription_path": None,
                "visual_analysis_path": None,
            }
        ]
        registry_path.write_text(json.dumps(assets))
        result = get_assets_response(registry_path)
        assert len(result["assets"]) == 1
        a = result["assets"][0]
        assert a["id"] == "a1"
        assert a["name"] == "clip.mp4"
        assert a["duration_ns"] == 10_000_000_000
        assert "thumbnail_url" in a

    def test_missing_registry_file(self, tmp_path):
        result = get_assets_response(tmp_path / "nonexistent.json")
        assert result == {"assets": []}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_web/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ave.web.api'`

- [ ] **Step 3: Implement API logic**

Create `src/ave/web/api.py`:

```python
"""REST API endpoint logic for timeline and asset data.

Pure functions that produce JSON-serializable dicts.
aiohttp handler wiring is in app.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from ave.web.timeline_model import TimelineModel


def get_timeline_response(model: TimelineModel) -> dict:
    """Build the /api/timeline JSON response."""
    return model.to_dict()


def get_assets_response(registry_path: Path) -> dict:
    """Build the /api/assets JSON response from the asset registry file."""
    if not registry_path.exists():
        return {"assets": []}

    try:
        raw = json.loads(registry_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"assets": []}

    assets = []
    for entry in raw:
        asset_id = entry.get("asset_id", "")
        original = entry.get("original_path", "")
        name = Path(original).name if original else asset_id
        duration_s = entry.get("duration_seconds", 0.0)

        assets.append({
            "id": asset_id,
            "name": name,
            "duration_ns": int(duration_s * 1_000_000_000),
            "resolution": f"{entry.get('width', 0)}x{entry.get('height', 0)}",
            "fps": entry.get("conformed_fps", entry.get("original_fps", 24.0)),
            "thumbnail_url": f"/api/assets/{asset_id}/thumbnail",
        })

    return {"assets": assets}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_web/test_api.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/ave/web/api.py tests/test_web/test_api.py
git commit -m "feat(web): add REST API endpoint logic for timeline and assets"
```

### Task 5: Chat protocol data types

**Files:**
- Create: `tests/test_web/test_chat_protocol.py`
- Create: `src/ave/web/chat.py`

- [ ] **Step 1: Write failing tests — message parsing and event formatting**

Create `tests/test_web/test_chat_protocol.py`:

```python
"""Tests for chat WebSocket protocol message parsing and event formatting."""

from ave.web.chat import (
    parse_client_message,
    format_text_delta,
    format_tool_start,
    format_tool_done,
    format_timeline_updated,
    format_done,
    format_error,
    format_busy,
    format_connected,
)


class TestParseClientMessage:
    """Parse incoming WebSocket messages."""

    def test_parse_message(self):
        result = parse_client_message('{"type": "message", "text": "hello"}')
        assert result == {"type": "message", "text": "hello"}

    def test_parse_cancel(self):
        result = parse_client_message('{"type": "cancel"}')
        assert result == {"type": "cancel"}

    def test_parse_invalid_json(self):
        result = parse_client_message("not json")
        assert result["type"] == "error"

    def test_parse_missing_type(self):
        result = parse_client_message('{"text": "hello"}')
        assert result["type"] == "error"


class TestFormatEvents:
    """Format outgoing WebSocket events."""

    def test_text_delta(self):
        event = format_text_delta("hello ")
        assert event == {"type": "text_delta", "text": "hello "}

    def test_tool_start(self):
        event = format_tool_start("search_tools", "tc_01")
        assert event == {"type": "tool_start", "tool_name": "search_tools", "tool_id": "tc_01"}

    def test_tool_done(self):
        event = format_tool_done("tc_01")
        assert event == {"type": "tool_done", "tool_id": "tc_01"}

    def test_timeline_updated(self):
        event = format_timeline_updated()
        assert event == {"type": "timeline_updated"}

    def test_done(self):
        event = format_done(3)
        assert event == {"type": "done", "turn_id": 3}

    def test_error(self):
        event = format_error("something broke")
        assert event == {"type": "error", "message": "something broke"}

    def test_busy(self):
        event = format_busy()
        assert event == {"type": "busy", "message": "Agent is processing"}

    def test_connected(self):
        event = format_connected("tok_123")
        assert event == {"type": "connected", "session": "tok_123"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_web/test_chat_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ave.web.chat'`

- [ ] **Step 3: Implement protocol functions**

Create `src/ave/web/chat.py`:

```python
"""Chat WebSocket handler — message protocol and agentic loop.

Protocol functions (parse/format) are pure and testable without aiohttp.
The ChatSession class requires aiohttp and anthropic (optional deps).
"""

from __future__ import annotations

import json


# --- Protocol: pure functions, no external deps ---


def parse_client_message(raw: str) -> dict:
    """Parse a client WebSocket message. Returns parsed dict or error dict."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"type": "error", "message": "Invalid JSON"}
    if "type" not in data:
        return {"type": "error", "message": "Missing 'type' field"}
    return data


def format_text_delta(text: str) -> dict:
    return {"type": "text_delta", "text": text}


def format_tool_start(tool_name: str, tool_id: str) -> dict:
    return {"type": "tool_start", "tool_name": tool_name, "tool_id": tool_id}


def format_tool_done(tool_id: str) -> dict:
    return {"type": "tool_done", "tool_id": tool_id}


def format_timeline_updated() -> dict:
    return {"type": "timeline_updated"}


def format_done(turn_id: int) -> dict:
    return {"type": "done", "turn_id": turn_id}


def format_error(message: str) -> dict:
    return {"type": "error", "message": message}


def format_busy() -> dict:
    return {"type": "busy", "message": "Agent is processing"}


def format_connected(session_token: str) -> dict:
    return {"type": "connected", "session": session_token}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_web/test_chat_protocol.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/ave/web/chat.py tests/test_web/test_chat_protocol.py
git commit -m "feat(web): add chat WebSocket protocol functions"
```

### Task 6: App factory and aiohttp wiring

**Files:**
- Create: `src/ave/web/app.py`
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`
- Create: `tests/test_web/test_web_integration.py`

- [ ] **Step 1: Update pyproject.toml with web dependency group**

In `pyproject.toml`, after line 28 (after `preview` group), add:

```toml
web = [
    "aiohttp>=3.9",
    "anthropic>=0.43",
]
```

And add to markers (after line 45):

```toml
    "web: marks tests that require aiohttp for web UI",
```

- [ ] **Step 2: Add requires_aiohttp marker to conftest.py**

In `tests/conftest.py`, add an availability function and marker:

```python
def _aiohttp_available():
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        return False

requires_aiohttp = pytest.mark.skipif(
    not _aiohttp_available(), reason="aiohttp not installed"
)
```

- [ ] **Step 3: Write failing integration tests**

Create `tests/test_web/test_web_integration.py`:

```python
"""Integration tests for the web application — requires aiohttp."""

import pytest

try:
    from aiohttp import web
    from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


@pytest.fixture
def app(tmp_path):
    """Create a test web application."""
    from ave.web.app import create_app
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "assets").mkdir()
    return create_app(project_dir=project_dir)


class TestAppFactory:
    """App factory creates valid application."""

    @pytest.mark.asyncio
    async def test_create_app_returns_application(self, tmp_path):
        from ave.web.app import create_app
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "assets").mkdir()
        app = create_app(project_dir=project_dir)
        assert isinstance(app, web.Application)

    @pytest.mark.asyncio
    async def test_app_has_timeline_model(self, app):
        from ave.web.timeline_model import TimelineModel
        assert isinstance(app["timeline_model"], TimelineModel)


class TestRESTEndpoints:
    """REST endpoints return correct responses."""

    @pytest.mark.asyncio
    async def test_timeline_endpoint(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        resp = await client.get("/api/timeline")
        assert resp.status == 200
        data = await resp.json()
        assert "layers" in data
        assert "duration_ns" in data
        assert "fps" in data

    @pytest.mark.asyncio
    async def test_assets_endpoint(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        resp = await client.get("/api/assets")
        assert resp.status == 200
        data = await resp.json()
        assert "assets" in data

    @pytest.mark.asyncio
    async def test_static_index(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        resp = await client.get("/")
        assert resp.status == 200
        text = await resp.text()
        assert "AVE" in text


class TestWebSocketChat:
    """Chat WebSocket connects and responds."""

    @pytest.mark.asyncio
    async def test_chat_ws_connects(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/ws/chat")
        msg = await ws.receive_json()
        assert msg["type"] == "connected"
        assert "session" in msg
        await ws.close()

    @pytest.mark.asyncio
    async def test_chat_ws_reconnect_same_session(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        ws1 = await client.ws_connect("/ws/chat")
        msg1 = await ws1.receive_json()
        token = msg1["session"]
        await ws1.close()

        ws2 = await client.ws_connect(f"/ws/chat?session={token}")
        msg2 = await ws2.receive_json()
        assert msg2["session"] == token
        await ws2.close()
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_web/test_web_integration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ave.web.app'`

- [ ] **Step 5: Implement app factory**

Create `src/ave/web/app.py`:

```python
"""AVE Web Application — aiohttp app factory.

Creates a unified aiohttp application integrating:
- Chat WebSocket at /ws/chat
- REST API at /api/
- Preview sub-app at /preview/ (existing PreviewServer)
- Static client files at /
"""

from __future__ import annotations

import uuid
from pathlib import Path

from aiohttp import web

from ave.web.api import get_timeline_response, get_assets_response
from ave.web.chat import (
    parse_client_message,
    format_connected,
    format_error,
)
from ave.web.timeline_model import TimelineModel


CLIENT_DIR = Path(__file__).parent / "client"


def create_app(
    project_dir: Path,
    xges_path: Path | None = None,
) -> web.Application:
    """Create the AVE web application."""
    app = web.Application()

    # Shared state
    timeline_model = TimelineModel()
    if xges_path and xges_path.exists():
        timeline_model.load_from_xges(xges_path)

    app["project_dir"] = Path(project_dir)
    app["timeline_model"] = timeline_model
    app["sessions"] = {}  # session_token -> ChatSession

    # REST API routes
    app.router.add_get("/api/timeline", _handle_timeline)
    app.router.add_get("/api/assets", _handle_assets)

    # Chat WebSocket
    app.router.add_get("/ws/chat", _handle_ws_chat)

    # Static client files (must be last — catch-all)
    if CLIENT_DIR.exists():
        app.router.add_get("/", _handle_index)
        app.router.add_static("/static", CLIENT_DIR)

    return app


async def _handle_index(request: web.Request) -> web.Response:
    """Serve the main client page."""
    index_path = CLIENT_DIR / "index.html"
    if not index_path.exists():
        return web.Response(text="Client not found", status=404)
    return web.Response(
        text=index_path.read_text(),
        content_type="text/html",
    )


async def _handle_timeline(request: web.Request) -> web.Response:
    """GET /api/timeline — return current timeline state."""
    model = request.app["timeline_model"]
    return web.json_response(get_timeline_response(model))


async def _handle_assets(request: web.Request) -> web.Response:
    """GET /api/assets — return asset registry contents."""
    project_dir = request.app["project_dir"]
    registry_path = project_dir / "assets" / "registry.json"
    return web.json_response(get_assets_response(registry_path))


async def _handle_ws_chat(request: web.Request) -> web.WebSocketResponse:
    """WebSocket endpoint for chat — handles session management."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Session management
    token = request.query.get("session", str(uuid.uuid4()))
    sessions = request.app["sessions"]

    if token not in sessions:
        sessions[token] = {"messages": [], "turn_count": 0}

    await ws.send_json(format_connected(token))

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                parsed = parse_client_message(msg.data)
                if parsed["type"] == "error":
                    await ws.send_json(format_error(parsed["message"]))
                elif parsed["type"] == "message":
                    # Agent processing placeholder — implemented in Task 9
                    await ws.send_json(format_error("Agent not yet connected"))
                elif parsed["type"] == "cancel":
                    pass  # Cancel handling — implemented in Task 9
            elif msg.type == web.WSMsgType.ERROR:
                break
    finally:
        pass  # Session stays in sessions dict for reconnect

    return ws


def run_server(
    project_dir: Path,
    xges_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Run the AVE web server."""
    app = create_app(project_dir, xges_path)
    web.run_app(app, host=host, port=port)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_web/test_web_integration.py -v`
Expected: All passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml tests/conftest.py \
  src/ave/web/app.py tests/test_web/test_web_integration.py
git commit -m "feat(web): add app factory with REST API and chat WebSocket"
```

---

## Chunk 3: Frontend Client

### Task 7: HTML layout and CSS

**Files:**
- Create: `src/ave/web/client/index.html`
- Create: `src/ave/web/client/styles.css`

- [ ] **Step 1: Create the main HTML layout**

Create `src/ave/web/client/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AVE — Agentic Video Editor</title>
<link rel="stylesheet" href="/static/styles.css">
</head>
<body>
<div id="app">
  <header id="header">
    <h1>AVE</h1>
    <button id="btn-assets" title="Toggle asset browser">Assets</button>
  </header>

  <div id="main">
    <!-- Chat Panel -->
    <div id="chat-panel">
      <div id="chat-messages"></div>
      <div id="chat-input-area">
        <textarea id="chat-input" placeholder="Describe your edit..." rows="2"></textarea>
        <button id="btn-send" title="Send">Send</button>
      </div>
      <div id="chat-status"></div>
    </div>

    <!-- Right Panel: Preview + Timeline -->
    <div id="right-panel">
      <div id="preview-area">
        <video id="video" muted></video>
        <canvas id="scrub-canvas"></canvas>
      </div>
      <div id="preview-controls">
        <button id="btn-play" title="Play">&#9654;</button>
        <button id="btn-pause" title="Pause">&#9646;&#9646;</button>
        <input id="seek" type="range" min="0" max="1000" value="0">
        <span id="timecode">00:00:00.000</span>
      </div>
      <div id="timeline-area">
        <canvas id="timeline-canvas"></canvas>
      </div>
    </div>
  </div>

  <!-- Asset Browser (collapsible) -->
  <div id="asset-browser" class="hidden">
    <div id="asset-grid"></div>
  </div>
</div>

<script src="/static/preview.js"></script>
<script src="/static/timeline.js"></script>
<script src="/static/chat.js"></script>
<script src="/static/assets.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create CSS styles**

Create `src/ave/web/client/styles.css`:

```css
* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  --bg-primary: #0f0f1a;
  --bg-secondary: #1a1a2e;
  --bg-tertiary: #16213e;
  --border: #0f3460;
  --text: #e0e0e0;
  --text-muted: #888;
  --accent: #8be9fd;
  --accent-dim: #4a7c8f;
  --success: #50fa7b;
  --error: #ff5555;
  --clip-video: #4a6fa5;
  --clip-audio: #5a9;
  --clip-av: #6a5acd;
}

body {
  font-family: system-ui, -apple-system, sans-serif;
  background: var(--bg-primary);
  color: var(--text);
  height: 100vh;
  overflow: hidden;
}

#app { display: flex; flex-direction: column; height: 100vh; }

#header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.4rem 1rem; background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
}
#header h1 { font-size: 1rem; color: var(--accent); letter-spacing: 0.1em; }
#header button {
  background: var(--bg-tertiary); border: 1px solid var(--border);
  color: var(--text); padding: 0.3rem 0.8rem; cursor: pointer;
  border-radius: 3px; font-size: 0.8rem;
}
#header button:hover { background: var(--border); }

#main { display: flex; flex: 1; overflow: hidden; }

/* Chat Panel */
#chat-panel {
  width: 360px; min-width: 280px;
  display: flex; flex-direction: column;
  border-right: 1px solid var(--border);
  background: var(--bg-secondary);
}
#chat-messages {
  flex: 1; overflow-y: auto; padding: 0.8rem;
  font-size: 0.9rem; line-height: 1.5;
}
.msg { margin-bottom: 0.8rem; padding: 0.5rem 0.7rem; border-radius: 6px; }
.msg-user { background: var(--bg-tertiary); margin-left: 1.5rem; }
.msg-agent { background: var(--bg-primary); margin-right: 0.5rem; }
.msg-tool {
  font-size: 0.8rem; color: var(--text-muted);
  padding: 0.3rem 0.5rem; font-family: monospace;
}
.msg-error { color: var(--error); }

#chat-input-area { display: flex; padding: 0.5rem; gap: 0.4rem; border-top: 1px solid var(--border); }
#chat-input {
  flex: 1; resize: none; background: var(--bg-primary);
  border: 1px solid var(--border); color: var(--text);
  padding: 0.5rem; border-radius: 4px; font-family: inherit; font-size: 0.9rem;
}
#chat-input:focus { outline: 1px solid var(--accent-dim); }
#btn-send {
  background: var(--accent-dim); border: none; color: var(--text);
  padding: 0.5rem 1rem; cursor: pointer; border-radius: 4px; font-weight: 600;
}
#btn-send:hover { background: var(--accent); color: var(--bg-primary); }
#btn-send:disabled { opacity: 0.4; cursor: not-allowed; }

#chat-status { font-size: 0.75rem; padding: 0.3rem 0.8rem; color: var(--text-muted); }

/* Right Panel */
#right-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

#preview-area {
  position: relative; background: #000;
  aspect-ratio: 16/9; max-height: 45vh;
}
#preview-area video, #preview-area canvas {
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  object-fit: contain;
}
#scrub-canvas { display: none; }
#scrub-canvas.active { display: block; }

#preview-controls {
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0.4rem 0.8rem; background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
}
#preview-controls button {
  background: var(--bg-tertiary); border: 1px solid var(--border);
  color: var(--text); padding: 0.3rem 0.6rem; cursor: pointer; border-radius: 3px;
}
#preview-controls button:hover { background: var(--border); }
#seek { flex: 1; }
#timecode { font-family: monospace; font-size: 0.85rem; min-width: 10ch; text-align: right; }

/* Timeline */
#timeline-area {
  flex: 1; background: var(--bg-primary); overflow: hidden;
  border-top: 1px solid var(--border); min-height: 120px;
}
#timeline-canvas { width: 100%; height: 100%; display: block; }

/* Asset Browser */
#asset-browser {
  position: fixed; bottom: 0; left: 0; right: 0;
  height: 200px; background: var(--bg-secondary);
  border-top: 2px solid var(--border); overflow-y: auto;
  padding: 0.8rem; z-index: 10;
}
#asset-browser.hidden { display: none; }
#asset-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 0.8rem; }
.asset-card {
  background: var(--bg-tertiary); border: 1px solid var(--border);
  border-radius: 4px; cursor: pointer; overflow: hidden;
}
.asset-card:hover { border-color: var(--accent-dim); }
.asset-card img { width: 100%; aspect-ratio: 16/9; object-fit: cover; background: #000; }
.asset-card .info { padding: 0.3rem 0.5rem; font-size: 0.75rem; }
.asset-card .info .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.asset-card .info .meta { color: var(--text-muted); }
```

- [ ] **Step 3: Verify static files are served**

Run: `python3 -m pytest tests/test_web/test_web_integration.py::TestRESTEndpoints::test_static_index -v`
Expected: PASS (the test checks that `/` returns HTML containing "AVE")

- [ ] **Step 4: Commit**

```bash
git add src/ave/web/client/index.html src/ave/web/client/styles.css
git commit -m "feat(web): add HTML layout and CSS for 3-panel editor UI"
```

### Task 8: Frontend JavaScript — timeline, preview, chat, assets

**Files:**
- Create: `src/ave/web/client/preview.js`
- Create: `src/ave/web/client/timeline.js`
- Create: `src/ave/web/client/chat.js`
- Create: `src/ave/web/client/assets.js`

- [ ] **Step 1: Create preview.js**

Create `src/ave/web/client/preview.js`:

```javascript
/**
 * Preview player — video playback + WebSocket frame scrubbing.
 * Refactored from src/ave/preview/client/index.html.
 */
(function() {
  'use strict';

  const video = document.getElementById('video');
  const canvas = document.getElementById('scrub-canvas');
  const ctx = canvas.getContext('2d');
  const seekBar = document.getElementById('seek');
  const timecodeEl = document.getElementById('timecode');
  const btnPlay = document.getElementById('btn-play');
  const btnPause = document.getElementById('btn-pause');

  let previewWs = null;
  let durationNs = 15000000000;
  let scrubbing = false;

  window.AVE = window.AVE || {};

  function formatTimecode(ns) {
    const totalMs = Math.floor(ns / 1000000);
    const h = String(Math.floor(totalMs / 3600000)).padStart(2, '0');
    const m = String(Math.floor((totalMs % 3600000) / 60000)).padStart(2, '0');
    const s = String(Math.floor((totalMs % 60000) / 1000)).padStart(2, '0');
    const ms = String(totalMs % 1000).padStart(3, '0');
    return h + ':' + m + ':' + s + '.' + ms;
  }

  function connectPreviewWs() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    previewWs = new WebSocket(proto + '//' + location.host + '/preview/ws');
    previewWs.onopen = function() {};
    previewWs.onclose = function() { setTimeout(connectPreviewWs, 3000); };
    previewWs.onmessage = function(ev) {
      var msg = JSON.parse(ev.data);
      if (msg.type === 'frame') { showFrame(msg.data); }
    };
  }

  function showFrame(b64) {
    var img = new Image();
    img.onload = function() {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      canvas.classList.add('active');
    };
    img.src = 'data:image/jpeg;base64,' + b64;
  }

  function requestFrame(ns) {
    if (previewWs && previewWs.readyState === WebSocket.OPEN) {
      previewWs.send(JSON.stringify({ type: 'frame', timestamp_ns: ns }));
    }
  }

  seekBar.addEventListener('input', function() {
    scrubbing = true;
    canvas.classList.add('active');
    var ns = Math.round((seekBar.value / 1000) * durationNs);
    timecodeEl.textContent = formatTimecode(ns);
    requestFrame(ns);
    if (window.AVE.timeline) { window.AVE.timeline.setPlayhead(ns); }
  });

  seekBar.addEventListener('change', function() { scrubbing = false; });

  btnPlay.addEventListener('click', function() {
    video.play();
    canvas.classList.remove('active');
  });

  btnPause.addEventListener('click', function() { video.pause(); });

  video.addEventListener('timeupdate', function() {
    if (!scrubbing) {
      var ns = Math.round(video.currentTime * 1e9);
      timecodeEl.textContent = formatTimecode(ns);
      seekBar.value = Math.round((ns / durationNs) * 1000);
      if (window.AVE.timeline) { window.AVE.timeline.setPlayhead(ns); }
    }
  });

  window.AVE.preview = {
    setDuration: function(ns) { durationNs = ns; },
    seekTo: function(ns) {
      seekBar.value = Math.round((ns / durationNs) * 1000);
      timecodeEl.textContent = formatTimecode(ns);
      requestFrame(ns);
    }
  };

  connectPreviewWs();
})();
```

- [ ] **Step 2: Create timeline.js**

Create `src/ave/web/client/timeline.js`:

```javascript
/**
 * Timeline visualization — read-only canvas renderer.
 * Fetches state from /api/timeline, renders clips as colored rectangles.
 */
(function() {
  'use strict';

  var canvas = document.getElementById('timeline-canvas');
  var ctx = canvas.getContext('2d');
  var state = { layers: [], duration_ns: 0, fps: 24.0 };
  var playheadNs = 0;
  var scrollX = 0;
  var pixelsPerNs = 0.0000001; // 100px per second default
  var LAYER_HEIGHT = 40;
  var LAYER_GAP = 4;
  var HEADER_HEIGHT = 24;
  var debounceTimer = null;

  var COLORS = {
    video: '#4a6fa5', audio: '#5a9977', av: '#6a5acd',
    bg: '#0f0f1a', layerBg: '#151525', text: '#e0e0e0',
    playhead: '#ff5555', tick: '#333', selected: '#8be9fd'
  };

  var selectedClipId = null;

  window.AVE = window.AVE || {};

  function resize() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    render();
  }

  function render() {
    var w = canvas.width, h = canvas.height;
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    drawTimescale(w);
    drawLayers(w, h);
    drawPlayhead(w, h);
  }

  function drawTimescale(w) {
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, HEADER_HEIGHT);
    ctx.fillStyle = COLORS.text;
    ctx.font = '10px monospace';

    var startNs = scrollX / pixelsPerNs;
    var endNs = startNs + w / pixelsPerNs;
    var tickInterval = computeTickInterval(endNs - startNs, w);

    var first = Math.floor(startNs / tickInterval) * tickInterval;
    for (var t = first; t <= endNs; t += tickInterval) {
      var x = (t - startNs) * pixelsPerNs;
      ctx.strokeStyle = COLORS.tick;
      ctx.beginPath();
      ctx.moveTo(x, HEADER_HEIGHT - 6);
      ctx.lineTo(x, HEADER_HEIGHT);
      ctx.stroke();
      ctx.fillText(formatTime(t), x + 2, HEADER_HEIGHT - 8);
    }
  }

  function computeTickInterval(rangeNs, widthPx) {
    var targetTicks = widthPx / 80;
    var raw = rangeNs / targetTicks;
    var intervals = [
      100000000, 250000000, 500000000,
      1000000000, 2000000000, 5000000000,
      10000000000, 30000000000, 60000000000
    ];
    for (var i = 0; i < intervals.length; i++) {
      if (intervals[i] >= raw) return intervals[i];
    }
    return intervals[intervals.length - 1];
  }

  function formatTime(ns) {
    var s = Math.floor(ns / 1000000000);
    var m = Math.floor(s / 60);
    s = s % 60;
    return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
  }

  function drawLayers(w, h) {
    var startNs = scrollX / pixelsPerNs;

    for (var li = 0; li < state.layers.length; li++) {
      var layer = state.layers[li];
      var y = HEADER_HEIGHT + li * (LAYER_HEIGHT + LAYER_GAP);

      ctx.fillStyle = COLORS.layerBg;
      ctx.fillRect(0, y, w, LAYER_HEIGHT);

      for (var ci = 0; ci < layer.clips.length; ci++) {
        var clip = layer.clips[ci];
        var cx = (clip.start_ns - startNs) * pixelsPerNs;
        var cw = clip.duration_ns * pixelsPerNs;

        if (cx + cw < 0 || cx > w) continue; // viewport culling

        var color = COLORS[clip.type] || COLORS.av;
        if (clip.id === selectedClipId) color = COLORS.selected;

        ctx.fillStyle = color;
        ctx.fillRect(cx, y + 1, cw, LAYER_HEIGHT - 2);

        // Clip label
        if (cw > 30) {
          ctx.fillStyle = COLORS.text;
          ctx.font = '11px system-ui';
          ctx.save();
          ctx.beginPath();
          ctx.rect(cx + 2, y, cw - 4, LAYER_HEIGHT);
          ctx.clip();
          ctx.fillText(clip.name || clip.id, cx + 4, y + LAYER_HEIGHT / 2 + 4);
          ctx.restore();
        }

        // Effect dots
        if (clip.effects && clip.effects.length > 0) {
          for (var ei = 0; ei < Math.min(clip.effects.length, 5); ei++) {
            ctx.fillStyle = '#fff';
            ctx.beginPath();
            ctx.arc(cx + 6 + ei * 8, y + LAYER_HEIGHT - 6, 2, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    }
  }

  function drawPlayhead(w, h) {
    var startNs = scrollX / pixelsPerNs;
    var x = (playheadNs - startNs) * pixelsPerNs;
    if (x < 0 || x > w) return;

    ctx.strokeStyle = COLORS.playhead;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  function fetchTimeline() {
    fetch('/api/timeline')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        state = data;
        if (state.duration_ns > 0 && window.AVE.preview) {
          window.AVE.preview.setDuration(state.duration_ns);
        }
        render();
      })
      .catch(function() {});
  }

  function debouncedFetch() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(fetchTimeline, 200);
  }

  // Click to select clip
  canvas.addEventListener('click', function(e) {
    var rect = canvas.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;
    var startNs = scrollX / pixelsPerNs;

    selectedClipId = null;
    for (var li = 0; li < state.layers.length; li++) {
      var y = HEADER_HEIGHT + li * (LAYER_HEIGHT + LAYER_GAP);
      if (my < y || my > y + LAYER_HEIGHT) continue;
      for (var ci = 0; ci < state.layers[li].clips.length; ci++) {
        var clip = state.layers[li].clips[ci];
        var cx = (clip.start_ns - startNs) * pixelsPerNs;
        var cw = clip.duration_ns * pixelsPerNs;
        if (mx >= cx && mx <= cx + cw) {
          selectedClipId = clip.id;
          // Send context to chat input
          var input = document.getElementById('chat-input');
          if (input && !input.value.includes('[clip:')) {
            input.value = '[clip: ' + clip.id + ' "' + clip.name + '"] ' + input.value;
          }
          break;
        }
      }
    }

    // Click on timescale = seek
    if (my < HEADER_HEIGHT) {
      var clickNs = startNs + mx / pixelsPerNs;
      playheadNs = Math.max(0, clickNs);
      if (window.AVE.preview) { window.AVE.preview.seekTo(playheadNs); }
    }

    render();
  });

  // Scroll to zoom
  canvas.addEventListener('wheel', function(e) {
    e.preventDefault();
    if (e.ctrlKey) {
      var factor = e.deltaY > 0 ? 0.8 : 1.25;
      pixelsPerNs *= factor;
      pixelsPerNs = Math.max(0.000000001, Math.min(0.001, pixelsPerNs));
    } else {
      scrollX += e.deltaY * 2;
      scrollX = Math.max(0, scrollX);
    }
    render();
  });

  window.addEventListener('resize', resize);

  window.AVE.timeline = {
    refresh: debouncedFetch,
    setPlayhead: function(ns) { playheadNs = ns; render(); }
  };

  resize();
  fetchTimeline();
})();
```

- [ ] **Step 3: Create chat.js**

Create `src/ave/web/client/chat.js`:

```javascript
/**
 * Chat panel — WebSocket client with streaming text display.
 */
(function() {
  'use strict';

  var messagesEl = document.getElementById('chat-messages');
  var inputEl = document.getElementById('chat-input');
  var sendBtn = document.getElementById('btn-send');
  var statusEl = document.getElementById('chat-status');

  var ws = null;
  var sessionToken = sessionStorage.getItem('ave_session') || '';
  var currentBubble = null;
  var processing = false;

  window.AVE = window.AVE || {};

  function connect() {
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var url = proto + '//' + location.host + '/ws/chat';
    if (sessionToken) url += '?session=' + sessionToken;

    ws = new WebSocket(url);
    ws.onopen = function() { statusEl.textContent = 'Connected'; };
    ws.onclose = function() {
      statusEl.textContent = 'Disconnected — reconnecting...';
      setTimeout(connect, 2000);
    };
    ws.onmessage = function(ev) {
      var msg = JSON.parse(ev.data);
      handleMessage(msg);
    };
  }

  function handleMessage(msg) {
    switch (msg.type) {
      case 'connected':
        sessionToken = msg.session;
        sessionStorage.setItem('ave_session', sessionToken);
        statusEl.textContent = 'Connected (session: ' + sessionToken.substring(0, 8) + '...)';
        break;

      case 'text_delta':
        if (!currentBubble) {
          currentBubble = addBubble('agent');
        }
        currentBubble.textContent += msg.text;
        scrollToBottom();
        break;

      case 'tool_start':
        addToolMessage('[Using ' + msg.tool_name + '...]');
        break;

      case 'tool_done':
        // Tool completed — could update the tool indicator
        break;

      case 'timeline_updated':
        if (window.AVE.timeline) { window.AVE.timeline.refresh(); }
        break;

      case 'done':
        currentBubble = null;
        setProcessing(false);
        break;

      case 'error':
        addErrorMessage(msg.message);
        currentBubble = null;
        setProcessing(false);
        break;

      case 'busy':
        addToolMessage('[Agent is still processing...]');
        break;
    }
  }

  function addBubble(role) {
    var div = document.createElement('div');
    div.className = 'msg msg-' + role;
    messagesEl.appendChild(div);
    scrollToBottom();
    return div;
  }

  function addToolMessage(text) {
    var div = document.createElement('div');
    div.className = 'msg msg-tool';
    div.textContent = text;
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function addErrorMessage(text) {
    var div = document.createElement('div');
    div.className = 'msg msg-error';
    div.textContent = 'Error: ' + text;
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setProcessing(val) {
    processing = val;
    sendBtn.disabled = val;
    inputEl.disabled = val;
    if (!val) inputEl.focus();
  }

  function sendMessage() {
    var text = inputEl.value.trim();
    if (!text || processing) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    addBubble('user').textContent = text;
    ws.send(JSON.stringify({ type: 'message', text: text }));
    inputEl.value = '';
    setProcessing(true);
  }

  sendBtn.addEventListener('click', sendMessage);
  inputEl.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  window.AVE.chat = {
    appendToInput: function(text) {
      inputEl.value = text + inputEl.value;
      inputEl.focus();
    }
  };

  connect();
})();
```

- [ ] **Step 4: Create assets.js**

Create `src/ave/web/client/assets.js`:

```javascript
/**
 * Asset browser — grid of ingested media with thumbnails.
 */
(function() {
  'use strict';

  var browserEl = document.getElementById('asset-browser');
  var gridEl = document.getElementById('asset-grid');
  var toggleBtn = document.getElementById('btn-assets');

  window.AVE = window.AVE || {};

  function formatDuration(ns) {
    var s = Math.floor(ns / 1000000000);
    var m = Math.floor(s / 60);
    s = s % 60;
    return m + ':' + String(s).padStart(2, '0');
  }

  function fetchAssets() {
    fetch('/api/assets')
      .then(function(r) { return r.json(); })
      .then(function(data) { renderAssets(data.assets || []); })
      .catch(function() {});
  }

  function renderAssets(assets) {
    gridEl.innerHTML = '';
    for (var i = 0; i < assets.length; i++) {
      var a = assets[i];
      var card = document.createElement('div');
      card.className = 'asset-card';
      card.innerHTML =
        '<img src="' + a.thumbnail_url + '" alt="' + a.name + '">' +
        '<div class="info">' +
          '<div class="name">' + a.name + '</div>' +
          '<div class="meta">' + formatDuration(a.duration_ns) + ' | ' + a.resolution + '</div>' +
        '</div>';
      card.addEventListener('click', (function(asset) {
        return function() {
          if (window.AVE.chat) {
            window.AVE.chat.appendToInput('Add "' + asset.name + '" to the timeline. ');
          }
          browserEl.classList.add('hidden');
        };
      })(a));
      gridEl.appendChild(card);
    }
  }

  toggleBtn.addEventListener('click', function() {
    browserEl.classList.toggle('hidden');
    if (!browserEl.classList.contains('hidden')) { fetchAssets(); }
  });
})();
```

- [ ] **Step 5: Commit**

```bash
git add src/ave/web/client/preview.js src/ave/web/client/timeline.js \
  src/ave/web/client/chat.js src/ave/web/client/assets.js
git commit -m "feat(web): add frontend JS — chat, timeline, preview, assets"
```

---

## Chunk 4: Chat Session with Anthropic SDK

### Task 9: ChatSession — agentic loop with Anthropic streaming

**Files:**
- Modify: `src/ave/web/chat.py`
- Create: `tests/test_web/test_chat_integration.py`

- [ ] **Step 1: Write failing integration tests with mock Anthropic client**

Create `tests/test_web/test_chat_integration.py`:

```python
"""Integration tests for chat — uses mock Anthropic client."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from aiohttp import web
    from aiohttp.test_utils import AioHTTPTestCase
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


@pytest.fixture
def app_with_mock_anthropic(tmp_path):
    """Create app with mocked Anthropic client."""
    from ave.web.app import create_app

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "assets").mkdir()
    app = create_app(project_dir=project_dir)
    return app


class TestChatSessionBusy:
    """Concurrent message rejection."""

    @pytest.mark.asyncio
    async def test_send_while_not_connected(self, aiohttp_client, app_with_mock_anthropic):
        """Messages before agent connected get error."""
        client = await aiohttp_client(app_with_mock_anthropic)
        ws = await client.ws_connect("/ws/chat")
        connected = await ws.receive_json()
        assert connected["type"] == "connected"

        # Send a message — should get error since no real agent
        await ws.send_json({"type": "message", "text": "hello"})
        resp = await ws.receive_json()
        # Without anthropic installed, should get an error
        assert resp["type"] in ("error", "done")
        await ws.close()


class TestChatSessionProtocol:
    """Protocol-level tests."""

    @pytest.mark.asyncio
    async def test_invalid_json(self, aiohttp_client, app_with_mock_anthropic):
        client = await aiohttp_client(app_with_mock_anthropic)
        ws = await client.ws_connect("/ws/chat")
        await ws.receive_json()  # connected

        await ws.send_str("not json")
        resp = await ws.receive_json()
        assert resp["type"] == "error"
        assert "Invalid JSON" in resp["message"]
        await ws.close()

    @pytest.mark.asyncio
    async def test_missing_type_field(self, aiohttp_client, app_with_mock_anthropic):
        client = await aiohttp_client(app_with_mock_anthropic)
        ws = await client.ws_connect("/ws/chat")
        await ws.receive_json()  # connected

        await ws.send_json({"text": "hello"})
        resp = await ws.receive_json()
        assert resp["type"] == "error"
        await ws.close()

    @pytest.mark.asyncio
    async def test_session_token_persisted(self, aiohttp_client, app_with_mock_anthropic):
        client = await aiohttp_client(app_with_mock_anthropic)

        ws1 = await client.ws_connect("/ws/chat")
        msg1 = await ws1.receive_json()
        token = msg1["session"]
        await ws1.close()

        ws2 = await client.ws_connect(f"/ws/chat?session={token}")
        msg2 = await ws2.receive_json()
        assert msg2["session"] == token
        await ws2.close()
```

- [ ] **Step 2: Run tests to verify they fail or pass with current stub**

Run: `python3 -m pytest tests/test_web/test_chat_integration.py -v`
Expected: Tests should pass with the current stub (returns error "Agent not yet connected")

- [ ] **Step 3: Implement ChatSession with Anthropic SDK integration**

Update `src/ave/web/chat.py` — add the ChatSession class after the protocol functions:

```python
# --- ChatSession: requires aiohttp + anthropic ---

import asyncio
import logging

logger = logging.getLogger(__name__)


class ChatSession:
    """Bridges WebSocket with Anthropic API tool-use loop.

    Manages conversation history, concurrency control, and cancellation.
    """

    def __init__(self, orchestrator, timeline_model):
        self._orchestrator = orchestrator
        self._timeline = timeline_model
        self._messages: list[dict] = []
        self._processing = False
        self._cancel_event = asyncio.Event()

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    @property
    def is_processing(self) -> bool:
        return self._processing

    async def handle_message(self, ws, text: str) -> None:
        """Process user message through the agentic tool-use loop."""
        if self._processing:
            await ws.send_json(format_busy())
            return

        self._processing = True
        self._cancel_event.clear()

        try:
            await self._agentic_loop(ws, text)
        except Exception as e:
            logger.exception("Agent error")
            await ws.send_json(format_error(str(e)))
        finally:
            self._processing = False

    async def _agentic_loop(self, ws, text: str) -> None:
        """Run the tool-use loop with Anthropic streaming API."""
        try:
            import anthropic
        except ImportError:
            await ws.send_json(format_error(
                "anthropic package not installed. Install with: pip install ave[web]"
            ))
            await ws.send_json(format_done(self._orchestrator.turn_count))
            return

        self._messages.append({"role": "user", "content": text})

        loop = asyncio.get_running_loop()
        client = anthropic.AsyncAnthropic()
        tools = self._get_tools_json()
        system_prompt = self._orchestrator.get_system_prompt()

        while True:
            if self._cancel_event.is_set():
                await ws.send_json(format_done(self._orchestrator.turn_count))
                return

            tool_calls = []

            async with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                tools=tools,
                messages=self._messages,
            ) as stream:
                async for event in stream:
                    if self._cancel_event.is_set():
                        break
                    await self._forward_stream_event(ws, event, tool_calls)

            response = await stream.get_final_message()
            self._messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                await ws.send_json(format_done(self._orchestrator.turn_count))
                return

            # Execute tool calls
            tool_results = []
            for tc in tool_calls:
                result_str = await loop.run_in_executor(
                    None,
                    self._orchestrator.handle_tool_call,
                    tc["name"],
                    tc["input"],
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result_str,
                })
                await ws.send_json(format_tool_done(tc["id"]))

                if self._is_timeline_modifying(tc["name"], tc["input"]):
                    self._timeline.reload_from_xges()
                    await ws.send_json(format_timeline_updated())

            self._messages.append({"role": "user", "content": tool_results})

    async def _forward_stream_event(self, ws, event, tool_calls: list) -> None:
        """Forward a streaming event to the WebSocket client."""
        if not hasattr(event, 'type'):
            return

        if event.type == 'text':
            await ws.send_json(format_text_delta(event.text))
        elif event.type == 'content_block_start':
            if hasattr(event, 'content_block') and event.content_block.type == 'tool_use':
                tc = {
                    "id": event.content_block.id,
                    "name": event.content_block.name,
                    "input": {},
                }
                tool_calls.append(tc)
                await ws.send_json(format_tool_start(
                    event.content_block.name,
                    event.content_block.id,
                ))
        elif event.type == 'input_json':
            pass  # Accumulate silently — input assembled by SDK

    def _get_tools_json(self) -> list[dict]:
        """Convert Orchestrator meta-tools to Anthropic API tool format."""
        return [
            {
                "name": mt.name,
                "description": mt.description,
                "input_schema": mt.parameters,
            }
            for mt in self._orchestrator.get_meta_tools()
        ]

    def _is_timeline_modifying(self, tool_name: str, tool_input: dict) -> bool:
        """Check if a meta-tool call modifies the timeline."""
        if tool_name != "call_tool":
            return False
        inner = tool_input.get("tool_name", "")
        modifying_domains = {"editing", "compositing", "motion_graphics", "scene"}
        try:
            schema = self._orchestrator.session.registry.get_tool_schema(inner)
            return schema.domain in modifying_domains
        except Exception:
            return False

    def cancel(self) -> None:
        """Signal cancellation of current processing."""
        self._cancel_event.set()
```

- [ ] **Step 4: Update app.py to use ChatSession**

In `src/ave/web/app.py`, update `_handle_ws_chat` to create and use ChatSession:

```python
async def _handle_ws_chat(request: web.Request) -> web.WebSocketResponse:
    """WebSocket endpoint for chat — handles session management."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    token = request.query.get("session", str(uuid.uuid4()))
    sessions = request.app["sessions"]

    if token not in sessions:
        from ave.web.chat import ChatSession
        from ave.agent.orchestrator import Orchestrator
        from ave.agent.session import EditingSession

        editing_session = EditingSession()
        orchestrator = Orchestrator(editing_session)
        sessions[token] = ChatSession(orchestrator, request.app["timeline_model"])

    session = sessions[token]
    await ws.send_json(format_connected(token))

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                parsed = parse_client_message(msg.data)
                if parsed["type"] == "error":
                    await ws.send_json(format_error(parsed["message"]))
                elif parsed["type"] == "message":
                    await session.handle_message(ws, parsed["text"])
                elif parsed["type"] == "cancel":
                    session.cancel()
            elif msg.type == web.WSMsgType.ERROR:
                break
    finally:
        pass  # Session stays for reconnect

    return ws
```

- [ ] **Step 5: Run all tests**

Run: `python3 -m pytest tests/test_web/ -v`
Expected: All passing

- [ ] **Step 6: Commit**

```bash
git add src/ave/web/chat.py src/ave/web/app.py \
  tests/test_web/test_chat_integration.py
git commit -m "feat(web): add ChatSession with Anthropic SDK agentic loop"
```

---

## Chunk 5: Preview Integration + Final Wiring

### Task 10: Mount preview server as sub-app

**Files:**
- Modify: `src/ave/web/app.py`

- [ ] **Step 1: Add preview sub-app mounting**

In `src/ave/web/app.py`, update `create_app` to optionally mount the preview server:

```python
def create_app(
    project_dir: Path,
    xges_path: Path | None = None,
    preview_video_path: Path | None = None,
) -> web.Application:
    # ... existing setup ...

    # Mount preview server as sub-app at /preview/
    try:
        from ave.preview.cache import SegmentCache
        from ave.preview.server import PreviewServer

        segments_dir = project_dir / "cache" / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        cache = SegmentCache(segments_dir)
        preview = PreviewServer(cache, segments_dir, preview_video_path)
        preview_app = preview.create_app()
        app.add_subapp("/preview/", preview_app)
    except ImportError:
        pass  # Preview deps not installed — skip

    return app
```

- [ ] **Step 2: Add thumbnail endpoint**

In `src/ave/web/app.py`, add route and handler:

```python
app.router.add_get("/api/assets/{asset_id}/thumbnail", _handle_thumbnail)
```

```python
async def _handle_thumbnail(request: web.Request) -> web.Response:
    """GET /api/assets/{asset_id}/thumbnail — serve asset thumbnail."""
    # Return a 1x1 gray pixel placeholder for now
    # Real implementation would extract first frame from proxy
    pixel = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    return web.Response(body=pixel, content_type="image/png")
```

- [ ] **Step 3: Run integration tests**

Run: `python3 -m pytest tests/test_web/ -v`
Expected: All passing

- [ ] **Step 4: Commit**

```bash
git add src/ave/web/app.py
git commit -m "feat(web): mount preview sub-app and add thumbnail endpoint"
```

### Task 11: Final verification and cleanup

- [ ] **Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All new tests pass, no regressions in existing tests

- [ ] **Step 2: Run linter**

Run: `python3 -m ruff check src/ave/web/ tests/test_web/`
Expected: Clean (or fix any issues)

- [ ] **Step 3: Verify the server starts**

Run: `python3 -c "from ave.web.app import create_app; import tempfile; from pathlib import Path; p = Path(tempfile.mkdtemp()); (p / 'assets').mkdir(); app = create_app(p); print('App created successfully')"`
Expected: "App created successfully"

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(web): Phase 8 Web UI complete — chat, timeline, preview, assets"
```
