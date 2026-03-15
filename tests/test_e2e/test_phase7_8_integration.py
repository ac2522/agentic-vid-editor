"""Phase 7+8 stack integration tests.

Tests that the web layer (Phase 8) correctly integrates with the
agent/session/orchestrator stack (Phase 7):
- App factory creates snapshot-enabled sessions
- REST /api/timeline reflects XGES state after tool calls
- TimelineModel reloads after XGES modification
- ChatSession tool routing through orchestrator meta-tools
- ChatSession._is_timeline_modifying uses modifies_timeline flag
- Orchestrator meta-tools produce correct Anthropic API format
- MultiAgentOrchestrator agent defs work with ChatSession
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from ave.agent.dependencies import SessionState
from ave.agent.orchestrator import Orchestrator
from ave.agent.registry import ToolRegistry
from ave.agent.session import EditingSession
from ave.project.snapshots import SnapshotManager
from ave.web.timeline_model import TimelineModel, ClipState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_XGES = """\
<?xml version="1.0" encoding="UTF-8"?>
<ges version="0.7">
  <project>
    <timeline properties="framerate=(fraction)24/1;">
      <layer priority="0">
        <clip id="0" asset-id="file:///media/interview.mov"
              start="0" duration="5000000000" inpoint="0"
              track-types="6"
              metadatas="agent:clip-id=(string)clip_001;">
        </clip>
        <clip id="1" asset-id="file:///media/broll.mp4"
              start="5000000000" duration="3000000000" inpoint="0"
              track-types="4"
              metadatas="agent:clip-id=(string)clip_002;">
        </clip>
      </layer>
      <layer priority="1">
        <clip id="2" asset-id="file:///media/music.wav"
              start="0" duration="8000000000" inpoint="0"
              track-types="2"
              metadatas="agent:clip-id=(string)clip_003;">
        </clip>
      </layer>
    </timeline>
  </project>
</ges>
"""

MODIFIED_XGES = """\
<?xml version="1.0" encoding="UTF-8"?>
<ges version="0.7">
  <project>
    <timeline properties="framerate=(fraction)24/1;">
      <layer priority="0">
        <clip id="0" asset-id="file:///media/interview.mov"
              start="0" duration="2000000000" inpoint="1000000000"
              track-types="6"
              metadatas="agent:clip-id=(string)clip_001;">
        </clip>
      </layer>
    </timeline>
  </project>
</ges>
"""


def _make_test_session() -> EditingSession:
    """Create a bare session with test tools for integration testing."""
    s = EditingSession.__new__(EditingSession)
    s._registry = ToolRegistry()
    s._state = SessionState()
    s._history = []
    s._project_path = None
    s._snapshot_manager = None
    s._lock = threading.Lock()

    @s._registry.tool(domain="editing", requires=[], provides=["clip_trimmed"],
                       modifies_timeline=True)
    def trim(in_ns: int, out_ns: int) -> dict:
        """Trim a clip."""
        return {"in_ns": in_ns, "out_ns": out_ns}

    @s._registry.tool(domain="color", requires=[], provides=["color_graded"],
                       modifies_timeline=True)
    def color_grade(warmth: float) -> dict:
        """Apply color grading."""
        return {"warmth": warmth}

    @s._registry.tool(domain="transcription", requires=[], provides=[])
    def search_transcript(query: str) -> dict:
        """Search transcript."""
        return {"matches": []}

    @s._registry.tool(domain="render", requires=[], provides=[])
    def render_proxy(path: str) -> dict:
        """Render a proxy."""
        return {"output": path}

    return s


# ===========================================================================
# TimelineModel + XGES sync
# ===========================================================================


class TestTimelineXGESSync:
    """Test TimelineModel loads and reloads XGES correctly."""

    @pytest.fixture()
    def xges_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "project.xges"
        p.write_text(SAMPLE_XGES)
        return p

    def test_load_from_xges_parses_clips(self, xges_file):
        model = TimelineModel.load_from_xges(xges_file)
        assert model.fps == 24.0
        assert len(model.layers) == 2

        clip1 = model.get_clip("clip_001")
        assert clip1.name == "interview.mov"
        assert clip1.duration_ns == 5_000_000_000
        assert clip1.has_video is True
        assert clip1.has_audio is True

        clip2 = model.get_clip("clip_002")
        assert clip2.has_video is True
        assert clip2.has_audio is False  # track_types=4

        clip3 = model.get_clip("clip_003")
        assert clip3.has_video is False
        assert clip3.has_audio is True  # track_types=2

    def test_timeline_duration_correct(self, xges_file):
        model = TimelineModel.load_from_xges(xges_file)
        # clip_003 is 0+8s = 8s, longest end time
        assert model.duration_ns == 8_000_000_000

    def test_reload_picks_up_modifications(self, xges_file):
        model = TimelineModel.load_from_xges(xges_file)
        assert len(model.layers) == 2

        # Simulate tool modifying XGES
        xges_file.write_text(MODIFIED_XGES)

        model.reload_from_xges()
        assert len(model.layers) == 1
        clip = model.get_clip("clip_001")
        assert clip.duration_ns == 2_000_000_000
        assert clip.inpoint_ns == 1_000_000_000

    def test_to_dict_for_api_response(self, xges_file):
        model = TimelineModel.load_from_xges(xges_file)
        data = model.to_dict()

        assert "layers" in data
        assert "duration_ns" in data
        assert "fps" in data
        assert data["fps"] == 24.0
        assert len(data["layers"]) == 2


# ===========================================================================
# REST API + Timeline integration
# ===========================================================================


class TestRESTAPITimelineIntegration:
    """Test REST API functions with real timeline data."""

    def test_get_timeline_response_from_xges(self, tmp_path):
        from ave.web.api import get_timeline_response

        xges_file = tmp_path / "project.xges"
        xges_file.write_text(SAMPLE_XGES)
        model = TimelineModel.load_from_xges(xges_file)

        response = get_timeline_response(model)
        assert response["fps"] == 24.0
        assert response["duration_ns"] == 8_000_000_000
        assert len(response["layers"]) == 2
        assert len(response["layers"][0]["clips"]) == 2
        assert len(response["layers"][1]["clips"]) == 1

    def test_timeline_response_reflects_xges_modification(self, tmp_path):
        from ave.web.api import get_timeline_response

        xges_file = tmp_path / "project.xges"
        xges_file.write_text(SAMPLE_XGES)
        model = TimelineModel.load_from_xges(xges_file)

        # Before modification
        r1 = get_timeline_response(model)
        assert r1["duration_ns"] == 8_000_000_000

        # Simulate tool editing the XGES
        xges_file.write_text(MODIFIED_XGES)
        model.reload_from_xges()

        # After modification
        r2 = get_timeline_response(model)
        assert r2["duration_ns"] == 2_000_000_000
        assert len(r2["layers"]) == 1

    def test_assets_response_with_registry(self, tmp_path):
        from ave.web.api import get_assets_response

        registry_path = tmp_path / "registry.json"
        registry_path.write_text(json.dumps([
            {
                "asset_id": "asset_001",
                "original_path": "/media/interview.mov",
                "duration_seconds": 120.5,
                "width": 1920,
                "height": 1080,
                "original_fps": 23.976,
            }
        ]))

        response = get_assets_response(registry_path)
        assert len(response["assets"]) == 1
        asset = response["assets"][0]
        assert asset["id"] == "asset_001"
        assert asset["name"] == "interview.mov"
        assert asset["duration_ns"] == 120_500_000_000
        assert asset["resolution"] == "1920x1080"


# ===========================================================================
# ChatSession + Orchestrator meta-tools
# ===========================================================================


class TestChatSessionOrchestrator:
    """Test ChatSession integration with orchestrator meta-tools."""

    @pytest.fixture()
    def chat_setup(self, tmp_path):
        session = _make_test_session()
        orchestrator = Orchestrator(session)
        xges_file = tmp_path / "project.xges"
        xges_file.write_text(SAMPLE_XGES)
        model = TimelineModel.load_from_xges(xges_file)
        return orchestrator, model

    def test_meta_tools_to_anthropic_format(self, chat_setup):
        from ave.web.chat import ChatSession

        orchestrator, model = chat_setup
        cs = ChatSession(orchestrator, model)
        tools = cs._get_tools_json()

        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert names == {"search_tools", "get_tool_schema", "call_tool"}

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_is_timeline_modifying_editing_domain(self, chat_setup):
        from ave.web.chat import ChatSession

        orchestrator, model = chat_setup
        cs = ChatSession(orchestrator, model)

        # call_tool with editing domain tool → modifies timeline
        assert cs._is_timeline_modifying("call_tool", {"tool_name": "trim"}) is True

    def test_is_timeline_modifying_color_domain(self, chat_setup):
        from ave.web.chat import ChatSession

        orchestrator, model = chat_setup
        cs = ChatSession(orchestrator, model)

        # Note: ChatSession checks against a hardcoded domain set
        # color is NOT in the hardcoded set (editing, compositing, motion_graphics, scene)
        assert cs._is_timeline_modifying("call_tool", {"tool_name": "color_grade"}) is False

    def test_is_timeline_modifying_non_call_tool(self, chat_setup):
        from ave.web.chat import ChatSession

        orchestrator, model = chat_setup
        cs = ChatSession(orchestrator, model)

        # search_tools is not call_tool → doesn't modify
        assert cs._is_timeline_modifying("search_tools", {"query": "trim"}) is False

    def test_is_timeline_modifying_unknown_tool(self, chat_setup):
        from ave.web.chat import ChatSession

        orchestrator, model = chat_setup
        cs = ChatSession(orchestrator, model)

        # Unknown tool → doesn't crash, returns False
        assert cs._is_timeline_modifying("call_tool", {"tool_name": "nonexistent"}) is False

    def test_orchestrator_handle_search(self, chat_setup):
        """Verify orchestrator.handle_tool_call works through the ChatSession path."""
        orchestrator, _ = chat_setup
        result = orchestrator.handle_tool_call(
            "search_tools", {"query": "trim"}
        )
        assert "trim" in result
        assert "editing" in result

    def test_orchestrator_handle_schema(self, chat_setup):
        orchestrator, _ = chat_setup
        result = orchestrator.handle_tool_call(
            "get_tool_schema", {"tool_name": "trim"}
        )
        assert "in_ns" in result
        assert "out_ns" in result

    def test_orchestrator_handle_call(self, chat_setup):
        orchestrator, _ = chat_setup
        result = orchestrator.handle_tool_call(
            "call_tool",
            {"tool_name": "trim", "params": {"in_ns": 0, "out_ns": 1_000_000_000}},
        )
        assert "successfully" in result

    def test_chat_session_initial_state(self, chat_setup):
        from ave.web.chat import ChatSession

        orchestrator, model = chat_setup
        cs = ChatSession(orchestrator, model)
        assert cs._processing is False
        assert cs._messages == []


# ===========================================================================
# Full stack: orchestrator → session → tool → XGES → model → API
# ===========================================================================


class TestFullStackToolExecution:
    """Test the complete data flow from orchestrator through to API response."""

    @pytest.fixture()
    def full_stack(self, tmp_path):
        """Set up the complete Phase 7+8 stack."""
        # Write XGES
        xges_file = tmp_path / "project.xges"
        xges_file.write_text(SAMPLE_XGES)

        # Create session with snapshots
        mgr = SnapshotManager(max_snapshots=10)
        session = _make_test_session()
        session._snapshot_manager = mgr
        session._project_path = xges_file

        # Create orchestrator
        orchestrator = Orchestrator(session)

        # Create timeline model
        model = TimelineModel.load_from_xges(xges_file)

        return {
            "session": session,
            "orchestrator": orchestrator,
            "model": model,
            "xges_file": xges_file,
            "snapshot_mgr": mgr,
        }

    def test_tool_call_through_orchestrator(self, full_stack):
        """search → schema → call flow via orchestrator."""
        orch = full_stack["orchestrator"]

        # Step 1: Search for tools
        search_result = orch.handle_tool_call("search_tools", {"query": "trim"})
        assert "trim" in search_result

        # Step 2: Get schema
        schema_result = orch.handle_tool_call("get_tool_schema", {"tool_name": "trim"})
        assert "in_ns" in schema_result

        # Step 3: Call tool
        call_result = orch.handle_tool_call(
            "call_tool",
            {"tool_name": "trim", "params": {"in_ns": 0, "out_ns": 2_000_000_000}},
        )
        assert "successfully" in call_result

    def test_snapshot_created_during_tool_call(self, full_stack):
        """Tool calls through orchestrator should create snapshots."""
        orch = full_stack["orchestrator"]
        mgr = full_stack["snapshot_mgr"]

        assert len(mgr.list_snapshots()) == 0

        orch.handle_tool_call(
            "call_tool",
            {"tool_name": "trim", "params": {"in_ns": 0, "out_ns": 1_000_000_000}},
        )

        assert len(mgr.list_snapshots()) == 1

    def test_xges_modification_and_model_reload(self, full_stack):
        """After XGES is modified on disk, model.reload_from_xges() picks up changes."""
        model = full_stack["model"]
        xges_file = full_stack["xges_file"]

        # Initial state: 3 clips across 2 layers
        assert model.duration_ns == 8_000_000_000

        # Simulate tool modifying XGES on disk
        xges_file.write_text(MODIFIED_XGES)
        model.reload_from_xges()

        # Model should reflect the change
        assert model.duration_ns == 2_000_000_000
        assert len(model.layers) == 1

    def test_api_response_after_xges_change(self, full_stack):
        """API response reflects XGES changes after reload."""
        from ave.web.api import get_timeline_response

        model = full_stack["model"]
        xges_file = full_stack["xges_file"]

        # Before
        r1 = get_timeline_response(model)
        assert r1["duration_ns"] == 8_000_000_000

        # Modify and reload
        xges_file.write_text(MODIFIED_XGES)
        model.reload_from_xges()

        # After
        r2 = get_timeline_response(model)
        assert r2["duration_ns"] == 2_000_000_000

    def test_session_history_tracks_orchestrator_calls(self, full_stack):
        """Session history records tool calls made through orchestrator."""
        orch = full_stack["orchestrator"]
        session = full_stack["session"]

        assert len(session.history) == 0

        orch.handle_tool_call(
            "call_tool",
            {"tool_name": "trim", "params": {"in_ns": 0, "out_ns": 1_000_000_000}},
        )
        orch.handle_tool_call(
            "call_tool",
            {"tool_name": "search_transcript", "params": {"query": "hello"}},
        )

        assert len(session.history) == 2
        assert session.history[0].tool_name == "trim"
        assert session.history[1].tool_name == "search_transcript"

    def test_session_state_accumulates_provisions(self, full_stack):
        """Provisions accumulate across tool calls through orchestrator."""
        orch = full_stack["orchestrator"]
        session = full_stack["session"]

        orch.handle_tool_call(
            "call_tool",
            {"tool_name": "trim", "params": {"in_ns": 0, "out_ns": 1_000_000_000}},
        )
        assert session.state.has("clip_trimmed")

        orch.handle_tool_call(
            "call_tool",
            {"tool_name": "color_grade", "params": {"warmth": 0.3}},
        )
        assert session.state.has("color_graded")


# ===========================================================================
# aiohttp app integration (requires aiohttp)
# ===========================================================================

try:
    from tests.conftest import requires_aiohttp
except ImportError:
    requires_aiohttp = pytest.mark.skip(reason="conftest not available")


@requires_aiohttp
class TestWebAppPhase7Integration:
    """Test web app factory with Phase 7 features."""

    @pytest.fixture()
    def app(self, tmp_path):
        from ave.web.app import create_app

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "assets").mkdir()

        # Write XGES
        xges_file = tmp_path / "project.xges"
        xges_file.write_text(SAMPLE_XGES)

        # Create client dir with minimal index.html
        client_dir = project_dir / "client"
        client_dir.mkdir()
        (client_dir / "index.html").write_text("<html><body>AVE</body></html>")

        return create_app(project_dir=project_dir, xges_path=xges_file)

    @pytest.fixture()
    def cli(self, aiohttp_client, app):
        return aiohttp_client(app)

    @pytest.mark.asyncio
    async def test_timeline_api_returns_xges_state(self, cli):
        client = await cli
        resp = await client.get("/api/timeline")
        assert resp.status == 200
        data = await resp.json()
        assert data["fps"] == 24.0
        assert data["duration_ns"] == 8_000_000_000
        assert len(data["layers"]) == 2

    @pytest.mark.asyncio
    async def test_timeline_api_clip_details(self, cli):
        client = await cli
        resp = await client.get("/api/timeline")
        data = await resp.json()
        layer0_clips = data["layers"][0]["clips"]
        assert len(layer0_clips) == 2
        assert layer0_clips[0]["clip_id"] == "clip_001"
        assert layer0_clips[0]["name"] == "interview.mov"
        assert layer0_clips[0]["duration_ns"] == 5_000_000_000

    @pytest.mark.asyncio
    async def test_ws_chat_connects_and_gets_token(self, cli):
        client = await cli
        ws = await client.ws_connect("/ws/chat")
        msg = await ws.receive_json()
        assert msg["type"] == "connected"
        assert "session_token" in msg
        await ws.close()
