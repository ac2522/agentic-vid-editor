"""E2E tests for the clip identity workflow through the full agent stack.

Tests the complete meta-tool chain that an LLM would follow:
1. search_tools → find list_clips
2. get_tool_schema → learn parameters
3. call_tool(list_clips) → get clip data, clip_exists provision set
4. search_tools → find editing tools (now accessible)
5. call_tool(trim) → execute with data from list_clips
6. Verify provisions, history, and state accumulate correctly
"""

from __future__ import annotations


import pytest

from ave.agent.orchestrator import Orchestrator
from ave.agent.session import EditingSession
from ave.project.snapshots import SnapshotManager


SAMPLE_XGES = """\
<?xml version="1.0" encoding="UTF-8"?>
<ges version="0.7">
  <project>
    <timeline properties="framerate=(fraction)24/1;">
      <layer priority="0">
        <clip id="0" asset-id="file:///media/interview.mov"
              start="0" duration="10000000000" inpoint="0"
              track-types="6"
              metadatas="agent:clip-id=(string)clip_001;">
        </clip>
        <clip id="1" asset-id="file:///media/broll.mp4"
              start="10000000000" duration="5000000000" inpoint="0"
              track-types="4"
              metadatas="agent:clip-id=(string)clip_002;">
        </clip>
      </layer>
    </timeline>
  </project>
</ges>
"""


class TestClipIdentityE2EWorkflow:
    """Full E2E test of the clip identity workflow through orchestrator meta-tools."""

    @pytest.fixture()
    def setup(self, tmp_path):
        xges_file = tmp_path / "project.xges"
        xges_file.write_text(SAMPLE_XGES)

        session = EditingSession()
        session.load_project(xges_file)
        orchestrator = Orchestrator(session)

        return orchestrator, session, xges_file

    def test_step1_search_finds_list_clips(self, setup):
        """Agent searches for clip-related tools and finds list_clips."""
        orchestrator, _, _ = setup
        result = orchestrator.handle_tool_call("search_tools", {"query": "clip list"})
        assert "list_clips" in result

    def test_step2_get_schema_for_list_clips(self, setup):
        """Agent gets the full schema for list_clips to learn its parameters."""
        orchestrator, _, _ = setup
        result = orchestrator.handle_tool_call(
            "get_tool_schema", {"tool_name": "list_clips"}
        )
        assert "xges_path" in result
        assert "clip_exists" in result  # Shows it provides clip_exists

    def test_step3_call_list_clips_returns_clip_data(self, setup):
        """Agent calls list_clips and gets concrete clip identities."""
        orchestrator, _, xges_file = setup
        result = orchestrator.handle_tool_call(
            "call_tool",
            {"tool_name": "list_clips", "params": {"xges_path": str(xges_file)}},
        )
        assert "clip_001" in result
        assert "clip_002" in result
        assert "interview.mov" in result
        assert "broll.mp4" in result
        assert "successfully" in result

    def test_step3_sets_clip_exists_provision(self, setup):
        """After calling list_clips, clip_exists provision is set."""
        orchestrator, session, xges_file = setup

        assert not session.state.has("clip_exists")

        orchestrator.handle_tool_call(
            "call_tool",
            {"tool_name": "list_clips", "params": {"xges_path": str(xges_file)}},
        )

        assert session.state.has("clip_exists")

    def test_step4_search_finds_trim_after_clips_known(self, setup):
        """Agent searches for trim tool (available because clip_exists is set)."""
        orchestrator, _, xges_file = setup
        # First get clip_exists
        orchestrator.handle_tool_call(
            "call_tool",
            {"tool_name": "list_clips", "params": {"xges_path": str(xges_file)}},
        )

        result = orchestrator.handle_tool_call("search_tools", {"query": "trim cut"})
        assert "trim" in result

    def test_step5_call_trim_with_clip_data(self, setup):
        """Agent calls trim using data from list_clips (clip duration = 10s)."""
        orchestrator, session, xges_file = setup

        # Step 3: list clips
        orchestrator.handle_tool_call(
            "call_tool",
            {"tool_name": "list_clips", "params": {"xges_path": str(xges_file)}},
        )

        # Step 5: trim clip_001 (duration 10s) to first 5 seconds
        result = orchestrator.handle_tool_call(
            "call_tool",
            {
                "tool_name": "trim",
                "params": {
                    "clip_duration_ns": 10_000_000_000,
                    "in_ns": 0,
                    "out_ns": 5_000_000_000,
                },
            },
        )
        assert "successfully" in result

    def test_full_workflow_provisions_accumulate(self, setup):
        """After the full workflow, all expected provisions are set."""
        orchestrator, session, xges_file = setup

        # list_clips → clip_exists
        orchestrator.handle_tool_call(
            "call_tool",
            {"tool_name": "list_clips", "params": {"xges_path": str(xges_file)}},
        )
        # trim → clip_trimmed
        orchestrator.handle_tool_call(
            "call_tool",
            {
                "tool_name": "trim",
                "params": {
                    "clip_duration_ns": 10_000_000_000,
                    "in_ns": 0,
                    "out_ns": 5_000_000_000,
                },
            },
        )

        assert session.state.has("timeline_loaded")
        assert session.state.has("clip_exists")
        assert session.state.has("clip_trimmed")

    def test_full_workflow_history_recorded(self, setup):
        """Session history records all tool calls from the workflow."""
        orchestrator, session, xges_file = setup

        orchestrator.handle_tool_call(
            "call_tool",
            {"tool_name": "list_clips", "params": {"xges_path": str(xges_file)}},
        )
        orchestrator.handle_tool_call(
            "call_tool",
            {
                "tool_name": "trim",
                "params": {
                    "clip_duration_ns": 10_000_000_000,
                    "in_ns": 0,
                    "out_ns": 5_000_000_000,
                },
            },
        )

        history = session.history
        assert len(history) == 2
        assert history[0].tool_name == "list_clips"
        assert history[1].tool_name == "trim"

    def test_timeline_info_provides_context(self, setup):
        """Agent can get timeline summary before listing clips."""
        orchestrator, _, xges_file = setup

        result = orchestrator.handle_tool_call(
            "call_tool",
            {"tool_name": "timeline_info", "params": {"xges_path": str(xges_file)}},
        )
        assert "24.0" in result  # fps
        assert "15000000000" in result or "15" in result  # ~15s duration

    def test_orchestrator_turn_count_increments(self, setup):
        """Each meta-tool call increments the orchestrator turn counter."""
        orchestrator, _, xges_file = setup

        assert orchestrator.turn_count == 0

        orchestrator.handle_tool_call("search_tools", {"query": "clips"})
        assert orchestrator.turn_count == 1

        orchestrator.handle_tool_call("get_tool_schema", {"tool_name": "list_clips"})
        assert orchestrator.turn_count == 2

        orchestrator.handle_tool_call(
            "call_tool",
            {"tool_name": "list_clips", "params": {"xges_path": str(xges_file)}},
        )
        assert orchestrator.turn_count == 3


class TestClipIdentityWithSnapshots:
    """Test clip identity workflow with snapshot manager active."""

    @pytest.fixture()
    def setup(self, tmp_path):
        xges_file = tmp_path / "project.xges"
        xges_file.write_text(SAMPLE_XGES)

        mgr = SnapshotManager(max_snapshots=10)
        session = EditingSession(snapshot_manager=mgr)
        session.load_project(xges_file)
        orchestrator = Orchestrator(session)

        return orchestrator, session, xges_file, mgr

    def test_list_clips_creates_snapshot(self, setup):
        """list_clips through orchestrator should create a snapshot."""
        orchestrator, _, xges_file, mgr = setup

        orchestrator.handle_tool_call(
            "call_tool",
            {"tool_name": "list_clips", "params": {"xges_path": str(xges_file)}},
        )

        snapshots = mgr.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0].label == "before list_clips"

    def test_trim_after_list_clips_creates_second_snapshot(self, setup):
        orchestrator, _, xges_file, mgr = setup

        orchestrator.handle_tool_call(
            "call_tool",
            {"tool_name": "list_clips", "params": {"xges_path": str(xges_file)}},
        )
        orchestrator.handle_tool_call(
            "call_tool",
            {
                "tool_name": "trim",
                "params": {
                    "clip_duration_ns": 10_000_000_000,
                    "in_ns": 0,
                    "out_ns": 5_000_000_000,
                },
            },
        )

        snapshots = mgr.list_snapshots()
        assert len(snapshots) == 2
        assert snapshots[1].label == "before trim"


class TestClipIdentityMultiAgent:
    """Test clip identity through multi-agent orchestrator."""

    def test_list_clips_in_editor_role_tools(self):
        """list_clips should NOT be in editor role (it's project domain)."""
        from ave.agent.multi_agent import MultiAgentOrchestrator
        from ave.agent.roles import EDITOR_ROLE

        session = EditingSession()
        ma = MultiAgentOrchestrator(session)
        editor_tools = ma.get_role_tools(EDITOR_ROLE)
        # list_clips is in project domain, not editing
        assert "list_clips" not in editor_tools

    def test_all_roles_can_discover_list_clips_via_search(self):
        """Any role can search and find list_clips through the meta-tools."""
        session = EditingSession()
        orchestrator = Orchestrator(session)
        result = orchestrator.handle_tool_call("search_tools", {"query": "list clips"})
        assert "list_clips" in result
