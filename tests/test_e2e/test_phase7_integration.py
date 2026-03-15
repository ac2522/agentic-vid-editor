"""Phase 7 cross-module integration tests.

Tests that Phase 7 subsystems work correctly together:
- EditingSession + SnapshotManager (snapshot, execute, rollback, provision restore)
- MultiAgentOrchestrator with real tool registry (role scoping, agent definitions)
- BM25 search engine indexed against full 11-domain registry
- Transition graph recording during real session tool calls
- VerifiedSession wrapping a real EditingSession
- Compositor + RenderScheduler together
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from ave.agent.dependencies import SessionState
from ave.agent.registry import ToolRegistry, PrerequisiteError
from ave.agent.session import EditingSession, ToolCall
from ave.project.snapshots import SnapshotManager


# ---------------------------------------------------------------------------
# Helpers — bare sessions with test tools (bypass _load_all_tools)
# ---------------------------------------------------------------------------

def _make_session(snapshot_mgr: SnapshotManager | None = None) -> EditingSession:
    """Create a bare session with test tools that can succeed or fail."""
    s = EditingSession.__new__(EditingSession)
    s._registry = ToolRegistry()
    s._state = SessionState()
    s._history = []
    s._project_path = None
    s._snapshot_manager = snapshot_mgr
    s._lock = threading.Lock()

    @s._registry.tool(domain="editing", requires=[], provides=["clip_trimmed"],
                       modifies_timeline=True)
    def trim(in_ns: int, out_ns: int) -> dict:
        """Trim a clip to new in/out points."""
        if in_ns >= out_ns:
            raise ValueError("in_ns must be less than out_ns")
        return {"in_ns": in_ns, "out_ns": out_ns, "duration_ns": out_ns - in_ns}

    @s._registry.tool(domain="color", requires=["clip_trimmed"], provides=["color_graded"],
                       modifies_timeline=True)
    def color_grade(warmth: float) -> dict:
        """Apply color grading."""
        return {"warmth": warmth}

    @s._registry.tool(domain="audio", requires=[], provides=["volume_set"],
                       modifies_timeline=True)
    def volume(level_db: float) -> dict:
        """Set audio volume."""
        return {"level_db": level_db}

    @s._registry.tool(domain="project", requires=[], provides=[])
    def probe_media(path: str) -> dict:
        """Probe a media file (read-only, no timeline modification)."""
        return {"path": path, "duration": 10.0}

    return s


# ===========================================================================
# EditingSession + SnapshotManager integration
# ===========================================================================


class TestSessionSnapshotIntegration:
    """Test snapshot capture/restore through the EditingSession."""

    @pytest.fixture()
    def xges_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "project.xges"
        p.write_text("<ges><timeline><layer/></timeline></ges>")
        return p

    @pytest.fixture()
    def session_with_snapshots(self, tmp_path, xges_file):
        mgr = SnapshotManager(max_snapshots=10)
        session = _make_session(snapshot_mgr=mgr)
        session._project_path = xges_file
        session._state.add("timeline_loaded")
        return session

    def test_snapshot_captured_before_tool_call(self, session_with_snapshots):
        session = session_with_snapshots
        assert session.snapshot_manager is not None
        assert len(session.snapshot_manager.list_snapshots()) == 0

        session.call_tool("trim", {"in_ns": 0, "out_ns": 1_000_000_000})

        # Snapshot should have been captured BEFORE the trim
        snapshots = session.snapshot_manager.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0].label == "before trim"

    def test_multiple_calls_create_multiple_snapshots(self, session_with_snapshots):
        session = session_with_snapshots
        session.call_tool("trim", {"in_ns": 0, "out_ns": 1_000_000_000})
        session.call_tool("color_grade", {"warmth": 0.3})

        snapshots = session.snapshot_manager.list_snapshots()
        assert len(snapshots) == 2
        assert snapshots[0].label == "before trim"
        assert snapshots[1].label == "before color_grade"

    def test_failed_tool_auto_restores_provisions(self, session_with_snapshots):
        session = session_with_snapshots
        # Successful trim adds clip_trimmed
        session.call_tool("trim", {"in_ns": 0, "out_ns": 1_000_000_000})
        assert session.state.has("clip_trimmed")

        # Record provisions before failed call
        provisions_before = session.state.provisions

        # This trim should fail (in_ns >= out_ns)
        with pytest.raises(ValueError, match="in_ns must be less than out_ns"):
            session.call_tool("trim", {"in_ns": 5, "out_ns": 3})

        # Provisions should be restored from snapshot
        assert session.state.provisions == provisions_before

    def test_snapshot_provisions_match_pre_call_state(self, session_with_snapshots):
        session = session_with_snapshots
        session.call_tool("volume", {"level_db": -3.0})

        snap = session.snapshot_manager.list_snapshots()[0]
        # The snapshot was taken BEFORE volume, so it should have
        # only timeline_loaded (not volume_set)
        # We need to check the actual snapshot object
        mgr = session.snapshot_manager
        actual_snap = mgr._snapshots[0]
        assert "timeline_loaded" in actual_snap.provisions
        assert "volume_set" not in actual_snap.provisions

    def test_no_snapshot_without_project_loaded(self, tmp_path):
        mgr = SnapshotManager(max_snapshots=10)
        session = _make_session(snapshot_mgr=mgr)
        # No project loaded — no _project_path set
        session.call_tool("probe_media", {"path": "/tmp/test.mp4"})
        assert len(mgr.list_snapshots()) == 0

    def test_no_snapshot_without_manager(self):
        session = _make_session(snapshot_mgr=None)
        # Should work normally without snapshot manager
        result = session.call_tool("probe_media", {"path": "/tmp/test.mp4"})
        assert result["path"] == "/tmp/test.mp4"

    def test_thread_safety_under_concurrent_calls(self, session_with_snapshots):
        """Verify lock serializes concurrent calls (no data corruption)."""
        session = session_with_snapshots
        results = []
        errors = []

        def call_volume(level):
            try:
                r = session.call_tool("volume", {"level_db": level})
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call_volume, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5
        # All calls should have been serialized — 5 snapshots captured
        assert len(session.snapshot_manager.list_snapshots()) == 5


# ===========================================================================
# MultiAgentOrchestrator + real tool registry
# ===========================================================================


class TestMultiAgentWithRealRegistry:
    """Test MultiAgentOrchestrator with the full 11-domain tool registry."""

    @pytest.fixture()
    def full_session(self) -> EditingSession:
        """Create a real EditingSession with all 11 domain tools loaded."""
        return EditingSession()

    def test_agent_definitions_cover_all_domains(self, full_session):
        from ave.agent.multi_agent import MultiAgentOrchestrator

        ma = MultiAgentOrchestrator(full_session)
        defs = ma.get_agent_definitions()

        # Should have definitions for all predefined roles
        assert len(defs) >= 4
        role_names = set(defs.keys())
        assert "editor" in role_names
        assert "colorist" in role_names
        assert "sound_designer" in role_names
        assert "transcriptionist" in role_names

    def test_role_tools_scoped_to_domains(self, full_session):
        from ave.agent.multi_agent import MultiAgentOrchestrator
        from ave.agent.roles import EDITOR_ROLE, COLORIST_ROLE

        ma = MultiAgentOrchestrator(full_session)

        editor_tools = ma.get_role_tools(EDITOR_ROLE)
        colorist_tools = ma.get_role_tools(COLORIST_ROLE)

        # Editor should have editing tools, not color tools
        editor_names = set(editor_tools)
        assert "trim" in editor_names
        assert "split" in editor_names
        assert "color_grade" not in editor_names

        # Colorist should have color tools, not editing tools
        colorist_names = set(colorist_tools)
        assert "color_grade" in colorist_names
        assert "trim" not in colorist_names

    def test_system_prompt_mentions_all_roles(self, full_session):
        from ave.agent.multi_agent import MultiAgentOrchestrator

        ma = MultiAgentOrchestrator(full_session)
        prompt = ma.get_system_prompt()

        assert "editor" in prompt.lower()
        assert "colorist" in prompt.lower()
        assert "sound" in prompt.lower() or "audio" in prompt.lower()
        assert "transcript" in prompt.lower()

    def test_get_role_for_domain(self, full_session):
        from ave.agent.multi_agent import MultiAgentOrchestrator

        ma = MultiAgentOrchestrator(full_session)

        assert ma.get_role_for_domain("editing") is not None
        assert ma.get_role_for_domain("color") is not None
        assert ma.get_role_for_domain("audio") is not None
        assert ma.get_role_for_domain("transcription") is not None
        # No role for render domain
        assert ma.get_role_for_domain("render") is None

    def test_agent_definition_dict_structure(self, full_session):
        from ave.agent.multi_agent import MultiAgentOrchestrator

        ma = MultiAgentOrchestrator(full_session)
        defs = ma.get_agent_definitions()

        for name, defn in defs.items():
            assert "description" in defn, f"Missing description for {name}"
            assert "prompt" in defn, f"Missing prompt for {name}"
            assert isinstance(defn["description"], str)
            assert len(defn["description"]) > 10


# ===========================================================================
# BM25 Search with full registry
# ===========================================================================


class TestBM25WithFullRegistry:
    """Test BM25 search engine against the real 11-domain, 40+ tool registry."""

    @pytest.fixture()
    def search_engine(self):
        from ave.agent.search import ToolSearchEngine

        session = EditingSession()
        engine = ToolSearchEngine()
        count = engine.reindex_all(session.registry)
        return engine, count

    def test_indexes_all_tools(self, search_engine):
        engine, count = search_engine
        assert count >= 34  # At least 34 tools from Phase 1-6

    def test_trim_query_finds_trim_tool(self, search_engine):
        engine, _ = search_engine
        results = engine.search("trim clip")
        assert len(results) > 0
        assert results[0].tool_name == "trim"

    def test_domain_filter_works(self, search_engine):
        engine, _ = search_engine
        results = engine.search("adjust", domain="audio")
        for hit in results:
            assert hit.domain == "audio"

    def test_color_query_finds_color_tools(self, search_engine):
        engine, _ = search_engine
        results = engine.search("color correction warm")
        tool_names = [r.tool_name for r in results]
        assert "color_grade" in tool_names

    def test_crossfade_finds_transition(self, search_engine):
        engine, _ = search_engine
        results = engine.search("crossfade dissolve")
        tool_names = [r.tool_name for r in results]
        assert "transition" in tool_names

    def test_nonsense_query_returns_empty(self, search_engine):
        engine, _ = search_engine
        results = engine.search("xyzzy plugh")
        assert len(results) == 0

    def test_bm25_scores_are_positive(self, search_engine):
        engine, _ = search_engine
        results = engine.search("volume")
        for hit in results:
            assert hit.score > 0


# ===========================================================================
# Transition graph recording during real session
# ===========================================================================


class TestTransitionGraphRecording:
    """Test that tool transitions are recorded correctly during session use."""

    def test_record_transitions_across_tool_calls(self):
        from ave.agent.transitions import ToolTransitionGraph

        session = _make_session()
        graph = ToolTransitionGraph()

        # Simulate a sequence of tool calls
        tools_called = ["probe_media", "trim", "color_grade", "volume"]
        for tool_name in tools_called:
            try:
                if tool_name == "probe_media":
                    session.call_tool(tool_name, {"path": "/test.mp4"})
                elif tool_name == "trim":
                    session.call_tool(tool_name, {"in_ns": 0, "out_ns": 1_000_000_000})
                elif tool_name == "color_grade":
                    session.call_tool(tool_name, {"warmth": 0.5})
                elif tool_name == "volume":
                    session.call_tool(tool_name, {"level_db": -6.0})
            except Exception:
                pass

        # Record transitions from history
        history = session.history
        for i in range(1, len(history)):
            graph.record(history[i - 1].tool_name, history[i].tool_name)

        # Verify transitions were recorded
        assert graph.get_transition_count("probe_media", "trim") == 1
        assert graph.get_transition_count("trim", "color_grade") == 1
        assert graph.get_transition_count("color_grade", "volume") == 1

        # Suggest next after trim
        suggestions = graph.suggest_next("trim")
        assert len(suggestions) > 0
        assert suggestions[0][0] == "color_grade"

    def test_transition_graph_json_round_trip(self):
        from ave.agent.transitions import ToolTransitionGraph

        graph = ToolTransitionGraph()
        graph.record("trim", "color_grade")
        graph.record("trim", "color_grade")
        graph.record("trim", "volume")

        json_str = graph.to_json()
        restored = ToolTransitionGraph.from_json(json_str)

        assert restored.get_transition_count("trim", "color_grade") == 2
        assert restored.get_transition_count("trim", "volume") == 1


# ===========================================================================
# VerifiedSession + real session
# ===========================================================================


class TestVerifiedSessionIntegration:
    """Test VerifiedSession wrapping a real EditingSession."""

    def test_delegates_tool_calls_and_tracks_turn(self):
        from ave.agent.verification import VerifiedSession

        session = _make_session()
        vs = VerifiedSession(session)

        vs.call_tool("probe_media", {"path": "/test.mp4"})
        vs.call_tool("trim", {"in_ns": 0, "out_ns": 2_000_000_000})

        assert vs.turn_tools == ["probe_media", "trim"]
        # Inner session should also have the history
        assert len(session.history) == 2

    def test_reset_turn_clears_tracking(self):
        from ave.agent.verification import VerifiedSession

        session = _make_session()
        vs = VerifiedSession(session)
        vs.call_tool("probe_media", {"path": "/test.mp4"})
        assert len(vs.turn_tools) == 1

        vs.reset_turn()
        assert len(vs.turn_tools) == 0

    def test_verify_turn_returns_none_without_verifier(self):
        from ave.agent.verification import VerifiedSession
        from ave.tools.verify import EditIntent

        session = _make_session()
        vs = VerifiedSession(session, verifier=None)
        vs.call_tool("trim", {"in_ns": 0, "out_ns": 1_000_000_000})

        intent = EditIntent(
            tool_name="trim",
            description="trim first second",
            expected_changes={"duration_seconds": 1.0},
        )
        result = vs.verify_turn(intent, Path("/fake/segment.mp4"))
        assert result is None


# ===========================================================================
# Compositor + RenderScheduler together
# ===========================================================================


class TestCompositorAndSchedulerIntegration:
    """Test compositor strategy with render scheduler."""

    def test_scheduler_with_compositor_selection(self):
        from ave.render.compositor import CompositorStrategy
        from ave.render.parallel import RenderJob, RenderScheduler

        # Select compositor
        selection = CompositorStrategy.select(preference="auto", available=["cpu"])
        assert selection.strategy == "cpu"

        # Create render jobs
        scheduler = RenderScheduler(max_workers=2)
        jobs = [
            RenderJob(segment_id=f"seg_{i}", start_ns=i * 2_000_000_000,
                      stop_ns=(i + 1) * 2_000_000_000, priority=i)
            for i in range(5)
        ]
        scheduler.enqueue(jobs)

        # Get first batch
        batch = scheduler.next_batch()
        assert len(batch) == 2  # max_workers=2
        assert batch[0].segment_id == "seg_0"  # lowest priority number = highest priority

        # Mark first complete
        scheduler.mark_complete("seg_0", Path("/out/seg_0.mp4"))
        assert scheduler.completed_count() == 1

        # Next batch should have 1 more (only 1 slot free)
        batch2 = scheduler.next_batch()
        assert len(batch2) == 1


# ===========================================================================
# modifies_timeline flag integration
# ===========================================================================


class TestModifiesTimelineFlag:
    """Test that modifies_timeline flag is correctly set on real tools."""

    @pytest.fixture()
    def registry(self) -> ToolRegistry:
        session = EditingSession()
        return session.registry

    def test_editing_tools_modify_timeline(self, registry):
        assert registry.tool_modifies_timeline("trim") is True
        assert registry.tool_modifies_timeline("split") is True
        assert registry.tool_modifies_timeline("concatenate") is True

    def test_audio_tools_modify_timeline(self, registry):
        assert registry.tool_modifies_timeline("volume") is True
        assert registry.tool_modifies_timeline("fade") is True
        assert registry.tool_modifies_timeline("normalize") is True

    def test_color_tools_modify_timeline(self, registry):
        assert registry.tool_modifies_timeline("color_grade") is True
        assert registry.tool_modifies_timeline("cdl") is True

    def test_compositing_tools_modify_timeline(self, registry):
        assert registry.tool_modifies_timeline("set_layer_order") is True
        assert registry.tool_modifies_timeline("apply_blend_mode") is True

    def test_read_only_tools_dont_modify(self, registry):
        assert registry.tool_modifies_timeline("probe_media") is False
        assert registry.tool_modifies_timeline("search_transcript") is False
        assert registry.tool_modifies_timeline("lut_parse") is False

    def test_motion_graphics_modify_timeline(self, registry):
        assert registry.tool_modifies_timeline("add_text_overlay") is True
        assert registry.tool_modifies_timeline("add_lower_third") is True


# ===========================================================================
# SDK bridge integration
# ===========================================================================


class TestSDKBridgeIntegration:
    """Test SDK bridge produces valid configuration from real session."""

    def test_create_ave_agent_options_structure(self):
        from ave.agent.sdk_bridge import create_ave_agent_options

        session = EditingSession()
        options = create_ave_agent_options(session)

        assert "agents" in options
        assert "system_prompt" in options
        assert "allowed_tools" in options
        assert isinstance(options["agents"], list)
        assert len(options["agents"]) >= 4  # 4 predefined roles

    def test_role_to_agent_definition_fields(self):
        from ave.agent.roles import EDITOR_ROLE
        from ave.agent.sdk_bridge import role_to_agent_definition

        session = EditingSession()
        defn = role_to_agent_definition(EDITOR_ROLE, session)

        assert "description" in defn
        assert "prompt" in defn
        assert "nanosecond" in defn["prompt"].lower() or "ns" in defn["prompt"].lower()
