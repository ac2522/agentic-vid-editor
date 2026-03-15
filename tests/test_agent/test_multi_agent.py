"""Tests for MultiAgentOrchestrator — role-based multi-agent coordination."""

from __future__ import annotations

import threading

import pytest

from ave.agent.registry import ToolRegistry
from ave.agent.dependencies import SessionState
from ave.agent.session import EditingSession
from ave.agent.orchestrator import Orchestrator
from ave.agent.roles import (
    AgentRole,
    ALL_ROLES,
    EDITOR_ROLE,
    COLORIST_ROLE,
    SOUND_DESIGNER_ROLE,
    TRANSCRIPTIONIST_ROLE,
)
from ave.agent.multi_agent import MultiAgentOrchestrator
from ave.agent.sdk_bridge import create_ave_agent_options, role_to_agent_definition


# -- fixtures ------------------------------------------------------------------


@pytest.fixture
def session() -> EditingSession:
    """Create a session with test tools across multiple domains."""
    s = EditingSession.__new__(EditingSession)
    s._registry = ToolRegistry()
    s._state = SessionState()
    s._history = []
    s._project_path = None
    s._snapshot_manager = None
    s._transition_graph = None
    s._lock = threading.Lock()

    @s._registry.tool(domain="editing", requires=[], provides=["clip_trimmed"])
    def trim_clip(start_ns: int, end_ns: int) -> dict:
        """Trim a clip to the given time range."""
        return {"start": start_ns, "end": end_ns}

    @s._registry.tool(domain="editing", requires=[], provides=[])
    def split_clip(position_ns: int) -> dict:
        """Split a clip at the given position."""
        return {"position": position_ns}

    @s._registry.tool(domain="color", requires=[], provides=["grade_applied"])
    def apply_lut(lut_path: str) -> dict:
        """Apply a LUT for color grading."""
        return {"lut": lut_path}

    @s._registry.tool(domain="audio", requires=[], provides=["audio_adjusted"])
    def set_volume(level_db: float) -> dict:
        """Set audio volume level."""
        return {"level": level_db}

    @s._registry.tool(domain="transcription", requires=[], provides=["transcript_ready"])
    def transcribe(media_path: str) -> dict:
        """Transcribe audio to text."""
        return {"path": media_path}

    @s._registry.tool(domain="compositing", requires=[], provides=[])
    def composite_layers(layers: int) -> dict:
        """Composite multiple layers."""
        return {"layers": layers}

    @s._registry.tool(domain="scene", requires=[], provides=[])
    def detect_scenes(threshold: float) -> dict:
        """Detect scene boundaries."""
        return {"threshold": threshold}

    return s


@pytest.fixture
def multi_orch(session: EditingSession) -> MultiAgentOrchestrator:
    """MultiAgentOrchestrator with default roles."""
    return MultiAgentOrchestrator(session)


# -- test MultiAgentOrchestrator creation with default roles -------------------


def test_creates_with_default_roles(multi_orch: MultiAgentOrchestrator):
    """MultiAgentOrchestrator initializes with all 4 default roles."""
    assert multi_orch.roles == ALL_ROLES
    assert len(multi_orch.roles) == 4


# -- test MultiAgentOrchestrator creation with custom roles --------------------


def test_creates_with_custom_roles(session: EditingSession):
    """MultiAgentOrchestrator accepts a custom subset of roles."""
    custom_roles = [EDITOR_ROLE, COLORIST_ROLE]
    orch = MultiAgentOrchestrator(session, roles=custom_roles)
    assert len(orch.roles) == 2
    assert orch.roles == (EDITOR_ROLE, COLORIST_ROLE)


# -- test session property -----------------------------------------------------


def test_session_property(multi_orch: MultiAgentOrchestrator, session: EditingSession):
    """Session is accessible via property."""
    assert multi_orch.session is session


# -- test base orchestrator is accessible --------------------------------------


def test_base_orchestrator_accessible(multi_orch: MultiAgentOrchestrator):
    """Base Orchestrator is accessible and is an Orchestrator instance."""
    assert isinstance(multi_orch.base_orchestrator, Orchestrator)
    assert multi_orch.base_orchestrator.session is multi_orch.session


# -- test get_agent_definitions returns dicts with correct keys ----------------


def test_get_agent_definitions_keys(multi_orch: MultiAgentOrchestrator):
    """get_agent_definitions returns dict with description and prompt for each role."""
    defs = multi_orch.get_agent_definitions()
    assert isinstance(defs, dict)
    assert len(defs) == 4

    for role_name, defn in defs.items():
        assert "description" in defn, f"Missing 'description' for {role_name}"
        assert "prompt" in defn, f"Missing 'prompt' for {role_name}"
        assert isinstance(defn["description"], str)
        assert isinstance(defn["prompt"], str)
        assert len(defn["description"]) > 0
        assert len(defn["prompt"]) > 0


def test_get_agent_definitions_role_names(multi_orch: MultiAgentOrchestrator):
    """Agent definition keys match role names."""
    defs = multi_orch.get_agent_definitions()
    assert set(defs.keys()) == {r.name for r in ALL_ROLES}


# -- test get_system_prompt includes role descriptions -------------------------


def test_get_system_prompt_includes_roles(multi_orch: MultiAgentOrchestrator):
    """System prompt mentions each role's name and description."""
    prompt = multi_orch.get_system_prompt()
    for role in ALL_ROLES:
        assert role.name in prompt, f"Role '{role.name}' not in system prompt"


def test_get_system_prompt_mentions_workflow(multi_orch: MultiAgentOrchestrator):
    """System prompt mentions the meta-tool workflow."""
    prompt = multi_orch.get_system_prompt()
    assert "search_tools" in prompt
    assert "call_tool" in prompt


# -- test get_role_tools returns tools from role's domains ---------------------


def test_get_role_tools_editor(multi_orch: MultiAgentOrchestrator):
    """Editor role gets tools from editing, compositing, motion_graphics, scene domains."""
    tools = multi_orch.get_role_tools(EDITOR_ROLE)
    assert "trim_clip" in tools
    assert "split_clip" in tools
    assert "composite_layers" in tools
    assert "detect_scenes" in tools
    # Should NOT include other domains
    assert "apply_lut" not in tools
    assert "set_volume" not in tools
    assert "transcribe" not in tools


def test_get_role_tools_colorist(multi_orch: MultiAgentOrchestrator):
    """Colorist role gets only color domain tools."""
    tools = multi_orch.get_role_tools(COLORIST_ROLE)
    assert "apply_lut" in tools
    assert len(tools) == 1


def test_get_role_tools_sound_designer(multi_orch: MultiAgentOrchestrator):
    """Sound designer gets only audio domain tools."""
    tools = multi_orch.get_role_tools(SOUND_DESIGNER_ROLE)
    assert "set_volume" in tools
    assert len(tools) == 1


def test_get_role_tools_transcriptionist(multi_orch: MultiAgentOrchestrator):
    """Transcriptionist gets only transcription domain tools."""
    tools = multi_orch.get_role_tools(TRANSCRIPTIONIST_ROLE)
    assert "transcribe" in tools
    assert len(tools) == 1


# -- test get_role_for_domain --------------------------------------------------


def test_get_role_for_domain_editing(multi_orch: MultiAgentOrchestrator):
    """Finds editor role for 'editing' domain."""
    role = multi_orch.get_role_for_domain("editing")
    assert role is EDITOR_ROLE


def test_get_role_for_domain_color(multi_orch: MultiAgentOrchestrator):
    """Finds colorist role for 'color' domain."""
    role = multi_orch.get_role_for_domain("color")
    assert role is COLORIST_ROLE


def test_get_role_for_domain_audio(multi_orch: MultiAgentOrchestrator):
    """Finds sound_designer role for 'audio' domain."""
    role = multi_orch.get_role_for_domain("audio")
    assert role is SOUND_DESIGNER_ROLE


def test_get_role_for_domain_none_for_unknown(multi_orch: MultiAgentOrchestrator):
    """Returns None for domains not covered by any role."""
    role = multi_orch.get_role_for_domain("nonexistent_domain")
    assert role is None


# -- test sdk_bridge functions -------------------------------------------------


def test_role_to_agent_definition(session: EditingSession):
    """role_to_agent_definition returns dict with description, prompt keys."""
    defn = role_to_agent_definition(EDITOR_ROLE, session)
    assert isinstance(defn, dict)
    assert "description" in defn
    assert "prompt" in defn
    assert EDITOR_ROLE.description in defn["description"]


def test_create_ave_agent_options(session: EditingSession):
    """create_ave_agent_options returns dict with agents, system_prompt, allowed_tools."""
    opts = create_ave_agent_options(session)
    assert isinstance(opts, dict)
    assert "agents" in opts
    assert "system_prompt" in opts
    assert "allowed_tools" in opts
    assert isinstance(opts["agents"], list)
    assert len(opts["agents"]) == 4  # one per default role


def test_create_ave_agent_options_custom_roles(session: EditingSession):
    """create_ave_agent_options with custom roles limits agents."""
    opts = create_ave_agent_options(session, roles=[EDITOR_ROLE])
    assert len(opts["agents"]) == 1


def test_create_ave_agent_options_model(session: EditingSession):
    """create_ave_agent_options passes model through."""
    opts = create_ave_agent_options(session, model="sonnet")
    # Model should appear in agent definitions
    for agent in opts["agents"]:
        assert agent.get("model") == "sonnet"
