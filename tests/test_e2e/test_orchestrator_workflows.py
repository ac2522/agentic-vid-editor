"""E2E orchestrator workflow tests — real LLM-driven tool selection.

Each test feeds a natural language video editing request through the
Orchestrator with a real Claude API call, then verifies the correct
tools were discovered and called.  Scenarios come in amateur/professional
pairs: same intent, different wording, both must produce valid results.

Requirements:
    - ANTHROPIC_API_KEY in environment (or .env)
    - Tests are skipped without it
    - Uses claude-haiku-4-5-20251001 to minimise cost
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ave.agent.orchestrator import Orchestrator
from ave.agent.session import EditingSession

# ---------------------------------------------------------------------------
# Skip marker — requires Anthropic API key
# ---------------------------------------------------------------------------

def _load_env():
    """Load .env file if present."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

_load_env()

def _anthropic_available() -> bool:
    try:
        import anthropic  # noqa: F401
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    except ImportError:
        return False

requires_anthropic = pytest.mark.skipif(
    not _anthropic_available(), reason="Anthropic API key not set or SDK missing"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEC = 1_000_000_000  # nanoseconds
MODEL = "claude-haiku-4-5-20251001"
MAX_TURNS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xges(tmp_path: Path) -> Path:
    """Create a minimal .xges file."""
    p = tmp_path / "project.xges"
    p.write_text("<ges version='0.7'></ges>")
    return p


def _make_transcript_json() -> str:
    """Realistic transcript with filler words."""
    transcript = {
        "language": "en",
        "duration": 30.0,
        "segments": [
            {
                "start": 0.0,
                "end": 10.0,
                "text": "So um today we're going to um talk about like video editing",
                "words": [
                    {"word": "So", "start": 0.0, "end": 0.2},
                    {"word": "um", "start": 0.3, "end": 0.5},
                    {"word": "today", "start": 0.6, "end": 1.0},
                    {"word": "we're", "start": 1.1, "end": 1.3},
                    {"word": "going", "start": 1.4, "end": 1.6},
                    {"word": "to", "start": 1.7, "end": 1.8},
                    {"word": "um", "start": 1.9, "end": 2.1},
                    {"word": "talk", "start": 2.2, "end": 2.5},
                    {"word": "about", "start": 2.6, "end": 2.9},
                    {"word": "like", "start": 3.0, "end": 3.2},
                    {"word": "video", "start": 3.3, "end": 3.7},
                    {"word": "editing", "start": 3.8, "end": 4.2},
                ],
            },
            {
                "start": 10.0,
                "end": 20.0,
                "text": "It's uh really important to uh get the pacing right",
                "words": [
                    {"word": "It's", "start": 10.0, "end": 10.3},
                    {"word": "uh", "start": 10.4, "end": 10.6},
                    {"word": "really", "start": 10.7, "end": 11.0},
                    {"word": "important", "start": 11.1, "end": 11.6},
                    {"word": "to", "start": 11.7, "end": 11.8},
                    {"word": "uh", "start": 11.9, "end": 12.1},
                    {"word": "get", "start": 12.2, "end": 12.4},
                    {"word": "the", "start": 12.5, "end": 12.6},
                    {"word": "pacing", "start": 12.7, "end": 13.1},
                    {"word": "right", "start": 13.2, "end": 13.5},
                ],
            },
        ],
    }
    return json.dumps(transcript)


def _make_scene_boundaries_json() -> str:
    """Realistic scene boundary data for rough-cut tests."""
    scenes = [
        {"start_ns": 0, "end_ns": 5 * SEC, "start_timecode": "00:00:00:00", "end_timecode": "00:00:05:00"},
        {"start_ns": 5 * SEC, "end_ns": 12 * SEC, "start_timecode": "00:00:05:00", "end_timecode": "00:00:12:00"},
        {"start_ns": 12 * SEC, "end_ns": 18 * SEC, "start_timecode": "00:00:12:00", "end_timecode": "00:00:18:00"},
        {"start_ns": 18 * SEC, "end_ns": 25 * SEC, "start_timecode": "00:00:18:00", "end_timecode": "00:00:25:00"},
        {"start_ns": 25 * SEC, "end_ns": 30 * SEC, "start_timecode": "00:00:25:00", "end_timecode": "00:00:30:00"},
    ]
    return json.dumps(scenes)


def _session_with_project(tmp_path: Path, extra_state: list[str] | None = None) -> EditingSession:
    """Create a session with a loaded project and optional extra state."""
    session = EditingSession()
    session.load_project(_make_xges(tmp_path))
    session.state.add("clip_exists")
    if extra_state:
        for s in extra_state:
            session.state.add(s)
    return session


def _meta_tool_to_api(mt) -> dict:
    """Convert MetaToolDef to Anthropic API tool format."""
    return {
        "name": mt.name,
        "description": mt.description,
        "input_schema": mt.parameters,
    }


def _run_agent_loop(
    orchestrator: Orchestrator,
    user_prompt: str,
    *,
    max_turns: int = MAX_TURNS,
    context: str = "",
) -> tuple[list, object]:
    """Drive a Claude tool-use loop through the orchestrator.

    Returns (history, state) from the session after the loop completes.
    """
    import anthropic

    client = anthropic.Anthropic()
    tools = [_meta_tool_to_api(mt) for mt in orchestrator.get_meta_tools()]

    system_prompt = (
        orchestrator.get_system_prompt()
        + "\n\nIMPORTANT: You MUST use the tools to accomplish the user's request. "
        "Always start by calling search_tools to find relevant tools, then "
        "get_tool_schema to see parameters, then call_tool to execute. "
        "Do NOT just respond with text — you must actually call the tools. "
        "NEVER ask clarifying questions — make reasonable assumptions and act immediately."
    )
    if context:
        system_prompt += f"\n\nAdditional context:\n{context}"

    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    for turn in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=system_prompt,
            tools=tools,
            messages=messages,
            temperature=0,
        )

        if response.stop_reason == "end_turn":
            break

        # Append assistant message
        messages.append({"role": "assistant", "content": response.content})

        # Process tool uses
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = orchestrator.handle_tool_call(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if not tool_results:
            break

        messages.append({"role": "user", "content": tool_results})

    return orchestrator.session.history, orchestrator.session.state


def _tool_names(history) -> list[str]:
    """Extract tool names from session history."""
    return [h.tool_name for h in history]


# ---------------------------------------------------------------------------
# 1. Instagram Reel Export
# ---------------------------------------------------------------------------


@requires_anthropic
@pytest.mark.slow
@pytest.mark.llm
class TestInstagramExport:
    """User wants to take a portion of 4K footage and export for Instagram."""

    def test_amateur_instagram_reel(self, tmp_path):
        """Amateur: vague request about making a vertical Instagram video."""
        session = _session_with_project(tmp_path)
        orchestrator = Orchestrator(session)

        history, state = _run_agent_loop(
            orchestrator,
            "I have a 4K video that's about 2 minutes long. I want to take just "
            "the last 12 seconds and turn it into a vertical video for Instagram. "
            "The clip duration is 120 seconds, so 120000000000 nanoseconds.",
            context=(
                "The clip is 120 seconds (120000000000 ns). "
                "The instagram_reel preset exists for vertical 1080x1920 output. "
                "Use the tools available — search, get schema, then call them."
            ),
        )

        names = _tool_names(history)
        # Must have trimmed the clip
        assert "trim" in names, f"Expected 'trim' in tool calls, got: {names}"
        assert state.has("clip_trimmed")

    def test_professional_instagram_reel(self, tmp_path):
        """Professional: precise technical spec for Instagram export."""
        session = _session_with_project(tmp_path)
        orchestrator = Orchestrator(session)

        history, state = _run_agent_loop(
            orchestrator,
            "Extract the tail 12-second segment from a 120-second clip "
            "(clip_duration_ns=120000000000, in_ns=108000000000, out_ns=120000000000). "
            "Then render with the instagram_reel preset.",
            context=(
                "The xges_path is 'project.xges' and output_path is 'reel.mp4'. "
                "Use the tools available — search, get schema, then call them."
            ),
        )

        names = _tool_names(history)
        assert "trim" in names, f"Expected 'trim' in tool calls, got: {names}"
        assert state.has("clip_trimmed")


# ---------------------------------------------------------------------------
# 2. Color Correction
# ---------------------------------------------------------------------------


@requires_anthropic
@pytest.mark.slow
@pytest.mark.llm
class TestColorCorrection:
    """User wants to fix exposure/color issues on a clip."""

    def test_amateur_color_fix(self, tmp_path):
        """Amateur: face too dark, background too bright."""
        session = _session_with_project(tmp_path)
        orchestrator = Orchestrator(session)

        history, state = _run_agent_loop(
            orchestrator,
            "The person's face in this video is way too dark and the background "
            "is really bright. I just want to be able to see their face better. "
            "Can you fix the lighting?",
            context=(
                "Use the color_grade tool to adjust lift/gamma/gain. "
                "Lifting shadows (lift > 0) brightens dark areas. "
                "Reducing gain (< 1.0) tames highlights. "
                "Use the tools available — search, get schema, then call them."
            ),
        )

        names = _tool_names(history)
        assert "color_grade" in names, f"Expected 'color_grade' in tool calls, got: {names}"
        assert state.has("color_graded")

    def test_professional_color_grade(self, tmp_path):
        """Professional: exact lift/gamma/gain values specified."""
        session = _session_with_project(tmp_path)
        orchestrator = Orchestrator(session)

        history, state = _run_agent_loop(
            orchestrator,
            "Apply color grade: lift RGB all 0.15, gamma RGB 1.0/0.95/1.0, "
            "gain RGB all 0.85, saturation 1.1. "
            "Use the color_grade tool with these exact values.",
            context="Use the tools available — search, get schema, then call them.",
        )

        names = _tool_names(history)
        assert "color_grade" in names, f"Expected 'color_grade' in tool calls, got: {names}"
        assert state.has("color_graded")

        # Verify the professional got their exact values applied
        color_call = next(h for h in history if h.tool_name == "color_grade")
        assert color_call.params["lift_r"] == pytest.approx(0.15, abs=0.01)
        assert color_call.params["gamma_g"] == pytest.approx(0.95, abs=0.01)
        assert color_call.params["gain_r"] == pytest.approx(0.85, abs=0.01)
        assert color_call.params["saturation"] == pytest.approx(1.1, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Transcript Cleanup
# ---------------------------------------------------------------------------


@requires_anthropic
@pytest.mark.slow
@pytest.mark.llm
class TestTranscriptCleanup:
    """User wants to remove filler words from a talking-head video."""

    def test_amateur_remove_ums(self, tmp_path):
        """Amateur: 'remove the ums and uhs'."""
        transcript_json = _make_transcript_json()
        session = _session_with_project(tmp_path, extra_state=["transcript_loaded"])
        orchestrator = Orchestrator(session)

        history, state = _run_agent_loop(
            orchestrator,
            "There's a lot of ums and uhs in this video. Can you clean those "
            "up so it sounds more professional?",
            context=(
                f"The transcript JSON is: {transcript_json}\n"
                "Pass this as the transcript_json parameter to transcription tools. "
                "Use the tools available — search, get schema, then call them."
            ),
        )

        names = _tool_names(history)
        assert "find_fillers" in names, f"Expected 'find_fillers' in tool calls, got: {names}"
        assert state.has("fillers_found")

    def test_professional_filler_cut(self, tmp_path):
        """Professional: precise filler-word pipeline."""
        transcript_json = _make_transcript_json()
        session = _session_with_project(tmp_path, extra_state=["transcript_loaded"])
        orchestrator = Orchestrator(session)

        history, state = _run_agent_loop(
            orchestrator,
            "Find all filler words (um, uh, like) in the transcript, then compute "
            "text cuts to remove them. Here is the transcript JSON to use for both "
            f"tool calls: {transcript_json}",
            context="Use the tools available — search, get schema, then call them.",
        )

        names = _tool_names(history)
        assert "find_fillers" in names, f"Expected 'find_fillers' in tool calls, got: {names}"
        assert state.has("fillers_found")


# ---------------------------------------------------------------------------
# 4. Highlight Reel / Rough Cut
# ---------------------------------------------------------------------------


@requires_anthropic
@pytest.mark.slow
@pytest.mark.llm
class TestHighlightReel:
    """User wants to assemble a highlight reel from detected scenes."""

    def test_amateur_best_moments(self, tmp_path):
        """Amateur: 'pick the best parts and make a short clip'."""
        scenes_json = _make_scene_boundaries_json()
        session = _session_with_project(tmp_path, extra_state=["scenes_detected"])
        orchestrator = Orchestrator(session)

        history, state = _run_agent_loop(
            orchestrator,
            "I have a 30-second video with 5 detected scenes. Can you pick a few "
            "of the best scenes and assemble them into a highlight clip?",
            context=(
                f"Scenes have already been detected. The scene boundaries JSON is: {scenes_json}\n"
                "Use the create_rough_cut tool with scenes_json and selected_indices (as JSON string). "
                "Use the tools available — search, get schema, then call them."
            ),
        )

        names = _tool_names(history)
        assert "create_rough_cut" in names, f"Expected 'create_rough_cut' in tool calls, got: {names}"
        assert state.has("rough_cut_created")

    def test_professional_rough_cut(self, tmp_path):
        """Professional: exact scene indices and parameters."""
        scenes_json = _make_scene_boundaries_json()
        session = _session_with_project(tmp_path, extra_state=["scenes_detected"])
        orchestrator = Orchestrator(session)

        history, state = _run_agent_loop(
            orchestrator,
            "Use the create_rough_cut tool to assemble selected scenes. "
            f"Pass scenes_json: {scenes_json} "
            'and selected_indices: "[0, 2, 4]" (as a JSON string). '
            "Set order to chronological, gap_ns to 500000000.",
            context="Use the tools available — search for 'rough cut', get schema, then call it.",
        )

        names = _tool_names(history)
        assert "create_rough_cut" in names, f"Expected 'create_rough_cut' in tool calls, got: {names}"
        assert state.has("rough_cut_created")

        # Verify the professional's exact parameters were used
        cut_call = next(h for h in history if h.tool_name == "create_rough_cut")
        assert cut_call.params["gap_ns"] == 500_000_000


# ---------------------------------------------------------------------------
# 5. Audio Mixing
# ---------------------------------------------------------------------------


@requires_anthropic
@pytest.mark.slow
@pytest.mark.llm
class TestAudioMixing:
    """User wants to fix audio balance between voice and music."""

    def test_amateur_fix_loud_music(self, tmp_path):
        """Amateur: 'music is too loud, can't hear the person'."""
        session = _session_with_project(tmp_path)
        orchestrator = Orchestrator(session)

        history, state = _run_agent_loop(
            orchestrator,
            "The background music in this video is way too loud. I can barely "
            "hear the person talking. Please turn the volume down on the music. "
            "Make it like -18dB or something so I can hear the speaker.",
            context=(
                "Use the volume tool (level_db parameter) to adjust the audio level. "
                "Use the tools available — search, get schema, then call them."
            ),
        )

        names = _tool_names(history)
        audio_tools = {"volume", "normalize", "fade"}
        assert any(t in audio_tools for t in names), (
            f"Expected at least one audio tool {audio_tools} in calls, got: {names}"
        )
        audio_states = {"volume_set", "audio_normalized", "fade_applied"}
        assert any(state.has(s) for s in audio_states), (
            f"Expected at least one audio state {audio_states}"
        )

    def test_professional_audio_normalize(self, tmp_path):
        """Professional: exact normalization and volume targets."""
        session = _session_with_project(tmp_path)
        orchestrator = Orchestrator(session)

        history, state = _run_agent_loop(
            orchestrator,
            "Normalize the dialogue track: current peak is -6dB, target peak -14dB. "
            "Then set the music track volume to -18dB.",
            context="Use the tools available — search, get schema, then call them.",
        )

        names = _tool_names(history)
        assert "normalize" in names, f"Expected 'normalize' in tool calls, got: {names}"
        assert state.has("audio_normalized")

        # Verify the exact normalization values
        norm_call = next(h for h in history if h.tool_name == "normalize")
        assert norm_call.params["current_peak_db"] == pytest.approx(-6.0, abs=0.1)
        assert norm_call.params["target_peak_db"] == pytest.approx(-14.0, abs=0.1)
