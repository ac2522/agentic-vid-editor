"""Scope-enforcement tests for EditingSession.call_tool."""

import pytest

from ave.agent.domains import Domain
from ave.agent.errors import ScopeViolationError
from ave.agent.roles import SOUND_DESIGNER_ROLE
from ave.agent.session import EditingSession


def _register_audio_and_video_tool(session: EditingSession) -> None:
    """Register one tool in AUDIO and one in VIDEO for scope testing."""

    def set_volume(clip_id: str, db: float) -> dict:
        """Set clip volume."""
        return {"clip_id": clip_id, "db": db}

    def add_watermark(clip_id: str) -> dict:
        """Add a watermark."""
        return {"clip_id": clip_id}

    session.registry.register(
        "set_volume",
        set_volume,
        domain="audio",
        domains_touched=(Domain.AUDIO,),
    )
    session.registry.register(
        "add_watermark",
        add_watermark,
        domain="video",
        domains_touched=(Domain.VIDEO,),
    )


def test_call_tool_without_role_bypasses_scope():
    """No role passed = no scope check (backward compat)."""
    session = EditingSession()
    _register_audio_and_video_tool(session)

    # Both should succeed without role
    assert session.call_tool("set_volume", {"clip_id": "c1", "db": -3.0})
    assert session.call_tool("add_watermark", {"clip_id": "c1"})


def test_call_tool_with_role_rejects_out_of_domain():
    """Sound designer may not call a VIDEO tool."""
    session = EditingSession()
    _register_audio_and_video_tool(session)

    with pytest.raises(ScopeViolationError, match="add_watermark"):
        session.call_tool(
            "add_watermark",
            {"clip_id": "c1"},
            agent_role=SOUND_DESIGNER_ROLE,
        )


def test_call_tool_with_role_allows_in_domain():
    """Sound designer may call an AUDIO tool."""
    session = EditingSession()
    _register_audio_and_video_tool(session)

    result = session.call_tool(
        "set_volume",
        {"clip_id": "c1", "db": -3.0},
        agent_role=SOUND_DESIGNER_ROLE,
    )
    assert result["db"] == -3.0


def test_scope_violation_contains_role_and_tool_and_domain():
    """Error message is actionable."""
    session = EditingSession()
    _register_audio_and_video_tool(session)

    with pytest.raises(ScopeViolationError) as excinfo:
        session.call_tool(
            "add_watermark",
            {"clip_id": "c1"},
            agent_role=SOUND_DESIGNER_ROLE,
        )

    msg = str(excinfo.value)
    assert "sound_designer" in msg.lower()
    assert "add_watermark" in msg
    assert "video" in msg.lower()
