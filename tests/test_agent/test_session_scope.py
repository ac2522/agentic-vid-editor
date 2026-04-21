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


def test_successful_tool_call_appends_activity_entry(tmp_path):
    """Every successful call adds one entry to the session's activity log."""
    from ave.agent.activity import ActivityLog

    log = ActivityLog(persist_path=tmp_path / "activity-log.jsonl")
    session = EditingSession(activity_log=log)
    _register_audio_and_video_tool(session)

    session.call_tool("set_volume", {"clip_id": "c1", "db": -3.0})
    session.call_tool("add_watermark", {"clip_id": "c1"})

    entries = log.entries()
    assert len(entries) == 2
    assert entries[0].tool_name == "set_volume"
    assert entries[1].tool_name == "add_watermark"


def test_failed_tool_call_does_not_append_activity_entry(tmp_path):
    """Exceptions abort the call — no activity entry is recorded."""
    from ave.agent.activity import ActivityLog

    log = ActivityLog(persist_path=tmp_path / "activity-log.jsonl")
    session = EditingSession(activity_log=log)

    def always_fails(clip_id: str) -> None:
        """Always fails."""
        raise RuntimeError("boom")

    session.registry.register(
        "always_fails",
        always_fails,
        domain="audio",
        domains_touched=(Domain.AUDIO,),
    )

    with pytest.raises(RuntimeError, match="boom"):
        session.call_tool("always_fails", {"clip_id": "c1"})

    assert log.entries() == []


def test_activity_entry_uses_role_name_when_provided(tmp_path):
    """Activity entries carry the role name as agent_id when a role is passed."""
    from ave.agent.activity import ActivityLog
    from ave.agent.roles import SOUND_DESIGNER_ROLE

    log = ActivityLog(persist_path=None)
    session = EditingSession(activity_log=log)
    _register_audio_and_video_tool(session)

    session.call_tool(
        "set_volume",
        {"clip_id": "c1", "db": -3.0},
        agent_role=SOUND_DESIGNER_ROLE,
    )
    entries = log.entries()
    assert entries[0].agent_id == "sound_designer"


def test_activity_entry_uses_anonymous_when_no_role(tmp_path):
    from ave.agent.activity import ActivityLog

    log = ActivityLog(persist_path=None)
    session = EditingSession(activity_log=log)
    _register_audio_and_video_tool(session)

    session.call_tool("set_volume", {"clip_id": "c1", "db": -3.0})
    assert log.entries()[0].agent_id == "anonymous"
