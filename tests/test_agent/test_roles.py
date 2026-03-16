"""Tests for AgentRole definitions and predefined roles."""

from __future__ import annotations

import pytest

from ave.agent.roles import (
    AgentRole,
    ALL_ROLES,
    EDITOR_ROLE,
    COLORIST_ROLE,
    SOUND_DESIGNER_ROLE,
    TRANSCRIPTIONIST_ROLE,
)


# -- test AgentRole is frozen/immutable ----------------------------------------


def test_agent_role_is_frozen():
    """AgentRole instances are immutable (frozen dataclass)."""
    role = AgentRole(
        name="test",
        description="Test role",
        system_prompt="You are a test agent.",
        domains=("testing",),
    )
    with pytest.raises(AttributeError):
        role.name = "changed"


# -- test all predefined roles have required fields ----------------------------


@pytest.mark.parametrize(
    "role",
    [EDITOR_ROLE, COLORIST_ROLE, SOUND_DESIGNER_ROLE, TRANSCRIPTIONIST_ROLE],
    ids=lambda r: r.name,
)
def test_predefined_role_has_required_fields(role: AgentRole):
    """Each predefined role has non-empty name, description, system_prompt, and domains."""
    assert isinstance(role.name, str) and role.name
    assert isinstance(role.description, str) and role.description
    assert isinstance(role.system_prompt, str) and role.system_prompt
    assert isinstance(role.domains, tuple) and len(role.domains) > 0


# -- test role names are unique ------------------------------------------------


def test_role_names_are_unique():
    """All predefined roles have distinct names."""
    names = [r.name for r in ALL_ROLES]
    assert len(names) == len(set(names))


# -- test ALL_ROLES contains all 4 roles --------------------------------------


def test_all_roles_contains_six():
    """ALL_ROLES tuple has all 6 predefined roles."""
    assert len(ALL_ROLES) == 6
    names = {r.name for r in ALL_ROLES}
    assert names == {
        "editor", "colorist", "sound_designer", "transcriptionist",
        "Researcher", "VFX Artist",
    }


# -- test system prompts mention nanosecond conventions ------------------------


@pytest.mark.parametrize(
    "role",
    [EDITOR_ROLE, COLORIST_ROLE, SOUND_DESIGNER_ROLE, TRANSCRIPTIONIST_ROLE],
    ids=lambda r: r.name,
)
def test_system_prompt_mentions_nanoseconds(role: AgentRole):
    """Role system prompts mention nanosecond timestamp conventions."""
    assert "nanosecond" in role.system_prompt.lower()


# -- test system prompts mention agent: metadata prefix ------------------------


def test_editor_prompt_mentions_agent_prefix():
    """Editor system prompt mentions GES metadata agent: prefix."""
    assert "agent:" in EDITOR_ROLE.system_prompt


# -- test editor role domains --------------------------------------------------


def test_editor_role_domains():
    """Editor role covers editing, compositing, motion_graphics, scene."""
    assert set(EDITOR_ROLE.domains) == {"editing", "compositing", "motion_graphics", "scene"}


def test_colorist_role_domains():
    """Colorist role covers color domain."""
    assert COLORIST_ROLE.domains == ("color",)


def test_sound_designer_role_domains():
    """Sound designer covers audio domain."""
    assert SOUND_DESIGNER_ROLE.domains == ("audio",)


def test_transcriptionist_role_domains():
    """Transcriptionist covers transcription domain."""
    assert TRANSCRIPTIONIST_ROLE.domains == ("transcription",)


# -- test ALL_ROLES is a tuple ------------------------------------------------


def test_all_roles_is_tuple():
    """ALL_ROLES is a tuple (immutable collection)."""
    assert isinstance(ALL_ROLES, tuple)
