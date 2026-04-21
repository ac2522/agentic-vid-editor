"""Tests for the Domain enum."""

from ave.agent.domains import Domain


def test_domain_enum_members():
    """All required domain members exist."""
    assert Domain.AUDIO.value == "audio"
    assert Domain.VIDEO.value == "video"
    assert Domain.SUBTITLE.value == "subtitle"
    assert Domain.VFX_MASK.value == "vfx_mask"
    assert Domain.COLOR.value == "color"
    assert Domain.TIMELINE_STRUCTURE.value == "timeline_structure"
    assert Domain.METADATA.value == "metadata"
    assert Domain.RENDER.value == "render"
    assert Domain.INGEST.value == "ingest"
    assert Domain.RESEARCH.value == "research"


def test_domain_from_string():
    """Domain.from_string maps legacy domain names to enum members."""
    assert Domain.from_string("audio") is Domain.AUDIO
    assert Domain.from_string("editing") is Domain.TIMELINE_STRUCTURE
    assert Domain.from_string("compositing") is Domain.VIDEO
    assert Domain.from_string("motion_graphics") is Domain.SUBTITLE
    assert Domain.from_string("scene") is Domain.TIMELINE_STRUCTURE
    assert Domain.from_string("transcription") is Domain.SUBTITLE
    assert Domain.from_string("vfx") is Domain.VFX_MASK
    assert Domain.from_string("color") is Domain.COLOR
    assert Domain.from_string("research") is Domain.RESEARCH


def test_domain_from_string_unknown_raises():
    """Unknown domain strings raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Unknown domain"):
        Domain.from_string("nonexistent")
