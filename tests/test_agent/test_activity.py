"""Tests for the ActivityLog append-only log."""

import json
from pathlib import Path

from ave.agent.activity import ActivityEntry, ActivityLog


def test_entry_roundtrip_dict():
    """ActivityEntry serializes and deserializes cleanly."""
    e = ActivityEntry(
        timestamp=1714000000.0,
        agent_id="sound_designer",
        tool_name="set_volume",
        summary="set_volume(clip_id='c1', db=-3.0)",
        snapshot_id="snap-1",
    )
    d = e.to_dict()
    assert d["agent_id"] == "sound_designer"
    assert ActivityEntry.from_dict(d) == e


def test_append_in_memory(tmp_path: Path):
    log = ActivityLog(persist_path=None)
    log.append(
        agent_id="editor",
        tool_name="trim_clip",
        summary="trim_clip(clip='c1', in=1.0, out=4.0)",
        snapshot_id="snap-a",
    )
    log.append(
        agent_id="colorist",
        tool_name="apply_lut",
        summary="apply_lut(clip='c1', lut='FilmLook.cube')",
        snapshot_id="snap-b",
    )
    entries = log.entries()
    assert len(entries) == 2
    assert entries[0].tool_name == "trim_clip"
    assert entries[1].agent_id == "colorist"


def test_append_persisted(tmp_path: Path):
    """Each append writes one JSON line to the persist path."""
    persist = tmp_path / "activity-log.jsonl"
    log = ActivityLog(persist_path=persist)
    log.append(agent_id="editor", tool_name="trim_clip", summary="t", snapshot_id="s1")
    log.append(agent_id="editor", tool_name="split_clip", summary="s", snapshot_id="s2")

    raw = persist.read_text().splitlines()
    assert len(raw) == 2
    assert json.loads(raw[0])["tool_name"] == "trim_clip"
    assert json.loads(raw[1])["tool_name"] == "split_clip"


def test_load_from_persisted_file(tmp_path: Path):
    """Opening an existing log reads prior entries."""
    persist = tmp_path / "activity-log.jsonl"
    log1 = ActivityLog(persist_path=persist)
    log1.append(agent_id="editor", tool_name="trim_clip", summary="t", snapshot_id="s1")

    log2 = ActivityLog(persist_path=persist)
    entries = log2.entries()
    assert len(entries) == 1
    assert entries[0].tool_name == "trim_clip"


def test_entries_since(tmp_path: Path):
    """entries_since(timestamp) returns only entries after the given time."""
    log = ActivityLog(persist_path=None)
    log.append(agent_id="a", tool_name="t1", summary="", snapshot_id="s1")
    cutoff = log.entries()[-1].timestamp
    log.append(agent_id="a", tool_name="t2", summary="", snapshot_id="s2")

    after = log.entries_since(cutoff)
    assert len(after) == 1
    assert after[0].tool_name == "t2"
