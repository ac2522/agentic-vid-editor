"""Append-only per-session activity log.

Emitted by EditingSession on every successful tool call. Feeds the
state-sync protocol and harness assertions.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActivityEntry:
    """A single activity record."""

    timestamp: float
    agent_id: str
    tool_name: str
    summary: str
    snapshot_id: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ActivityEntry:
        return cls(
            timestamp=float(d["timestamp"]),
            agent_id=str(d["agent_id"]),
            tool_name=str(d["tool_name"]),
            summary=str(d["summary"]),
            snapshot_id=str(d["snapshot_id"]),
        )


class ActivityLog:
    """Append-only log persisted as JSONL.

    In-memory when persist_path is None; otherwise each append writes
    one JSON line and the constructor loads any existing file.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._persist_path = persist_path
        self._entries: list[ActivityEntry] = []
        if persist_path is not None and persist_path.exists():
            self._entries = [
                ActivityEntry.from_dict(json.loads(line))
                for line in persist_path.read_text().splitlines()
                if line.strip()
            ]

    def append(
        self,
        *,
        agent_id: str,
        tool_name: str,
        summary: str,
        snapshot_id: str,
    ) -> ActivityEntry:
        entry = ActivityEntry(
            timestamp=time.time(),
            agent_id=agent_id,
            tool_name=tool_name,
            summary=summary,
            snapshot_id=snapshot_id,
        )
        self._entries.append(entry)
        if self._persist_path is not None:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self._persist_path.open("a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        return entry

    def entries(self) -> list[ActivityEntry]:
        return list(self._entries)

    def entries_since(self, timestamp: float) -> list[ActivityEntry]:
        return [e for e in self._entries if e.timestamp > timestamp]
