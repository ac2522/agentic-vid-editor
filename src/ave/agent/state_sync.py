"""State-sync protocol — compact summaries injected into agent turns.

The summary is a concise text block that tells the agent what the current
timeline state is and what has happened since its last turn. It's prepended
to the agent's user-message payload at turn start.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ave.agent.activity import ActivityEntry, ActivityLog

if TYPE_CHECKING:
    from ave.agent.session import EditingSession


@dataclass(frozen=True)
class StateSummary:
    """Compact state summary for agent context injection."""

    generated_at: float
    state_provisions: tuple[str, ...]
    recent_entries: tuple[ActivityEntry, ...] = field(default_factory=tuple)

    def render(self) -> str:
        """Render as a compact text block (~300 token target)."""
        ts = datetime.fromtimestamp(self.generated_at, tz=timezone.utc).strftime("%H:%M:%S")
        lines = [f"STATE SUMMARY (as of {ts} UTC):"]
        provisions = ", ".join(sorted(self.state_provisions)) or "none"
        lines.append(f"  Session provisions: {provisions}")
        if self.recent_entries:
            lines.append("  Activity since your last turn:")
            for e in self.recent_entries:
                etime = datetime.fromtimestamp(e.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
                lines.append(f"    - {etime} ({e.agent_id}) {e.tool_name}: {e.summary}")
        else:
            lines.append("  No activity since your last turn.")
        lines.append("")
        lines.append("For full detail, call get_project_state.")
        return "\n".join(lines)


def build_state_summary(
    *,
    session: EditingSession,
    activity_log: ActivityLog,
    since_timestamp: float,
) -> StateSummary:
    """Construct a state summary for the next agent turn."""
    return StateSummary(
        generated_at=time.time(),
        state_provisions=tuple(sorted(session.state.provisions)),
        recent_entries=tuple(activity_log.entries_since(since_timestamp)),
    )
