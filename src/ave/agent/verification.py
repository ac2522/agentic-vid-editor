"""Verification-aware session wrapper — end-of-turn edit verification.

Wraps EditingSession with deferred verification: tool calls are tracked
per turn and verified together at the end rather than per-call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ave.agent.session import EditingSession
from ave.tools.verify import EditIntent, VerificationResult, VerificationBackend


class VerifiedSession:
    """Wraps EditingSession with end-of-turn verification.

    Verification is deferred to end-of-turn (not per-tool-call) to avoid
    rendering segments after every micro-operation.

    Usage:
        vs = VerifiedSession(session, verifier)
        # Multiple tool calls in a turn...
        vs.call_tool("trim", {...})
        vs.call_tool("color_grade", {...})
        # Verify at end of turn
        result = vs.verify_turn(intent, segment_path)
    """

    def __init__(
        self,
        session: EditingSession,
        verifier: VerificationBackend | None = None,
    ) -> None:
        self._session = session
        self._verifier = verifier
        self._turn_tools: list[str] = []

    @property
    def session(self) -> EditingSession:
        return self._session

    def call_tool(self, tool_name: str, params: dict) -> Any:
        """Execute tool via session, tracking which tools modified timeline."""
        result = self._session.call_tool(tool_name, params)
        self._turn_tools.append(tool_name)
        return result

    def verify_turn(
        self,
        intent: EditIntent,
        segment_path: Path,
    ) -> VerificationResult | None:
        """Verify the current turn's edits against intent.

        Returns None if no verifier configured.
        Resets turn tracking after verification.
        """
        if self._verifier is None:
            return None
        result = self._verifier.verify(intent, segment_path)
        self.reset_turn()
        return result

    def reset_turn(self) -> None:
        """Reset turn tracking without verifying."""
        self._turn_tools.clear()

    @property
    def turn_tools(self) -> list[str]:
        """Tools called in the current turn."""
        return list(self._turn_tools)
