"""Pure WebSocket chat protocol functions for the AVE web UI.

These functions handle message serialisation/deserialisation with no framework
dependencies — they transform between raw JSON strings and Python dicts.
"""

from __future__ import annotations

import json


def parse_client_message(raw: str) -> dict:
    """Parse a raw JSON string from the client.

    Returns the parsed dict on success, or ``{"type": "error", "message": "..."}``
    on invalid JSON or missing ``type`` field.
    """
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return {"type": "error", "message": f"Invalid JSON: {exc}"}

    if not isinstance(msg, dict) or "type" not in msg:
        return {"type": "error", "message": "Missing required 'type' field"}

    return msg


def format_text_delta(text: str) -> dict:
    """Format a streaming text chunk for the client."""
    return {"type": "text_delta", "text": text}


def format_tool_start(tool_name: str, tool_id: str) -> dict:
    """Format a tool-execution-started event."""
    return {"type": "tool_start", "tool_name": tool_name, "tool_id": tool_id}


def format_tool_done(tool_id: str) -> dict:
    """Format a tool-execution-completed event."""
    return {"type": "tool_done", "tool_id": tool_id}


def format_timeline_updated() -> dict:
    """Format a timeline-state-changed notification."""
    return {"type": "timeline_updated"}


def format_done(turn_id: int) -> dict:
    """Format an end-of-turn message."""
    return {"type": "done", "turn_id": turn_id}


def format_error(message: str) -> dict:
    """Format an error message for the client."""
    return {"type": "error", "message": message}


def format_busy() -> dict:
    """Format a busy/throttle message."""
    return {"type": "busy"}


def format_connected(session_token: str) -> dict:
    """Format a connection-acknowledged message."""
    return {"type": "connected", "session_token": session_token}
