"""Pure REST API endpoint functions for the AVE web UI.

These functions contain no framework dependencies (no aiohttp) — they transform
data models into JSON-serialisable dicts suitable for HTTP responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ave.agent.session import EditingSession, SessionError

if TYPE_CHECKING:
    from ave.web.timeline_model import TimelineModel


def get_timeline_response(model: TimelineModel) -> dict:
    """Return the timeline state as a JSON-serialisable dict."""
    return model.to_dict()


def get_assets_response(registry_path: Path) -> dict:
    """Read the asset registry JSON and return an API-formatted response.

    Returns ``{"assets": []}`` when the file is missing or contains invalid JSON.
    """
    try:
        raw = registry_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {"assets": []}

    try:
        entries = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"assets": []}

    assets: list[dict] = []
    for entry in entries:
        asset_id = entry["asset_id"]
        assets.append(
            {
                "id": asset_id,
                "name": Path(entry["original_path"]).name,
                "duration_ns": int(entry["duration_seconds"] * 1_000_000_000),
                "resolution": f"{entry['width']}x{entry['height']}",
                "fps": entry["original_fps"],
                "thumbnail_url": f"/api/assets/{asset_id}/thumbnail",
            }
        )

    return {"assets": assets}


def undo_response(session: EditingSession, turn_id: str) -> tuple[int, dict]:
    """Pure handler: execute undo on a session and return (status_code, response_body)."""
    if not turn_id:
        return 400, {"ok": False, "error": "missing turn_id"}
    try:
        session.undo_turn(turn_id)
    except KeyError:
        return 404, {"ok": False, "error": "unknown turn"}
    except SessionError as exc:
        return 400, {"ok": False, "error": str(exc)}
    return 200, {"ok": True, "turn_id": turn_id, "direction": "undo"}


def redo_response(session: EditingSession, turn_id: str) -> tuple[int, dict]:
    """Pure handler: execute redo on a session and return (status_code, response_body)."""
    if not turn_id:
        return 400, {"ok": False, "error": "missing turn_id"}
    try:
        session.redo_turn(turn_id)
    except KeyError:
        return 404, {"ok": False, "error": "unknown turn"}
    except SessionError as exc:
        return 400, {"ok": False, "error": str(exc)}
    return 200, {"ok": True, "turn_id": turn_id, "direction": "redo"}
