"""Pure REST API endpoint functions for the AVE web UI.

These functions contain no framework dependencies (no aiohttp) — they transform
data models into JSON-serialisable dicts suitable for HTTP responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

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
