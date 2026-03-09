"""Asset registry — JSON metadata store for ingested media."""

import json
from pathlib import Path

from pydantic import BaseModel


class AssetEntry(BaseModel):
    """Metadata for a single ingested media asset."""

    asset_id: str
    original_path: Path
    working_path: Path
    proxy_path: Path
    original_fps: float
    conformed_fps: float
    duration_seconds: float
    width: int
    height: int
    codec: str
    camera_color_space: str
    camera_transfer: str
    idt_reference: str | None = None
    transcription_path: Path | None = None
    visual_analysis_path: Path | None = None


class AssetRegistry:
    """JSON-backed registry of ingested media assets."""

    def __init__(self, path: Path):
        self._path = path
        self._entries: dict[str, AssetEntry] = {}
        if path.exists():
            self.load()

    def add(self, entry: AssetEntry) -> None:
        self._entries[entry.asset_id] = entry

    def get(self, asset_id: str) -> AssetEntry:
        return self._entries[asset_id]

    def remove(self, asset_id: str) -> None:
        del self._entries[asset_id]

    def list_all(self) -> list[AssetEntry]:
        return list(self._entries.values())

    def count(self) -> int:
        return len(self._entries)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.model_dump(mode="json") for k, v in self._entries.items()}
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def load(self) -> None:
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text())
        self._entries = {k: AssetEntry(**v) for k, v in raw.items()}
