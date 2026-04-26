"""Versioned artifact storage with regression guards."""

from __future__ import annotations

import difflib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ave.optimize.artifacts import ArtifactKind, ContextArtifact


class ArtifactStore:
    """Versioned storage for optimized artifacts.

    Storage layout:
        <base_dir>/optimized/
        ├── campaigns.jsonl
        └── artifacts/
            ├── role.editor.system_prompt/
            │   ├── v1.txt
            │   ├── v2.txt
            │   └── meta.json
            └── tool.trim_clip.description/
                ├── v1.txt
                └── meta.json
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir / "optimized"

    def _artifact_dir(self, artifact_id: str) -> Path:
        candidate = (self.base_dir / "artifacts" / artifact_id).resolve()
        root = (self.base_dir / "artifacts").resolve()
        if not str(candidate).startswith(str(root) + "/") and candidate != root:
            raise ValueError(f"Invalid artifact_id: {artifact_id!r}")
        return candidate

    def _meta_path(self, artifact_id: str) -> Path:
        return self._artifact_dir(artifact_id) / "meta.json"

    def _load_meta(self, artifact_id: str) -> list[dict[str, Any]]:
        meta_path = self._meta_path(artifact_id)
        if not meta_path.exists():
            return []
        return json.loads(meta_path.read_text())

    def _save_meta(self, artifact_id: str, meta: list[dict[str, Any]]) -> None:
        meta_path = self._meta_path(artifact_id)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2))

    def save(
        self, artifact: ContextArtifact, score: float, campaign_id: str
    ) -> int:
        """Save an artifact version. Returns the version number."""
        artifact_dir = self._artifact_dir(artifact.id)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        meta = self._load_meta(artifact.id)
        version = len(meta) + 1

        # Write content
        version_path = artifact_dir / f"v{version}.txt"
        version_path.write_text(artifact.content)

        # Update meta
        meta.append(
            {
                "version": version,
                "score": score,
                "campaign_id": campaign_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind": artifact.kind.value if isinstance(artifact.kind, ArtifactKind) else artifact.kind,
            }
        )
        self._save_meta(artifact.id, meta)

        # Log to campaigns
        self._log_campaign(campaign_id, artifact.id, version, score)

        return version

    def load_best(self, artifact_id: str) -> ContextArtifact | None:
        """Load the highest-scoring version of an artifact."""
        meta = self._load_meta(artifact_id)
        if not meta:
            return None

        best = max(meta, key=lambda m: m["score"])
        version_path = self._artifact_dir(artifact_id) / f"v{best['version']}.txt"
        if not version_path.exists():
            return None

        content = version_path.read_text()
        kind_str = best.get("kind", "system_prompt")
        try:
            kind = ArtifactKind(kind_str)
        except ValueError:
            kind = ArtifactKind.SYSTEM_PROMPT

        return ContextArtifact(
            id=artifact_id,
            kind=kind,
            content=content,
            source_location="",
            metadata={},
        )

    def current_best_score(self, artifact_id: str) -> float | None:
        """Return the score of the current best version, or None if no versions."""
        meta = self._load_meta(artifact_id)
        if not meta:
            return None
        return max(m["score"] for m in meta)

    def history(self, artifact_id: str) -> list[dict[str, Any]]:
        """Return version history with scores."""
        return self._load_meta(artifact_id)

    def diff(self, artifact_id: str, v1: int, v2: int) -> str:
        """Return unified diff between two versions."""
        dir_ = self._artifact_dir(artifact_id)
        path1 = dir_ / f"v{v1}.txt"
        path2 = dir_ / f"v{v2}.txt"

        if not path1.exists() or not path2.exists():
            raise FileNotFoundError(
                f"Version file not found for {artifact_id}: v{v1} or v{v2}"
            )
        text1 = path1.read_text().splitlines(keepends=True)
        text2 = path2.read_text().splitlines(keepends=True)

        diff_lines = difflib.unified_diff(
            text1, text2, fromfile=f"v{v1}", tofile=f"v{v2}"
        )
        return "".join(diff_lines)

    def _log_campaign(
        self, campaign_id: str, artifact_id: str, version: int, score: float
    ) -> None:
        """Append entry to campaigns.jsonl."""
        campaigns_path = self.base_dir / "campaigns.jsonl"
        campaigns_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "campaign_id": campaign_id,
            "artifact_id": artifact_id,
            "version": version,
            "score": score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(campaigns_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
