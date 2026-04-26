"""Artifact store for rendered MP4s and judge traces.

Layout:
    <root>/<scenario_id>/<run_id>/
        render.mp4
        trace.json
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactInfo:
    scenario_id: str
    run_id: str
    mp4_path: str | None = None
    trace_path: str | None = None


class ArtifactStore:
    def __init__(self, root: Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _dir(self, scenario_id: str, run_id: str) -> Path:
        d = self._root / scenario_id / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_render(self, *, scenario_id: str, run_id: str, mp4_path: Path) -> ArtifactInfo:
        dst_dir = self._dir(scenario_id, run_id)
        dst = dst_dir / "render.mp4"
        shutil.copy2(mp4_path, dst)
        return ArtifactInfo(scenario_id, run_id, mp4_path=str(dst))

    def write_trace(self, *, scenario_id: str, run_id: str, trace: dict) -> ArtifactInfo:
        dst_dir = self._dir(scenario_id, run_id)
        dst = dst_dir / "trace.json"
        dst.write_text(json.dumps(trace, indent=2, default=str))
        return ArtifactInfo(scenario_id, run_id, trace_path=str(dst))

    def prune(self, *, retention_days: int) -> int:
        """Remove artifact files older than retention_days. Returns count pruned."""
        cutoff = time.time() - retention_days * 86400
        pruned = 0
        for f in self._root.rglob("*"):
            if f.is_file():
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        pruned += 1
                except FileNotFoundError:
                    continue
        for d in sorted(self._root.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                try:
                    d.rmdir()
                except OSError:
                    pass
        return pruned
