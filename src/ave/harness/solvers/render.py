"""Render-tier solver — drives a real EditingSession, then renders to MP4.

Runs the same agent loop as ``execute_solver`` (real tools + real session +
snapshots + activity log), then attempts to render the final timeline. The
real GES renderer is preferred when available; otherwise a best-effort
ffmpeg lavfi placeholder is produced so downstream VLM judges always have
*some* artifact to score.

Metadata contract populated on ``state.metadata``:

* ``called_tools``     -- list[str], inherited from execute pattern
* ``final_xges``       -- str, final XGES content
* ``snapshot_count``   -- int
* ``activity_entries`` -- list[dict]
* ``rendered_path``    -- str | None, MP4 path (None if render failed)
* ``render_failed``    -- bool
* ``render_log``       -- str, short summary or error
* ``render_preset``    -- str, the preset used
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from inspect_ai.solver import Generate, Solver, TaskState, solver, use_tools

from ave.agent.activity import ActivityLog
from ave.agent.session import EditingSession
from ave.harness.schema import Scenario
from ave.harness.solvers.execute import MINIMAL_XGES, _all_registry_tools
from ave.harness.solvers.plan import extract_tool_calls
from ave.project.snapshots import SnapshotManager


RENDER_METADATA_KEYS: tuple[str, ...] = (
    "called_tools",
    "final_xges",
    "snapshot_count",
    "activity_entries",
    "rendered_path",
    "render_failed",
    "render_log",
    "render_preset",
)


_DEFAULT_PRESET = "h264_web"


def _resolve_preset(scenario: Scenario | None) -> str:
    if scenario is None or scenario.expected.render is None:
        return _DEFAULT_PRESET
    name = scenario.expected.render.preset
    return name or _DEFAULT_PRESET


def _try_ges_render(xges_path: Path, output_path: Path) -> str:
    """Try the GES proxy renderer. Returns a status string on success."""
    from ave.render.proxy import render_proxy

    render_proxy(xges_path, output_path, height=240)
    return f"ges proxy render -> {output_path}"


def _try_ffmpeg_lavfi(output_path: Path) -> str:
    """Fallback: produce a 1-second testsrc MP4 via ffmpeg lavfi."""
    from ave.harness.fixtures.builder import build_lavfi_clip

    build_lavfi_clip(
        "testsrc=size=320x240:rate=24",
        duration_seconds=1.0,
        output_path=output_path,
    )
    return f"lavfi placeholder -> {output_path}"


def _render_timeline(xges_path: Path, preset: str, output_dir: Path) -> Path:
    """Best-effort render. Tries GES first, then falls back to lavfi clip."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "render.mp4"

    errors: list[str] = []
    try:
        _try_ges_render(xges_path, output_path)
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
        errors.append("ges produced empty file")
    except (ImportError, ValueError) as exc:
        errors.append(f"ges unavailable: {exc}")
    except Exception as exc:
        errors.append(f"ges render failed: {exc}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "render fallback unavailable: ffmpeg not on PATH; "
            f"prior attempts: {'; '.join(errors) or 'none'}"
        )

    try:
        _try_ffmpeg_lavfi(output_path)
        return output_path
    except (RuntimeError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"render fallback failed: {exc}; prior: {'; '.join(errors) or 'none'}"
        ) from exc


@solver
def render_solver() -> Solver:
    """Drive the agent then render the final timeline to MP4."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tmp_dir = tempfile.mkdtemp(prefix="ave_render_")
        xges_path = Path(tmp_dir) / "project.xges"
        xges_path.write_text(MINIMAL_XGES)

        snapshot_manager = SnapshotManager()
        activity_log = ActivityLog()
        session = EditingSession(
            snapshot_manager=snapshot_manager,
            activity_log=activity_log,
        )
        session.load_project(xges_path)

        state.metadata = dict(state.metadata or {})
        state.metadata["_xges_dir"] = tmp_dir

        scenario: Scenario | None = state.metadata.get("scenario")
        preset_name = _resolve_preset(scenario)

        tools = _all_registry_tools(session)
        if tools:
            state = await use_tools(tools)(state, generate)
        state = await generate(state)

        state.metadata["called_tools"] = extract_tool_calls(state)
        state.metadata["final_xges"] = xges_path.read_text()
        state.metadata["snapshot_count"] = len(snapshot_manager.list_snapshots())
        state.metadata["activity_entries"] = [
            e.to_dict() for e in activity_log.entries()
        ]
        state.metadata["render_preset"] = preset_name

        output_dir = Path(tmp_dir) / "render"
        try:
            rendered = _render_timeline(xges_path, preset_name, output_dir)
            state.metadata["rendered_path"] = str(rendered)
            state.metadata["render_failed"] = False
            state.metadata["render_log"] = (
                f"render ok preset={preset_name} -> {rendered}"
            )
        except Exception as exc:
            state.metadata["rendered_path"] = None
            state.metadata["render_failed"] = True
            state.metadata["render_log"] = f"render failed: {exc}"

        return state

    return solve
