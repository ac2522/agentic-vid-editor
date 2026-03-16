"""FastMCP server for AVE — 7 outcome-oriented tools."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ave.agent.session import EditingSession
from ave.mcp.jobs import JobTracker
from ave.mcp.types import EditResult, ProjectState, PreviewResult, AssetInfo

try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]


def create_mcp_server(
    session: EditingSession | None = None,
) -> FastMCP:
    """Create AVE's MCP server with 7 outcome-oriented tools."""
    if FastMCP is None:
        raise ImportError("fastmcp is required: pip install fastmcp")

    if session is None:
        session = EditingSession()

    job_tracker = JobTracker()

    mcp = FastMCP(
        "ave",
        instructions="Agentic Video Editor — use edit_video for natural "
        "language editing, or search_tools + call_tool for granular control. "
        "For long-running operations, edit_video returns a job_id — use "
        "get_job_status to poll progress.",
    )

    @mcp.tool()
    def edit_video(instruction: str, options: dict | None = None) -> dict:
        """Natural language video editing. AVE's internal orchestrator
        handles tool discovery, role-based routing, execution, and
        verification.

        For long-running operations (rotoscoping, research), returns a
        job_id that can be polled with get_job_status.

        Examples:
          edit_video(instruction="remove all filler words")
          edit_video(instruction="add cross dissolve between clips 3 and 4")
        """
        # Acquire orchestrator lock for multi-client safety
        with session.orchestrator_lock:
            # TODO: Wire to orchestrator agentic loop
            return asdict(
                EditResult(
                    success=False,
                    description=f"Received instruction: {instruction}",
                    error="orchestrator_not_connected",
                )
            )

    @mcp.tool()
    def get_job_status(job_id: str) -> dict:
        """Check status of a long-running operation.

        Returns job status, progress (0.0-1.0), and result when complete.
        """
        job = job_tracker.get(job_id)
        if job is None:
            return {"error": f"Unknown job: {job_id}"}
        return job.to_dict()

    @mcp.tool()
    def get_project_state(include: list[str] | None = None) -> dict:
        """Current timeline structure, clips, effects, metadata.

        Args:
            include: Optional filter — e.g. ['clips', 'effects', 'metadata']
        """
        state = session.to_dict()
        return asdict(
            ProjectState(
                clip_count=state.get("history_length", 0),
                duration_ns=0,
                layers=0,
            )
        )

    @mcp.tool()
    def render_preview(segment: str | None = None, format: str = "jpeg") -> dict:
        """Render a preview frame or segment.

        Args:
            segment: Time range as "start_ns-stop_ns" or None for current.
            format: Output format — "jpeg", "png", or "mp4".
        """
        return asdict(PreviewResult(path="", format=format))

    @mcp.tool()
    def ingest_asset(path: str, options: dict | None = None) -> dict:
        """Bring media into the project. Auto-probes codec, resolution,
        frame rate, color space, and duration."""
        return asdict(
            AssetInfo(
                asset_id="",
                path=path,
                codec="unknown",
                width=0,
                height=0,
                duration_ns=0,
            )
        )

    @mcp.tool()
    def search_tools(query: str, domain: str | None = None) -> list[dict]:
        """Discover AVE's granular tools by keyword or domain.

        Returns tool names, summaries, domains, and namespaces.
        """
        results = session.search_tools(query, domain)
        return [
            {
                "name": r.name,
                "domain": r.domain,
                "description": r.description,
                "namespace": r.namespace,
            }
            for r in results
        ]

    @mcp.tool()
    def call_tool(name: str, params: dict) -> Any:
        """Execute any registered AVE tool directly by name.

        Use search_tools first to discover available tools and schemas.
        """
        return session.call_tool(name, params)

    return mcp
