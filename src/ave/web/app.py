"""App factory and aiohttp wiring for the AVE web UI.

Creates the aiohttp ``web.Application`` with REST endpoints, static file
serving, and a WebSocket handler for the agent chat interface.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from aiohttp import web

from ave.web.api import get_assets_response, get_timeline_response
from ave.web.chat import (
    ChatSession,
    format_connected,
    format_error,
    parse_client_message,
)
from ave.web.timeline_model import TimelineModel

logger = logging.getLogger(__name__)

# 1x1 transparent PNG (placeholder thumbnail)
_PLACEHOLDER_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def _handle_index(request: web.Request) -> web.Response:
    """Serve the main client HTML page."""
    client_dir: Path = request.app["client_dir"]
    index_path = client_dir / "index.html"
    try:
        html = index_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise web.HTTPNotFound(text="index.html not found")
    return web.Response(text=html, content_type="text/html")


async def _handle_timeline(request: web.Request) -> web.Response:
    """Return timeline state as JSON."""
    model: TimelineModel = request.app["timeline_model"]
    data = get_timeline_response(model)
    return web.json_response(data)


async def _handle_assets(request: web.Request) -> web.Response:
    """Return asset registry as JSON."""
    project_dir: Path = request.app["project_dir"]
    registry_path = project_dir / "assets" / "registry.json"
    data = get_assets_response(registry_path)
    return web.json_response(data)


async def _handle_thumbnail(request: web.Request) -> web.Response:
    """Return a placeholder thumbnail PNG."""
    return web.Response(body=_PLACEHOLDER_PNG, content_type="image/png")


def _create_chat_session(timeline_model: TimelineModel) -> ChatSession | None:
    """Try to create a ChatSession with full orchestrator.

    Returns ``None`` if dependencies (e.g. tool modules) are unavailable.
    """
    try:
        from ave.agent.orchestrator import Orchestrator
        from ave.agent.session import EditingSession

        editing_session = EditingSession()
        orchestrator = Orchestrator(editing_session)
        return ChatSession(orchestrator, timeline_model)
    except Exception:
        logger.debug("Could not create ChatSession — dependencies missing", exc_info=True)
        return None


async def _handle_chat_ws(request: web.Request) -> web.WebSocketResponse:
    """WebSocket handler for the agent chat interface."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    sessions: dict = request.app["sessions"]
    timeline_model: TimelineModel = request.app["timeline_model"]

    # Determine session token: reuse from query string or generate new
    token = request.query.get("session")
    if token and token in sessions:
        # Existing session — reconnect
        chat_session = sessions[token]
    else:
        # New session
        token = uuid.uuid4().hex
        chat_session = _create_chat_session(timeline_model)
        sessions[token] = chat_session

    # Send connected acknowledgement
    await ws.send_json(format_connected(token))

    # Message loop
    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            parsed = parse_client_message(msg.data)
            if parsed.get("type") == "error":
                await ws.send_json(parsed)
            elif parsed.get("type") == "message":
                if chat_session is None:
                    await ws.send_json(format_error("Agent not available"))
                else:
                    await chat_session.handle_message(ws, parsed.get("text", ""))
            elif parsed.get("type") == "cancel":
                if chat_session is not None:
                    chat_session.cancel()
            else:
                await ws.send_json(format_error(f"Unknown message type: {parsed.get('type')}"))
        elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
            break

    return ws


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    project_dir: Path | str | None = None,
    xges_path: Path | str | None = None,
) -> web.Application:
    """Create and configure the AVE web application.

    Parameters
    ----------
    project_dir:
        Root directory of the video project.  Defaults to cwd.
    xges_path:
        Path to the XGES timeline file.  If provided the timeline model
        is populated from this file.
    """
    project_dir = Path(project_dir) if project_dir else Path.cwd()

    # Build timeline model
    if xges_path:
        model = TimelineModel.load_from_xges(Path(xges_path))
    else:
        model = TimelineModel()

    app = web.Application()

    # Shared state
    app["timeline_model"] = model
    app["project_dir"] = project_dir
    app["sessions"] = {}

    # Determine client directory — prefer project-local, fall back to
    # package-bundled client/ (which may not exist yet).
    client_dir = project_dir / "client"
    if not client_dir.is_dir():
        client_dir = Path(__file__).parent / "client"
    app["client_dir"] = client_dir

    # Routes
    app.router.add_get("/", _handle_index)
    app.router.add_get("/api/timeline", _handle_timeline)
    app.router.add_get("/api/assets", _handle_assets)
    app.router.add_get("/api/assets/{asset_id}/thumbnail", _handle_thumbnail)
    app.router.add_get("/ws/chat", _handle_chat_ws)

    # Static files from client directory (if it exists)
    if client_dir.is_dir():
        app.router.add_static("/static", client_dir, show_index=False)

    # Try to mount preview sub-app
    try:
        from ave.preview.cache import SegmentCache
        from ave.preview.server import PreviewServer

        segments_dir = project_dir / "cache" / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        cache = SegmentCache(segments_dir)
        preview_server = PreviewServer(cache, segments_dir)
        preview_app = preview_server.create_app()
        app.add_subapp("/preview/", preview_app)
    except (ImportError, Exception):
        pass  # Preview not available — skip silently

    return app


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_server(
    project_dir: Path | str,
    xges_path: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Create and run the AVE web server (blocking)."""
    app = create_app(project_dir=project_dir, xges_path=xges_path)
    web.run_app(app, host=host, port=port)
