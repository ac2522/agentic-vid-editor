"""Preview server — HTTP segments + WebSocket frame scrubbing.

Uses aiohttp for both HTTP static serving and WebSocket connections.
"""

from __future__ import annotations

import asyncio
import base64
import json
import weakref
from pathlib import Path

from ave.preview.cache import SegmentCache
from ave.preview.frame import extract_frame

try:
    from aiohttp import web
except ImportError:
    raise ImportError(
        "aiohttp required for preview server. Install with: pip install ave[preview]"
    )


class PreviewServer:
    """Preview server combining HTTP segment serving and WebSocket frame scrubbing."""

    def __init__(
        self,
        cache: SegmentCache,
        segments_dir: Path,
        video_path: Path | None = None,
    ) -> None:
        self._cache = cache
        self._segments_dir = Path(segments_dir)
        self._segments_dir.mkdir(parents=True, exist_ok=True)
        self._video_path = video_path
        self._websockets: weakref.WeakSet[web.WebSocketResponse] = weakref.WeakSet()
        self._app: web.Application | None = None

    def create_app(self) -> web.Application:
        """Create the aiohttp application with all routes."""
        app = web.Application()
        app.add_routes([
            web.get("/", self._handle_index),
            web.get("/ws", self._handle_websocket),
            web.get("/api/status", self._handle_status),
        ])
        # Serve segment files as static
        app.add_routes([web.static("/segments", self._segments_dir)])
        # Store reference so tests can access the server instance
        app["preview_server"] = self
        self._app = app
        return app

    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve the browser preview client."""
        client_html = (Path(__file__).parent / "client" / "index.html").read_text()
        return web.Response(text=client_html, content_type="text/html")

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Return cache status as JSON."""
        return web.json_response(self._cache.segment_count())

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for frame scrubbing and playback."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._websockets.add(ws)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    await self._process_ws_message(ws, msg.data)
                elif msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            self._websockets.discard(ws)

        return ws

    async def _process_ws_message(self, ws: web.WebSocketResponse, data: str) -> None:
        """Process an incoming WebSocket message."""
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            await ws.send_json({"type": "error", "message": "Invalid JSON"})
            return

        msg_type = message.get("type")

        if msg_type == "frame":
            await self._handle_frame_request(ws, message)
        elif msg_type == "playback":
            await ws.send_json({
                "type": "playback_ack",
                "state": message.get("state"),
            })
        else:
            await ws.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}",
            })

    async def _handle_frame_request(
        self, ws: web.WebSocketResponse, message: dict
    ) -> None:
        """Extract and return a frame at the requested timecode."""
        timestamp_ns = message.get("timestamp_ns")
        if timestamp_ns is None:
            await ws.send_json({"type": "error", "message": "Missing timestamp_ns"})
            return

        if self._video_path is None:
            await ws.send_json({"type": "error", "message": "No video source configured"})
            return

        loop = asyncio.get_running_loop()
        try:
            frame_bytes = await loop.run_in_executor(
                None, lambda: extract_frame(self._video_path, timestamp_ns, format="jpeg")
            )
            frame_b64 = base64.b64encode(frame_bytes).decode("ascii")
            await ws.send_json({
                "type": "frame",
                "timestamp_ns": timestamp_ns,
                "data": frame_b64,
                "format": "jpeg",
            })
        except Exception as e:
            await ws.send_json({"type": "error", "message": str(e)})

    async def notify_invalidation(self, start_ns: int, end_ns: int) -> None:
        """Notify all connected clients about segment invalidation."""
        message = {"type": "invalidation", "start_ns": start_ns, "end_ns": end_ns}
        for ws in list(self._websockets):
            if not ws.closed:
                await ws.send_json(message)

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Run the server."""
        app = self.create_app()
        web.run_app(app, host=host, port=port)


def create_app(
    cache: SegmentCache,
    segments_dir: Path,
    video_path: Path | None = None,
) -> web.Application:
    """Factory function to create the preview app."""
    server = PreviewServer(cache, segments_dir, video_path)
    return server.create_app()
