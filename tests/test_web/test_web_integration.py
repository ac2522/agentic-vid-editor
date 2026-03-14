"""Integration tests for the AVE web app factory and endpoints."""

from __future__ import annotations

import json

import pytest

from tests.conftest import requires_aiohttp

pytestmark = [requires_aiohttp]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app(tmp_path):
    """Create a test application via the app factory."""
    from ave.web.app import create_app

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "assets").mkdir()
    # Create a minimal client/index.html so the static route works
    client_dir = project_dir / "client"
    client_dir.mkdir()
    (client_dir / "index.html").write_text(
        "<html><head><title>AVE</title></head><body>AVE</body></html>"
    )
    return create_app(project_dir=project_dir)


@pytest.fixture
async def client(app, aiohttp_client):
    """Create an aiohttp test client."""
    return await aiohttp_client(app)


# ---------------------------------------------------------------------------
# TestAppFactory
# ---------------------------------------------------------------------------


class TestAppFactory:
    """Test that the app factory produces a well-configured application."""

    def test_returns_application(self, app):
        from aiohttp import web

        assert isinstance(app, web.Application)

    def test_has_timeline_model(self, app):
        from ave.web.timeline_model import TimelineModel

        assert "timeline_model" in app
        assert isinstance(app["timeline_model"], TimelineModel)

    def test_has_project_dir(self, app):
        assert "project_dir" in app

    def test_has_sessions(self, app):
        assert "sessions" in app
        assert isinstance(app["sessions"], dict)


# ---------------------------------------------------------------------------
# TestRESTEndpoints
# ---------------------------------------------------------------------------


class TestRESTEndpoints:
    """Test the REST API endpoints."""

    async def test_timeline_returns_json(self, client):
        resp = await client.get("/api/timeline")
        assert resp.status == 200
        data = await resp.json()
        assert "layers" in data
        assert "duration_ns" in data
        assert "fps" in data

    async def test_assets_returns_json(self, client):
        resp = await client.get("/api/assets")
        assert resp.status == 200
        data = await resp.json()
        assert "assets" in data

    async def test_index_returns_html_with_ave(self, client):
        resp = await client.get("/")
        assert resp.status == 200
        text = await resp.text()
        assert "AVE" in text

    async def test_thumbnail_placeholder(self, client):
        resp = await client.get("/api/assets/test-asset/thumbnail")
        assert resp.status == 200
        assert resp.content_type == "image/png"


# ---------------------------------------------------------------------------
# TestWebSocketChat
# ---------------------------------------------------------------------------


class TestWebSocketChat:
    """Test the WebSocket chat endpoint."""

    async def test_connect_receives_connected_event(self, client):
        ws = await client.ws_connect("/ws/chat")
        msg = await ws.receive_json()
        assert msg["type"] == "connected"
        assert "session_token" in msg
        await ws.close()

    async def test_reconnect_preserves_session(self, client):
        # First connection — get a session token
        ws1 = await client.ws_connect("/ws/chat")
        msg1 = await ws1.receive_json()
        token = msg1["session_token"]
        await ws1.close()

        # Second connection — send the same token
        ws2 = await client.ws_connect(f"/ws/chat?session={token}")
        msg2 = await ws2.receive_json()
        assert msg2["type"] == "connected"
        assert msg2["session_token"] == token
        await ws2.close()

    async def test_message_returns_error_without_anthropic(self, client):
        ws = await client.ws_connect("/ws/chat")
        _connected = await ws.receive_json()
        await ws.send_json({"type": "message", "text": "hello"})
        resp = await ws.receive_json()
        assert resp["type"] == "error"
        # Either "anthropic package not installed" or "Agent not available"
        assert any(
            phrase in resp["message"]
            for phrase in ("anthropic", "Agent not available")
        )
        await ws.close()
