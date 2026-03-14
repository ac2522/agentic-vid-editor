"""Integration tests for ChatSession and WebSocket chat with agentic loop.

These tests verify protocol-level behaviour and do NOT require the
anthropic package to be installed.
"""

from __future__ import annotations

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
    client_dir = project_dir / "client"
    client_dir.mkdir()
    (client_dir / "index.html").write_text("<html><body>AVE</body></html>")
    return create_app(project_dir=project_dir)


@pytest.fixture
async def client(app, aiohttp_client):
    """Create an aiohttp test client."""
    return await aiohttp_client(app)


# ---------------------------------------------------------------------------
# TestChatProtocol — protocol-level WebSocket behaviour
# ---------------------------------------------------------------------------


class TestChatProtocol:
    """Protocol-level tests that work regardless of anthropic availability."""

    async def test_invalid_json_returns_error(self, client):
        ws = await client.ws_connect("/ws/chat")
        _connected = await ws.receive_json()
        await ws.send_str("not valid json {{{")
        resp = await ws.receive_json()
        assert resp["type"] == "error"
        assert "Invalid JSON" in resp["message"]
        await ws.close()

    async def test_missing_type_field_returns_error(self, client):
        ws = await client.ws_connect("/ws/chat")
        _connected = await ws.receive_json()
        await ws.send_json({"text": "hello"})
        resp = await ws.receive_json()
        assert resp["type"] == "error"
        assert "type" in resp["message"]
        await ws.close()

    async def test_unknown_type_returns_error(self, client):
        ws = await client.ws_connect("/ws/chat")
        _connected = await ws.receive_json()
        await ws.send_json({"type": "unknown_type"})
        resp = await ws.receive_json()
        assert resp["type"] == "error"
        assert "Unknown message type" in resp["message"]
        await ws.close()

    async def test_session_token_preserved_on_reconnect(self, client):
        # First connection
        ws1 = await client.ws_connect("/ws/chat")
        msg1 = await ws1.receive_json()
        assert msg1["type"] == "connected"
        token = msg1["session_token"]
        await ws1.close()

        # Reconnect with same token
        ws2 = await client.ws_connect(f"/ws/chat?session={token}")
        msg2 = await ws2.receive_json()
        assert msg2["type"] == "connected"
        assert msg2["session_token"] == token
        await ws2.close()

    async def test_new_session_gets_unique_token(self, client):
        ws1 = await client.ws_connect("/ws/chat")
        msg1 = await ws1.receive_json()
        token1 = msg1["session_token"]
        await ws1.close()

        ws2 = await client.ws_connect("/ws/chat")
        msg2 = await ws2.receive_json()
        token2 = msg2["session_token"]
        await ws2.close()

        assert token1 != token2


# ---------------------------------------------------------------------------
# TestChatSessionMessage — message handling
# ---------------------------------------------------------------------------


class TestChatSessionMessage:
    """Test message handling through the WebSocket."""

    async def test_message_gets_response(self, client):
        """Sending a message should get either an error or a response.

        Without anthropic installed, the ChatSession will return an error
        about the missing package, then a done event.
        If the session couldn't be created, it returns 'Agent not available'.
        """
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

    async def test_cancel_does_not_error(self, client):
        """Sending cancel when nothing is running should not crash."""
        ws = await client.ws_connect("/ws/chat")
        _connected = await ws.receive_json()
        await ws.send_json({"type": "cancel"})
        # No error response expected — cancel is silently accepted
        # Send another message to verify the connection is still alive
        await ws.send_json({"type": "message", "text": "hi"})
        resp = await ws.receive_json()
        assert resp["type"] == "error"
        await ws.close()


# ---------------------------------------------------------------------------
# TestChatSessionUnit — unit tests for ChatSession class
# ---------------------------------------------------------------------------


class TestChatSessionUnit:
    """Unit tests for ChatSession without network."""

    def test_chat_session_init(self):
        from ave.web.chat import ChatSession
        from ave.web.timeline_model import TimelineModel

        class FakeOrchestrator:
            turn_count = 0

        session = ChatSession(FakeOrchestrator(), TimelineModel())
        assert session._processing is False
        assert session._messages == []

    def test_cancel_sets_event(self):
        from ave.web.chat import ChatSession
        from ave.web.timeline_model import TimelineModel

        class FakeOrchestrator:
            turn_count = 0

        session = ChatSession(FakeOrchestrator(), TimelineModel())
        assert not session._cancel_event.is_set()
        session.cancel()
        assert session._cancel_event.is_set()
