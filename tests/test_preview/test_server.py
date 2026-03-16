"""Tests for ave.preview.server — Preview server + WebSocket frame scrubbing."""

from __future__ import annotations

# pytest-asyncio auto mode so bare async defs are collected as async tests
import pytest_asyncio  # noqa: F401

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

aiohttp = pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402

from ave.preview.cache import SegmentCache  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> SegmentCache:
    """Create a SegmentCache with a few registered segments."""
    c = SegmentCache(tmp_path / "cache")
    c.register_segments(
        [
            (0, 0, 5_000_000_000),
            (1, 5_000_000_000, 10_000_000_000),
            (2, 10_000_000_000, 15_000_000_000),
        ]
    )
    return c


@pytest.fixture
def segments_dir(tmp_path: Path) -> Path:
    d = tmp_path / "segments"
    d.mkdir()
    return d


@pytest.fixture
def app(cache: SegmentCache, segments_dir: Path):
    from ave.preview.server import create_app

    return create_app(cache, segments_dir)


@pytest.fixture
def app_with_video(cache: SegmentCache, segments_dir: Path, tmp_path: Path):
    """App with a fake video_path set so frame requests don't fail with 'No video source'."""
    from ave.preview.server import create_app

    fake_video = tmp_path / "fake.mp4"
    fake_video.write_bytes(b"fake")
    return create_app(cache, segments_dir, video_path=fake_video)


# ---------------------------------------------------------------------------
# 1. test_server_creates_app
# ---------------------------------------------------------------------------


def test_server_creates_app(cache: SegmentCache, segments_dir: Path):
    from ave.preview.server import create_app

    application = create_app(cache, segments_dir)
    assert isinstance(application, web.Application)


# ---------------------------------------------------------------------------
# 2. test_server_has_routes
# ---------------------------------------------------------------------------


def test_server_has_routes(app):
    resources = [r.canonical for r in app.router.resources()]
    assert "/" in resources
    assert "/ws" in resources
    assert "/api/status" in resources
    assert any("/segments" in r for r in resources)


# ---------------------------------------------------------------------------
# 3. test_api_status_returns_cache_state
# ---------------------------------------------------------------------------


async def test_api_status_returns_cache_state(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/api/status")
    assert resp.status == 200
    data = await resp.json()
    assert data["total"] == 3
    assert data["dirty"] == 3
    assert data["clean"] == 0
    assert data["rendering"] == 0


# ---------------------------------------------------------------------------
# 4. test_ws_frame_request
# ---------------------------------------------------------------------------


async def test_ws_frame_request(aiohttp_client, app_with_video):
    client = await aiohttp_client(app_with_video)
    fake_frame = b"\xff\xd8\xff\xe0JFIF_FAKE"

    with patch("ave.preview.server.extract_frame", return_value=fake_frame) as mock_ef:
        async with client.ws_connect("/ws") as ws:
            await ws.send_json({"type": "frame", "timestamp_ns": 1_000_000_000})
            msg = await ws.receive_json()
            assert msg["type"] == "frame"
            mock_ef.assert_called_once()


# ---------------------------------------------------------------------------
# 5. test_ws_frame_response_format
# ---------------------------------------------------------------------------


async def test_ws_frame_response_format(aiohttp_client, app_with_video):
    client = await aiohttp_client(app_with_video)
    fake_frame = b"\xff\xd8\xff\xe0JFIF_FAKE"

    with patch("ave.preview.server.extract_frame", return_value=fake_frame):
        async with client.ws_connect("/ws") as ws:
            await ws.send_json({"type": "frame", "timestamp_ns": 2_000_000_000})
            msg = await ws.receive_json()

            assert msg["type"] == "frame"
            assert msg["timestamp_ns"] == 2_000_000_000
            assert msg["format"] == "jpeg"
            decoded = base64.b64decode(msg["data"])
            assert decoded == fake_frame


# ---------------------------------------------------------------------------
# 6. test_ws_invalid_message
# ---------------------------------------------------------------------------


async def test_ws_invalid_message(aiohttp_client, app):
    client = await aiohttp_client(app)
    async with client.ws_connect("/ws") as ws:
        await ws.send_str("not json at all {{{")
        msg = await ws.receive_json()
        assert msg["type"] == "error"
        assert "Invalid JSON" in msg["message"]


# ---------------------------------------------------------------------------
# 7. test_ws_playback_state
# ---------------------------------------------------------------------------


async def test_ws_playback_state(aiohttp_client, app):
    client = await aiohttp_client(app)
    async with client.ws_connect("/ws") as ws:
        await ws.send_json(
            {
                "type": "playback",
                "state": "playing",
                "position_ns": 0,
            }
        )
        msg = await ws.receive_json()
        assert msg["type"] == "playback_ack"
        assert msg["state"] == "playing"


# ---------------------------------------------------------------------------
# 8. test_ws_invalidation_notification
# ---------------------------------------------------------------------------


async def test_ws_invalidation_notification(aiohttp_client, app):
    from ave.preview.server import PreviewServer

    client = await aiohttp_client(app)
    server: PreviewServer = app["preview_server"]

    async with client.ws_connect("/ws") as ws:
        await server.notify_invalidation(0, 5_000_000_000)
        msg = await ws.receive_json()
        assert msg["type"] == "invalidation"
        assert msg["start_ns"] == 0
        assert msg["end_ns"] == 5_000_000_000


# ---------------------------------------------------------------------------
# 9. test_segment_file_served
# ---------------------------------------------------------------------------


async def test_segment_file_served(aiohttp_client, app, segments_dir: Path):
    client = await aiohttp_client(app)
    seg_file = segments_dir / "test_segment.mp4"
    seg_file.write_bytes(b"fake_mp4_data")

    resp = await client.get("/segments/test_segment.mp4")
    assert resp.status == 200
    body = await resp.read()
    assert body == b"fake_mp4_data"


# ---------------------------------------------------------------------------
# 10. test_segment_file_not_found
# ---------------------------------------------------------------------------


async def test_segment_file_not_found(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/segments/missing.mp4")
    assert resp.status == 404


# ---------------------------------------------------------------------------
# 11. test_index_serves_client_html
# ---------------------------------------------------------------------------


async def test_index_serves_client_html(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/")
    assert resp.status == 200
    assert resp.content_type == "text/html"
    text = await resp.text()
    assert "<html" in text.lower() or "<!doctype" in text.lower()
