"""Tests for Tavily Search backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("aiohttp")

from ave.tools.search_tavily import TavilySearchBackend  # noqa: E402


@pytest.mark.asyncio
class TestTavilySearchBackend:
    async def test_search_returns_results(self):
        mock_response = {
            "results": [
                {
                    "title": "Film Grain in Resolve",
                    "url": "https://forum.blackmagicdesign.com/grain",
                    "content": "Apply film grain using the OpenFX panel...",
                }
            ]
        }
        backend = TavilySearchBackend(api_key="test-key")
        with patch.object(
            backend, "_post_search", new_callable=AsyncMock, return_value=mock_response
        ):
            results = await backend.search("film grain resolve")
        assert len(results) == 1
        assert results[0].title == "Film Grain in Resolve"
        assert len(results[0].snippet) <= 500

    async def test_search_empty_results(self):
        backend = TavilySearchBackend(api_key="test-key")
        with patch.object(
            backend, "_post_search", new_callable=AsyncMock, return_value={"results": []}
        ):
            results = await backend.search("impossible query xyz")
        assert len(results) == 0

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="API key"):
            TavilySearchBackend(api_key="")
