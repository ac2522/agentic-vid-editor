"""Tests for Brave Search backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ave.tools.search_brave import BraveSearchBackend


@pytest.mark.asyncio
class TestBraveSearchBackend:
    async def test_search_returns_results(self):
        mock_response = {
            "web": {
                "results": [
                    {
                        "title": "Film Grain Tutorial",
                        "url": "https://example.com/grain",
                        "description": "How to add grain",
                    }
                ]
            }
        }
        backend = BraveSearchBackend(api_key="test-key")
        with patch.object(
            backend, "_get_json", new_callable=AsyncMock, return_value=mock_response
        ):
            results = await backend.search("film grain davinci resolve")
        assert len(results) == 1
        assert results[0].title == "Film Grain Tutorial"
        assert results[0].source == "example.com"

    async def test_search_empty_results(self):
        mock_response = {"web": {"results": []}}
        backend = BraveSearchBackend(api_key="test-key")
        with patch.object(
            backend, "_get_json", new_callable=AsyncMock, return_value=mock_response
        ):
            results = await backend.search("impossible query")
        assert len(results) == 0

    async def test_search_with_source_bias(self):
        mock_response = {"web": {"results": []}}
        backend = BraveSearchBackend(api_key="test-key")
        with patch.object(
            backend, "_get_json", new_callable=AsyncMock, return_value=mock_response
        ) as mock:
            await backend.search(
                "film grain", source_bias=["forum.blackmagicdesign.com"]
            )
            call_args = str(mock.call_args)
            assert "site:forum.blackmagicdesign.com" in call_args

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="API key"):
            BraveSearchBackend(api_key="")
