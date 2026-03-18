"""Tests for search data models and protocol."""

from __future__ import annotations

import pytest

from ave.tools.search import SearchResult, PageContent, VIDEO_EDITING_SOURCES


class TestSearchDataModels:
    def test_search_result_frozen(self):
        r = SearchResult(
            title="Film Grain in Resolve",
            url="https://forum.blackmagicdesign.com/123",
            snippet="Here's how to add film grain...",
            source="forum.blackmagicdesign.com",
        )
        assert r.title == "Film Grain in Resolve"
        with pytest.raises(AttributeError):
            r.title = "changed"  # type: ignore[misc]

    def test_page_content(self):
        p = PageContent(
            url="https://example.com",
            text="Full page text here",
            headings=["Introduction", "Method"],
        )
        assert len(p.headings) == 2

    def test_video_editing_sources_exist(self):
        assert "forum.blackmagicdesign.com" in VIDEO_EDITING_SOURCES
        assert len(VIDEO_EDITING_SOURCES) >= 5
