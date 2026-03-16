"""Search backend protocol and data models for web research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str


@dataclass(frozen=True)
class PageContent:
    url: str
    text: str
    headings: list[str]


VIDEO_EDITING_SOURCES = (
    "forum.blackmagicdesign.com",
    "community.frame.io",
    "liftgammagain.com",
    "reddit.com/r/colorgrading",
    "reddit.com/r/VideoEditing",
    "cinematography.com",
    "docs.arri.com",
    "sony.com/en/articles/technical",
)


class SearchBackend(Protocol):
    """Protocol for web search backends. Type-annotation only."""

    async def search(
        self, query: str, max_results: int = 10
    ) -> list[SearchResult]: ...

    async def fetch_page(self, url: str) -> PageContent: ...
