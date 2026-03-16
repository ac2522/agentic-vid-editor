"""Brave Search API backend for web research."""

from __future__ import annotations

import re

import aiohttp

from ave.tools.search import SearchResult, PageContent


class BraveSearchBackend:
    """Brave Search API backend."""

    _BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Brave Search API key required")
        self._api_key = api_key

    async def search(
        self,
        query: str,
        max_results: int = 10,
        source_bias: list[str] | None = None,
    ) -> list[SearchResult]:
        if source_bias:
            site_query = " OR ".join(f"site:{s}" for s in source_bias[:3])
            query = f"{query} ({site_query})"

        data = await self._get_json(query, max_results)
        results: list[SearchResult] = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            url = item.get("url", "")
            parts = url.split("/")
            source = parts[2] if len(parts) > 2 else ""
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("description", ""),
                    source=source,
                )
            )
        return results

    async def fetch_page(self, url: str) -> PageContent:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                text = await resp.text()
        headings = re.findall(r"<h[1-6][^>]*>(.*?)</h[1-6]>", text)
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return PageContent(url=url, text=clean[:10000], headings=headings)

    async def _get_json(self, query: str, count: int) -> dict:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params = {"q": query, "count": str(count)}
        async with aiohttp.ClientSession() as session:
            async with session.get(self._BASE_URL, headers=headers, params=params) as resp:
                return await resp.json()
