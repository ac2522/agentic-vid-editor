"""Tavily Search API backend — optimized for LLM consumption.

Default search backend for AVE. Free tier: 1,000 credits/month.
Returns pre-ranked, LLM-formatted results with relevance scores.
"""

from __future__ import annotations

import aiohttp

from ave.tools.search import SearchResult, PageContent


class TavilySearchBackend:
    """Tavily Search API backend — optimized for agent workflows."""

    _BASE_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Tavily API key required (set AVE_SEARCH_API_KEY)")
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

        data = await self._post_search(query, max_results)
        results: list[SearchResult] = []
        for item in data.get("results", [])[:max_results]:
            url = item.get("url", "")
            parts = url.split("/")
            source = parts[2] if len(parts) > 2 else ""
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("content", "")[:500],
                    source=source,
                )
            )
        return results

    async def fetch_page(self, url: str) -> PageContent:
        """Tavily's extract endpoint returns clean, LLM-ready text."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.tavily.com/extract",
                json={"api_key": self._api_key, "urls": [url]},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
        results = data.get("results", [])
        if results:
            raw = results[0].get("raw_content", "")
            return PageContent(url=url, text=raw[:10000], headings=[])
        return PageContent(url=url, text="", headings=[])

    async def _post_search(self, query: str, max_results: int) -> dict:
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_answer": False,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._BASE_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                return await resp.json()
