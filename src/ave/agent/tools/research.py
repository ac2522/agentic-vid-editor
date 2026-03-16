"""Research domain tools — web search and technique synthesis."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_research_tools(registry: ToolRegistry) -> None:
    """Register research domain tools."""

    @registry.tool(
        domain="research",
        tags=["web", "search", "forum", "technique", "lookup", "internet"],
    )
    def web_search(
        query: str, sources: list[str] | None = None
    ) -> dict:
        """Search the web for video editing techniques and information.

        Args:
            query: Search query string.
            sources: Optional list of domains to bias toward
                (e.g., ["forum.blackmagicdesign.com"]).
        """
        return {"query": query, "sources": sources or [], "results": []}

    @registry.tool(
        domain="research",
        tags=["research", "technique", "approach", "forum", "synthesis"],
    )
    def research_technique(question: str) -> dict:
        """Research a video editing technique. Searches web, reads forums,
        synthesizes findings into a ResearchBrief with 1-3 approaches.

        Args:
            question: The technique or question to research.
        """
        return {"question": question, "status": "requires_async_execution"}
