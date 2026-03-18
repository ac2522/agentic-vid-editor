"""Research agent — synthesizes web search into structured approaches."""

from __future__ import annotations

from dataclasses import dataclass

from ave.tools.search import SearchResult, PageContent


@dataclass(frozen=True)
class Approach:
    name: str
    description: str
    tool_mapping: str
    source: str
    trade_offs: str


@dataclass(frozen=True)
class ResearchBrief:
    question: str
    approaches: list[Approach]
    sources: list[str]
    confidence: float


def synthesize_research(
    question: str,
    results: list[SearchResult],
    page_contents: list[PageContent],
) -> ResearchBrief:
    """Synthesize search results into structured approaches.

    This is the non-LLM path — extracts approaches heuristically.
    The LLM-powered path uses the researcher subagent via Anthropic API.
    """
    approaches: list[Approach] = []
    sources: list[str] = []

    for result in results:
        sources.append(result.url)
        if result.snippet:
            approaches.append(
                Approach(
                    name=result.title[:80],
                    description=result.snippet,
                    tool_mapping="",
                    source=result.url,
                    trade_offs="",
                )
            )

    # Enrich from page contents
    for page in page_contents:
        if page.url not in sources:
            sources.append(page.url)

    # Deduplicate by name
    seen: set[str] = set()
    unique: list[Approach] = []
    for a in approaches:
        if a.name not in seen:
            seen.add(a.name)
            unique.append(a)

    confidence = min(1.0, len(unique) * 0.3) if unique else 0.0

    return ResearchBrief(
        question=question,
        approaches=unique[:3],
        sources=sources,
        confidence=confidence,
    )
