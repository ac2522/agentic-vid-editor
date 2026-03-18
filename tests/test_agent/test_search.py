"""Tests for BM25-style tool search engine."""

from __future__ import annotations


from ave.agent.registry import ToolRegistry
from ave.agent.search import SearchHit, ToolSearchEngine


def _make_registry() -> ToolRegistry:
    """Create a test registry with known tools across domains."""
    registry = ToolRegistry()

    @registry.tool(domain="editing", tags=["cut", "shorten"])
    def trim(duration_ns: int, in_ns: int, out_ns: int) -> dict:
        """Trim a clip to new in/out points."""
        return {}

    @registry.tool(domain="editing", tags=["divide", "cut"])
    def split(position_ns: int) -> dict:
        """Split a clip at the given position."""
        return {}

    @registry.tool(domain="editing", tags=["join", "merge"])
    def concat(clips: list) -> dict:
        """Concatenate multiple clips together."""
        return {}

    @registry.tool(domain="audio", tags=["loudness", "level"])
    def volume(level: float) -> dict:
        """Adjust volume level of a clip."""
        return {}

    @registry.tool(domain="audio", tags=["normalize", "loudness"])
    def loudnorm(target_lufs: float = -23.0) -> dict:
        """Normalize audio loudness to target LUFS."""
        return {}

    @registry.tool(domain="ingest", tags=["import", "media"])
    def ingest_file(path: str) -> dict:
        """Import a media file into the project."""
        return {}

    return registry


class TestToolSearchEngine:
    """Tests for ToolSearchEngine."""

    def test_reindex_all_indexes_all_tools(self):
        registry = _make_registry()
        engine = ToolSearchEngine()
        count = engine.reindex_all(registry)
        assert count == 6

    def test_search_exact_name_returns_first(self):
        registry = _make_registry()
        engine = ToolSearchEngine()
        engine.reindex_all(registry)

        results = engine.search("trim")
        assert len(results) > 0
        assert results[0].tool_name == "trim"

    def test_search_domain_filter(self):
        registry = _make_registry()
        engine = ToolSearchEngine()
        engine.reindex_all(registry)

        results = engine.search("cut", domain="editing")
        assert all(r.domain == "editing" for r in results)
        # "volume" has domain "audio" and should not appear
        assert all(r.tool_name != "volume" for r in results)

    def test_search_domain_only_returns_all_domain_tools(self):
        registry = _make_registry()
        engine = ToolSearchEngine()
        engine.reindex_all(registry)

        results = engine.search("", domain="audio")
        names = {r.tool_name for r in results}
        assert names == {"volume", "loudnorm"}

    def test_search_nonsense_returns_empty(self):
        registry = _make_registry()
        engine = ToolSearchEngine()
        engine.reindex_all(registry)

        results = engine.search("xyzzyplugh")
        assert results == []

    def test_idf_rare_term_scores_higher(self):
        """A term appearing in fewer documents should get higher IDF."""
        registry = _make_registry()
        engine = ToolSearchEngine()
        engine.reindex_all(registry)

        # "concatenate" only appears in concat's description
        rare_results = engine.search("concatenate")
        # "clip" appears in multiple tool descriptions
        common_results = engine.search("clip")

        # Both should find results
        assert len(rare_results) > 0
        assert len(common_results) > 0

        # The rare term match should have higher score than common term matches
        assert rare_results[0].score > common_results[0].score

    def test_search_limit(self):
        registry = _make_registry()
        engine = ToolSearchEngine()
        engine.reindex_all(registry)

        results = engine.search("clip", limit=2)
        assert len(results) <= 2

    def test_reindex_recomputes_idf(self):
        """After adding more tools and reindexing, IDF should change."""
        registry = _make_registry()
        engine = ToolSearchEngine()
        engine.reindex_all(registry)

        results_before = engine.search("clip")
        scores_before = {r.tool_name: r.score for r in results_before}

        # Add more tools that mention "clip"
        @registry.tool(domain="editing", tags=["clip"])
        def duplicate_clip(clip_id: str) -> dict:
            """Duplicate a clip in the timeline."""
            return {}

        @registry.tool(domain="editing", tags=["clip"])
        def delete_clip(clip_id: str) -> dict:
            """Delete a clip from the timeline."""
            return {}

        engine.reindex_all(registry)
        results_after = engine.search("clip")
        scores_after = {r.tool_name: r.score for r in results_after}

        # "clip" is now in more documents, so IDF should decrease,
        # meaning scores for tools that matched before should change
        common_tools = set(scores_before) & set(scores_after)
        assert len(common_tools) > 0
        # At least one tool's score should have changed
        assert any(scores_before[t] != scores_after[t] for t in common_tools)

    def test_search_returns_search_hit_instances(self):
        registry = _make_registry()
        engine = ToolSearchEngine()
        engine.reindex_all(registry)

        results = engine.search("trim")
        assert all(isinstance(r, SearchHit) for r in results)
        hit = results[0]
        assert hit.tool_name == "trim"
        assert hit.domain == "editing"
        assert hit.score > 0
        assert "Trim" in hit.description

    def test_name_match_boosted_over_description(self):
        """Name matches should be weighted higher than description matches."""
        registry = _make_registry()
        engine = ToolSearchEngine()
        engine.reindex_all(registry)

        # "volume" is both a tool name and appears in volume's description
        results = engine.search("volume")
        assert results[0].tool_name == "volume"

    def test_tag_match_works(self):
        """Searching by tag should find the tool."""
        registry = _make_registry()
        engine = ToolSearchEngine()
        engine.reindex_all(registry)

        results = engine.search("shorten")
        assert len(results) > 0
        assert results[0].tool_name == "trim"
