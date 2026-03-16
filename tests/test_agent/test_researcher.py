"""Tests for research brief synthesis."""

from __future__ import annotations

from ave.agent.researcher import Approach, ResearchBrief, synthesize_research
from ave.tools.search import SearchResult


class TestApproach:
    def test_creation(self):
        a = Approach(
            name="ARRI Official LUT",
            description="Download ARRI's official LogC4 to Rec.709 LUT",
            tool_mapping="Use lut_apply tool with downloaded LUT file",
            source="https://docs.arri.com/luts",
            trade_offs="Accurate but no creative control",
        )
        assert a.name == "ARRI Official LUT"


class TestResearchBrief:
    def test_creation(self):
        brief = ResearchBrief(
            question="How to match ARRI LogC4 to Rec.709?",
            approaches=[
                Approach("LUT", "Use LUT", "lut_apply", "url", "simple"),
            ],
            sources=["https://example.com"],
            confidence=0.8,
        )
        assert len(brief.approaches) == 1
        assert brief.confidence == 0.8


class TestSynthesizeResearch:
    def test_synthesize_from_multiple_sources(self):
        results = [
            SearchResult("LUT Method", "https://a.com", "Use official LUT pack", "a.com"),
            SearchResult("Manual Grade", "https://b.com", "S-curve with lift", "b.com"),
            SearchResult("ACES Pipeline", "https://c.com", "Use ACES IDT transform", "c.com"),
        ]
        brief = synthesize_research("LogC4 to Rec.709", results, [])
        assert len(brief.approaches) == 3
        assert brief.confidence >= 0.8
        assert len(brief.sources) == 3

    def test_synthesize_caps_at_three_approaches(self):
        results = [
            SearchResult(f"Method {i}", f"https://{i}.com", f"Do thing {i}", f"{i}.com")
            for i in range(10)
        ]
        brief = synthesize_research("test", results, [])
        assert len(brief.approaches) <= 3

    def test_synthesize_empty_results(self):
        brief = synthesize_research("test", [], [])
        assert len(brief.approaches) == 0
        assert brief.confidence == 0.0

    def test_synthesize_deduplicates_by_name(self):
        results = [
            SearchResult("Same Title", "https://a.com", "Version A", "a.com"),
            SearchResult("Same Title", "https://b.com", "Version B", "b.com"),
        ]
        brief = synthesize_research("test", results, [])
        assert len(brief.approaches) == 1
