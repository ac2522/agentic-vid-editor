"""Tests for context artifact models and extraction."""

from __future__ import annotations

import pytest

from ave.optimize.artifacts import (
    ArtifactExtractor,
    ArtifactKind,
    ContextArtifact,
)


class TestContextArtifact:
    """Tests for the ContextArtifact data model."""

    def test_create_system_prompt_artifact(self):
        artifact = ContextArtifact(
            id="role.editor.system_prompt",
            kind=ArtifactKind.SYSTEM_PROMPT,
            content="You are a professional video editor.",
            source_location="src/ave/agent/roles.py:25-55",
            metadata={"role_name": "editor"},
        )
        assert artifact.id == "role.editor.system_prompt"
        assert artifact.kind == ArtifactKind.SYSTEM_PROMPT
        assert "video editor" in artifact.content

    def test_create_tool_description_artifact(self):
        artifact = ContextArtifact(
            id="tool.trim_clip.description",
            kind=ArtifactKind.TOOL_DESCRIPTION,
            content="Trim a clip by adjusting start and end points.",
            source_location="src/ave/tools/edit.py",
            metadata={"domain": "editing", "tags": ("trim", "cut")},
        )
        assert artifact.kind == ArtifactKind.TOOL_DESCRIPTION
        assert artifact.metadata["domain"] == "editing"

    def test_artifact_is_frozen(self):
        artifact = ContextArtifact(
            id="role.editor.system_prompt",
            kind=ArtifactKind.SYSTEM_PROMPT,
            content="content",
            source_location="file.py",
            metadata={},
        )
        with pytest.raises(AttributeError):
            artifact.content = "new content"  # type: ignore[misc]

    def test_artifact_kinds_are_strings(self):
        assert ArtifactKind.SYSTEM_PROMPT == "system_prompt"
        assert ArtifactKind.TOOL_DESCRIPTION == "tool_description"
        assert ArtifactKind.ROLE_DESCRIPTION == "role_description"
        assert ArtifactKind.ORCHESTRATOR_PROMPT == "orchestrator"

    def test_artifact_id_uses_dots_not_colons(self):
        artifact = ContextArtifact(
            id="role.editor.system_prompt",
            kind=ArtifactKind.SYSTEM_PROMPT,
            content="content",
            source_location="file.py",
            metadata={},
        )
        assert ":" not in artifact.id
        assert "." in artifact.id


class TestArtifactExtractor:
    """Tests for extracting artifacts from AVE code structures."""

    def _make_role(self, name: str, description: str, system_prompt: str):
        """Create a mock AgentRole-like object."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class MockRole:
            name: str
            description: str
            system_prompt: str
            domains: tuple[str, ...] = ()

        return MockRole(name=name, description=description, system_prompt=system_prompt)

    def test_extract_from_roles_produces_two_artifacts_per_role(self):
        extractor = ArtifactExtractor()
        roles = [
            self._make_role("editor", "Expert video editor", "You are a video editor."),
        ]
        artifacts = extractor.extract_from_roles(roles)
        assert len(artifacts) == 2
        kinds = {a.kind for a in artifacts}
        assert ArtifactKind.SYSTEM_PROMPT in kinds
        assert ArtifactKind.ROLE_DESCRIPTION in kinds

    def test_extract_from_roles_content_matches(self):
        extractor = ArtifactExtractor()
        roles = [
            self._make_role("colorist", "Color specialist", "You are a colorist."),
        ]
        artifacts = extractor.extract_from_roles(roles)
        prompt_artifact = next(a for a in artifacts if a.kind == ArtifactKind.SYSTEM_PROMPT)
        desc_artifact = next(a for a in artifacts if a.kind == ArtifactKind.ROLE_DESCRIPTION)
        assert prompt_artifact.content == "You are a colorist."
        assert desc_artifact.content == "Color specialist"

    def test_extract_from_roles_ids_are_dotted(self):
        extractor = ArtifactExtractor()
        roles = [
            self._make_role("editor", "desc", "prompt"),
        ]
        artifacts = extractor.extract_from_roles(roles)
        ids = {a.id for a in artifacts}
        assert "role.editor.system_prompt" in ids
        assert "role.editor.description" in ids

    def test_extract_from_roles_metadata_includes_role_name(self):
        extractor = ArtifactExtractor()
        roles = [
            self._make_role("editor", "desc", "prompt"),
        ]
        artifacts = extractor.extract_from_roles(roles)
        for artifact in artifacts:
            assert artifact.metadata["role_name"] == "editor"

    def test_extract_from_registry(self):
        from ave.agent.registry import ToolRegistry

        registry = ToolRegistry()

        @registry.tool(domain="editing", tags=["trim", "cut"])
        def trim_clip(start_ns: int, end_ns: int) -> dict:
            """Trim a clip by adjusting start and end points.

            Args:
                start_ns: Start time in nanoseconds.
                end_ns: End time in nanoseconds.
            """
            return {}

        extractor = ArtifactExtractor()
        artifacts = extractor.extract_from_registry(registry)
        assert len(artifacts) == 1
        assert artifacts[0].id == "tool.trim_clip.description"
        assert artifacts[0].kind == ArtifactKind.TOOL_DESCRIPTION
        assert "Trim a clip" in artifacts[0].content
        assert artifacts[0].metadata["domain"] == "editing"

    def test_extract_from_registry_multiple_tools(self):
        from ave.agent.registry import ToolRegistry

        registry = ToolRegistry()

        @registry.tool(domain="editing")
        def trim_clip() -> dict:
            """Trim a clip."""
            return {}

        @registry.tool(domain="audio")
        def adjust_volume() -> dict:
            """Adjust audio volume."""
            return {}

        extractor = ArtifactExtractor()
        artifacts = extractor.extract_from_registry(registry)
        assert len(artifacts) == 2
        domains = {a.metadata["domain"] for a in artifacts}
        assert domains == {"editing", "audio"}

    def test_extract_all_combines_roles_and_registry(self):
        from ave.agent.registry import ToolRegistry

        registry = ToolRegistry()

        @registry.tool(domain="editing")
        def trim_clip() -> dict:
            """Trim a clip."""
            return {}

        roles = [
            self._make_role("editor", "Video editor", "You edit videos."),
        ]
        extractor = ArtifactExtractor()
        artifacts = extractor.extract_all(roles=roles, registry=registry)
        # 2 from role + 1 from tool
        assert len(artifacts) == 3

    def test_extract_all_with_no_sources_returns_empty(self):
        extractor = ArtifactExtractor()
        artifacts = extractor.extract_all()
        assert artifacts == []
