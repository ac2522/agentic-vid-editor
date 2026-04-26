"""Context artifact models and extraction from AVE code structures."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence


class ArtifactKind(str, Enum):
    """Kind of text artifact sent to an LLM."""

    SYSTEM_PROMPT = "system_prompt"
    ROLE_DESCRIPTION = "role_description"
    TOOL_DESCRIPTION = "tool_description"
    ORCHESTRATOR_PROMPT = "orchestrator"


@dataclass(frozen=True)
class ContextArtifact:
    """A text artifact sent to an LLM that can be optimized.

    Attributes:
        id: Dot-separated unique ID (e.g., "role.editor.system_prompt").
        kind: The type of artifact.
        content: Current text content.
        source_location: File path + optional line range.
        metadata: Extra context (domain, tags, role_name, etc.).
    """

    id: str
    kind: ArtifactKind
    content: str
    source_location: str
    metadata: dict[str, Any]


class ArtifactExtractor:
    """Extracts optimizable text artifacts from AVE code structures."""

    def extract_from_roles(self, roles: Sequence) -> list[ContextArtifact]:
        """Extract system_prompt and description from each AgentRole.

        Produces two artifacts per role:
        - role.<name>.system_prompt (kind=SYSTEM_PROMPT)
        - role.<name>.description (kind=ROLE_DESCRIPTION)
        """
        artifacts: list[ContextArtifact] = []
        for role in roles:
            name = role.name
            artifacts.append(
                ContextArtifact(
                    id=f"role.{name}.system_prompt",
                    kind=ArtifactKind.SYSTEM_PROMPT,
                    content=role.system_prompt,
                    source_location="src/ave/agent/roles.py",
                    metadata={"role_name": name},
                )
            )
            artifacts.append(
                ContextArtifact(
                    id=f"role.{name}.description",
                    kind=ArtifactKind.ROLE_DESCRIPTION,
                    content=role.description,
                    source_location="src/ave/agent/roles.py",
                    metadata={"role_name": name},
                )
            )
        return artifacts

    def extract_from_registry(self, registry) -> list[ContextArtifact]:
        """Extract full docstrings from registered tools.

        Produces one artifact per tool:
        - tool.<name>.description (kind=TOOL_DESCRIPTION)
        """
        artifacts: list[ContextArtifact] = []
        for name, info in registry._tools.items():
            func = info["func"]
            docstring = inspect.getdoc(func) or ""
            domain = info["domain"]
            tags = tuple(info.get("tags", []))
            artifacts.append(
                ContextArtifact(
                    id=f"tool.{name}.description",
                    kind=ArtifactKind.TOOL_DESCRIPTION,
                    content=docstring,
                    source_location=f"{inspect.getfile(func)}",
                    metadata={"domain": domain, "tags": tags},
                )
            )
        return artifacts

    def extract_all(
        self,
        roles: Sequence | None = None,
        registry=None,
    ) -> list[ContextArtifact]:
        """Extract all artifacts from all provided sources."""
        artifacts: list[ContextArtifact] = []
        if roles:
            artifacts.extend(self.extract_from_roles(roles))
        if registry is not None:
            artifacts.extend(self.extract_from_registry(registry))
        return artifacts
