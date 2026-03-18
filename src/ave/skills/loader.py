"""Skill loader — registers metadata, matches intents, loads body on demand."""

from __future__ import annotations

import re

from ave.skills.discovery import SkillMeta

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


class SkillLoader:
    """Registers skill metadata, matches intents, loads body on demand."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillMeta] = {}

    def register(self, meta: SkillMeta) -> None:
        self._skills[meta.name] = meta

    def get(self, name: str) -> SkillMeta | None:
        return self._skills.get(name)

    def load_body(self, meta: SkillMeta) -> str:
        """Load and return the skill body (frontmatter stripped)."""
        text = meta.path.read_text()
        return _FRONTMATTER_RE.sub("", text).strip()

    def match(self, intent: str, limit: int = 5) -> list[SkillMeta]:
        """Match an intent string against skill triggers and descriptions."""
        intent_lower = intent.lower()
        scored: list[tuple[float, SkillMeta]] = []

        for meta in self._skills.values():
            score = 0.0
            for trigger in meta.triggers:
                if trigger.lower() in intent_lower:
                    score += 2.0
            for word in meta.description.lower().split():
                if len(word) > 3 and word in intent_lower:
                    score += 0.5
            if score > 0:
                scored.append((score, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [meta for _, meta in scored[:limit]]
