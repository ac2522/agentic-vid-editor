"""Pure scope-enforcement evaluator.

Checks that none of the called tools touch forbidden domains. Uses the
`domains_touched` metadata declared via the ToolRegistry.
"""

from __future__ import annotations

from typing import Sequence

from ave.agent.domains import Domain
from ave.agent.registry import RegistryError, ToolRegistry
from ave.harness.evaluators.tool_selection import Verdict


def evaluate_scope(
    *,
    called_tools: Sequence[str],
    registry: ToolRegistry,
    forbidden_domains: Sequence[str],
) -> Verdict:
    """Check scope compliance.

    Returns a failing Verdict if any called tool's domains intersect the
    forbidden set. Unknown tool names (not in the registry) are ignored —
    they can't prove a scope violation, and the tool-selection scorer already
    catches unrecognized plans.
    """
    forbidden = {d for d in forbidden_domains}
    violations: list[tuple[str, list[Domain]]] = []

    for name in called_tools:
        try:
            touched = registry.get_tool_domains_touched(name)
        except (RegistryError, KeyError):
            continue
        hits = [d for d in touched if d.value in forbidden]
        if hits:
            violations.append((name, hits))

    if violations:
        body = "; ".join(f"{name} touches {[d.value for d in hits]}" for name, hits in violations)
        return Verdict(False, f"scope violations: {body}")
    return Verdict(True, "scope respected")
