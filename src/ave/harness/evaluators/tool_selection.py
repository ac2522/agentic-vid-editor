"""Pure tool-selection evaluator — decides pass/fail from a called-tool list.

Separated from Inspect AI plumbing for unit-test simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ave.harness.schema import PlanExpected


@dataclass(frozen=True)
class Verdict:
    """Evaluation verdict."""

    passed: bool
    reason: str


def evaluate_plan(
    called_tools: Sequence[str],
    expected: PlanExpected,
) -> Verdict:
    """Decide whether a sequence of tool-call names satisfies the plan's expectations.

    Rules applied in order:
    1. If no tools were called AND irrelevance_allowed is True -> pass.
    2. If no tools were called AND irrelevance_allowed is False -> fail.
    3. Any tool in ``tools_forbidden`` that was called -> fail.
    4. Every tool in ``all_of`` must appear at least once -> else fail.
    5. If ``any_of`` is non-empty, at least one member must appear -> else fail.
    6. Otherwise -> pass.
    """
    called = list(called_tools)

    if not called:
        if expected.irrelevance_allowed:
            return Verdict(True, "irrelevance allowed; agent called no tools")
        return Verdict(False, "agent called no tools, but irrelevance_allowed=False")

    forbidden_hits = [t for t in expected.tools_forbidden if t in called]
    if forbidden_hits:
        return Verdict(False, f"forbidden tools invoked: {forbidden_hits}")

    missing_all = [t for t in expected.tools_required.all_of if t not in called]
    if missing_all:
        return Verdict(False, f"missing required (all_of): {missing_all}")

    if expected.tools_required.any_of:
        hits = [t for t in expected.tools_required.any_of if t in called]
        if not hits:
            return Verdict(
                False,
                f"none of the any_of tools called: expected one of "
                f"{list(expected.tools_required.any_of)}",
            )

    return Verdict(True, "plan satisfies all constraints")
