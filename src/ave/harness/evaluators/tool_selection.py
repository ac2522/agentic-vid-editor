"""Pure tool-selection evaluator — decides pass/fail from a called-tool list.

Separated from Inspect AI plumbing for unit-test simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from ave.harness.schema import PlanExpected


# Machine-readable rule tags so downstream scorers/dashboards can aggregate
# verdicts without regexing the free-form ``reason`` string.
VerdictRule = Literal[
    "ok",
    "irrelevance_ok",
    "irrelevance_required",
    "forbidden_tools",
    "missing_all_of",
    "missing_any_of",
    "scope_violation",
    "scope_respected",
]


@dataclass(frozen=True)
class Verdict:
    """Evaluation verdict.

    ``rule`` identifies which branch of the evaluator fired, for
    machine-readable aggregation. ``reason`` is the human-readable form.
    """

    passed: bool
    reason: str
    rule: VerdictRule = "ok"


def evaluate_plan(
    called_tools: Sequence[str],
    expected: PlanExpected,
) -> Verdict:
    """Decide whether a sequence of tool-call names satisfies the plan's expectations.

    Rules applied in order (first match wins):
    1. No tools called AND ``irrelevance_allowed`` is True -> pass (``irrelevance_ok``).
    2. No tools called AND ``irrelevance_allowed`` is False -> fail (``irrelevance_required``).
    3. Any tool in ``tools_forbidden`` was called -> fail (``forbidden_tools``).
       Precedence: forbidden is checked BEFORE all_of, so a tool listed in both
       forbidden and all_of will always fail the forbidden check — scenarios
       that mix these are ill-formed.
    4. Every tool in ``all_of`` must appear at least once -> else fail (``missing_all_of``).
    5. If ``any_of`` is non-empty, at least one member must appear -> else fail (``missing_any_of``).
    6. Otherwise -> pass (``ok``).
    """
    called = list(called_tools)

    if not called:
        if expected.irrelevance_allowed:
            return Verdict(True, "irrelevance allowed; agent called no tools", "irrelevance_ok")
        return Verdict(
            False,
            "agent called no tools, but irrelevance_allowed=False",
            "irrelevance_required",
        )

    forbidden_hits = [t for t in expected.tools_forbidden if t in called]
    if forbidden_hits:
        return Verdict(False, f"forbidden tools invoked: {forbidden_hits}", "forbidden_tools")

    missing_all = [t for t in expected.tools_required.all_of if t not in called]
    if missing_all:
        return Verdict(False, f"missing required (all_of): {missing_all}", "missing_all_of")

    if expected.tools_required.any_of:
        hits = [t for t in expected.tools_required.any_of if t in called]
        if not hits:
            return Verdict(
                False,
                f"none of the any_of tools called: expected one of "
                f"{list(expected.tools_required.any_of)}",
                "missing_any_of",
            )

    return Verdict(True, "plan satisfies all constraints", "ok")
