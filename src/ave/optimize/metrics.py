"""Evaluation metrics for context optimization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar

from ave.optimize.datasets import EvalItem


@dataclass(frozen=True)
class ToolCall:
    """A tool call made by the LLM."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """Structured LLM output for metric scoring."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class MetricResult:
    """Result of a metric evaluation."""

    name: str
    value: float  # 0.0 to 1.0
    reason: str


class ToolSelectionAccuracy:
    """Measures whether the LLM selected the correct tool(s).

    Uses Jaccard similarity between expected and actual tool sets.
    """

    name: ClassVar[str] = "tool_selection_accuracy"

    def score(self, item: EvalItem, response: LLMResponse) -> MetricResult:
        expected = set(item.expected_tools)
        actual = {tc.name for tc in response.tool_calls}

        if not expected and not actual:
            return MetricResult(
                name=self.name,
                value=1.0,
                reason="Both expected and actual tool sets are empty.",
            )

        if not expected or not actual:
            return MetricResult(
                name=self.name,
                value=0.0,
                reason=f"Expected {expected}, got {actual}.",
            )

        intersection = expected & actual
        union = expected | actual
        jaccard = len(intersection) / len(union)
        return MetricResult(
            name=self.name,
            value=jaccard,
            reason=f"Jaccard={jaccard:.2f}: expected={expected}, actual={actual}.",
        )


class ForbiddenToolsPenalty:
    """Hard-penalty for invoking any forbidden tool.

    Returns 1.0 if no forbidden tool was called, 0.0 if any was.
    Reads forbidden tools from ``item.context["tools_forbidden"]``.
    Aligns with harness pass/fail semantics — forbidden = hard fail.
    """

    name: ClassVar[str] = "forbidden_tools_penalty"

    def score(self, item: EvalItem, response: LLMResponse) -> MetricResult:
        forbidden = set(item.context.get("tools_forbidden", []) if item.context else [])
        if not forbidden:
            return MetricResult(
                name=self.name, value=1.0,
                reason="no forbidden constraint",
            )
        actual = {tc.name for tc in response.tool_calls}
        violations = sorted(actual & forbidden)
        if not violations:
            return MetricResult(
                name=self.name, value=1.0,
                reason="no forbidden tools called",
            )
        return MetricResult(
            name=self.name, value=0.0,
            reason=f"forbidden tools invoked: {violations}",
        )


class ConventionCompliance:
    """Measures adherence to AVE conventions in LLM output.

    Checks for:
    - Nanosecond timestamp usage (numbers >= 1_000_000)
    - agent: metadata prefix usage
    - DNxHR/MXF codec references
    """

    name: ClassVar[str] = "convention_compliance"

    _NS_PATTERN = re.compile(r"\b\d{7,}\b")  # Numbers >= 1M likely nanoseconds
    _AGENT_PREFIX = re.compile(r"agent:")
    _CODEC_PATTERN = re.compile(r"DNxHR|MXF|ProRes", re.IGNORECASE)

    def score(self, item: EvalItem, response: LLMResponse) -> MetricResult:
        text = response.text
        checks_found: list[str] = []

        if self._NS_PATTERN.search(text):
            checks_found.append("nanosecond_timestamps")
        if self._AGENT_PREFIX.search(text):
            checks_found.append("agent_prefix")
        if self._CODEC_PATTERN.search(text):
            checks_found.append("codec_reference")

        if not checks_found:
            return MetricResult(
                name=self.name,
                value=0.0,
                reason="No AVE conventions found in output.",
            )

        # Score based on how many convention signals are present
        # Max 3 signals, normalize to 0-1
        value = min(len(checks_found) / 3.0, 1.0)
        return MetricResult(
            name=self.name,
            value=value,
            reason=f"Conventions found: {', '.join(checks_found)}.",
        )
