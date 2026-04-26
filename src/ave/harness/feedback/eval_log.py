"""Parse Inspect AI eval logs from harness runs into feedback rows.

A FeedbackRow captures a single (sample, scorer) verdict in a flat shape
suitable for the optimize/ pipeline. Failures are the primary training
signal — what tools the agent should have called vs what it did call,
plus the verdict rule tag for aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ave.optimize.metrics import LLMResponse, ToolCall


@dataclass(frozen=True)
class FeedbackRow:
    sample_id: str
    task: str
    scorer_name: str
    passed: bool
    verdict_rule: str | None
    reason: str
    called_tools: list[str] = field(default_factory=list)
    expected_tools: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


def messages_to_llm_response(messages) -> LLMResponse:
    """Walk Inspect AI ChatMessage list, build an LLMResponse with text + tool calls.

    Aggregates assistant text across multiple turns; collects every tool_call
    from every assistant message in order.
    """
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role != "assistant":
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content:
            text_parts.append(content)
        elif isinstance(content, list):
            for c in content:
                t = getattr(c, "text", None)
                if t:
                    text_parts.append(t)
        for tc in getattr(msg, "tool_calls", None) or []:
            name = getattr(tc, "function", None) or getattr(tc, "name", None)
            args = getattr(tc, "arguments", None) or {}
            if name:
                tool_calls.append(ToolCall(name=str(name), arguments=dict(args)))
    return LLMResponse(text="\n".join(text_parts), tool_calls=tuple(tool_calls))


def eval_log_to_feedback_rows(log_path: Path) -> list[FeedbackRow]:
    """Read an Inspect AI .eval file and emit one FeedbackRow per (sample, scorer)."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    rows: list[FeedbackRow] = []
    for sample in log.samples or []:
        called_tools = list(sample.metadata.get("called_tools", []) if sample.metadata else [])
        scenario = (sample.metadata or {}).get("scenario")
        expected_tools: tuple[str, ...] = ()
        if scenario is not None and getattr(scenario, "expected", None):
            plan = scenario.expected.plan
            if plan is not None:
                expected_tools = tuple(plan.tools_required.all_of) + tuple(plan.tools_required.any_of)
        task = str(getattr(sample, "input", "") or "")
        for scorer_name, score in (sample.scores or {}).items():
            rule = None
            score_meta = getattr(score, "metadata", None) or {}
            if isinstance(score_meta, dict):
                rule = score_meta.get("rule")
            rows.append(FeedbackRow(
                sample_id=str(sample.id),
                task=task,
                scorer_name=str(scorer_name),
                passed=bool(getattr(score, "value", 0) == 1),
                verdict_rule=str(rule) if rule else None,
                reason=str(getattr(score, "explanation", "") or ""),
                called_tools=called_tools,
                expected_tools=expected_tools,
                metadata=dict(score_meta) if isinstance(score_meta, dict) else {},
            ))
    return rows


def summarize_failures(rows: list[FeedbackRow]) -> list[FeedbackRow]:
    """Filter for failed rows (these become training signal)."""
    return [r for r in rows if not r.passed]
