"""Tests for Inspect AI eval log → harness feedback extraction."""
from pathlib import Path
import pytest
from tests.conftest import requires_inspect


@requires_inspect
def test_messages_to_llm_response_extracts_tool_calls():
    """ChatMessageAssistant with tool_calls should map into LLMResponse.tool_calls."""
    from inspect_ai.model import ChatMessageAssistant
    from inspect_ai.tool import ToolCall
    from ave.harness.feedback.eval_log import messages_to_llm_response

    msgs = [
        ChatMessageAssistant(
            content="planning...",
            tool_calls=[
                ToolCall(id="1", function="trim", arguments={"start": 0}),
                ToolCall(id="2", function="concat", arguments={}),
            ],
        ),
    ]
    resp = messages_to_llm_response(msgs)
    assert resp.text == "planning..."
    names = [tc.name for tc in resp.tool_calls]
    assert names == ["trim", "concat"]
    assert resp.tool_calls[0].arguments == {"start": 0}


@requires_inspect
def test_messages_to_llm_response_no_tool_calls():
    from inspect_ai.model import ChatMessageAssistant
    from ave.harness.feedback.eval_log import messages_to_llm_response

    resp = messages_to_llm_response([ChatMessageAssistant(content="just text")])
    assert resp.text == "just text"
    assert resp.tool_calls == ()


@requires_inspect
def test_messages_to_llm_response_aggregates_text_across_assistant_msgs():
    from inspect_ai.model import ChatMessageAssistant
    from ave.harness.feedback.eval_log import messages_to_llm_response

    msgs = [
        ChatMessageAssistant(content="part one"),
        ChatMessageAssistant(content="part two"),
    ]
    resp = messages_to_llm_response(msgs)
    assert "part one" in resp.text
    assert "part two" in resp.text


@requires_inspect
def test_eval_log_to_feedback_rows_with_real_log(tmp_path: Path):
    """End-to-end: run a tiny task with mockllm, then parse the resulting log."""
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import get_model
    from inspect_ai.scorer import Score, Scorer, Target, scorer
    from inspect_ai.solver import Generate, Solver, TaskState, solver

    from ave.harness.feedback.eval_log import eval_log_to_feedback_rows

    @solver
    def stub_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state = await generate(state)
            state.metadata = dict(state.metadata or {})
            state.metadata["called_tools"] = ["a", "b"]
            return state
        return solve

    @scorer(metrics=[])
    def stub_scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return Score(value=1, explanation="all good", metadata={"rule": "ok"})
        return score

    log_dir = tmp_path / "logs"
    inspect_eval(
        Task(
            dataset=MemoryDataset([Sample(id="x", input="hi", target="x", metadata={"task_meta": True})]),
            solver=stub_solver(),
            scorer=stub_scorer(),
        ),
        model=get_model("mockllm/mock"),
        display="plain",
        log_dir=str(log_dir),
    )
    log_file = next(log_dir.glob("*.eval"))
    rows = eval_log_to_feedback_rows(log_file)
    assert len(rows) >= 1
    row = rows[0]
    assert row.sample_id == "x"
    assert row.scorer_name in ("stub_scorer", "score")  # name may vary
    assert row.passed is True
    assert "all good" in row.reason
    assert row.called_tools == ["a", "b"]


@requires_inspect
def test_summarize_failures_filters_passed(tmp_path: Path):
    """summarize_failures() returns only failing rows."""
    from ave.harness.feedback.eval_log import FeedbackRow, summarize_failures

    rows = [
        FeedbackRow(
            sample_id="a", task="t", scorer_name="s", passed=True,
            verdict_rule="ok", reason="fine",
            called_tools=["x"], expected_tools=("x",),
        ),
        FeedbackRow(
            sample_id="b", task="t", scorer_name="s", passed=False,
            verdict_rule="forbidden_tools", reason="forbidden invoked",
            called_tools=["bad"], expected_tools=("good",),
        ),
    ]
    failures = summarize_failures(rows)
    assert len(failures) == 1
    assert failures[0].sample_id == "b"
