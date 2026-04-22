"""Plan-level solver — runs the agent with non-executing tool stubs.

The agent chooses tools from the scenario's allowed-agent domains; the
stubs simply record that the tool was invoked. The scorer then reads
``state.metadata["called_tools"]`` to decide pass/fail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inspect_ai.solver import Generate, Solver, TaskState, solver

if TYPE_CHECKING:
    pass


def extract_tool_calls(state: TaskState) -> list[str]:
    """Walk the conversation and collect the names of invoked tools, in order."""
    names: list[str] = []
    for msg in state.messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            fn = getattr(tc, "function", None) or getattr(tc, "name", None)
            if fn:
                names.append(str(fn))
    return names


@solver
def plan_solver() -> Solver:
    """Drive the agent through a single tool-planning turn.

    The scenario's declared ``allowed_agents`` are informational at this
    rung; we leave tool wiring to Inspect AI's configured model harness.
    On return, ``state.metadata["called_tools"]`` holds the invocation
    sequence for the scorer.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        state.metadata = dict(state.metadata or {})
        state.metadata["called_tools"] = extract_tool_calls(state)
        return state

    return solve
