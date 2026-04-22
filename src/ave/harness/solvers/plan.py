"""Plan-level solver — runs the agent with non-executing tool stubs.

The agent chooses tools from the scenario's declared tool universe (the
union of ``tools_required.all_of``, ``tools_required.any_of`` and
``tools_forbidden``); the stubs simply record that a tool was invoked and
return a short OK string. The scorer then reads
``state.metadata["called_tools"]`` to decide pass/fail.

Without wired tools, real LLMs (Claude, Gemini, ...) see no tool universe
on ``state.tools`` and cannot produce tool calls; only the mockllm backend
fabricates tool_call structures without tools being declared.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver, use_tools
from inspect_ai.tool import Tool, ToolDef, ToolParams
from inspect_ai.util import JSONSchema

from ave.agent.registry import ParamInfo, RegistryError, ToolRegistry, ToolSchema
from ave.agent.session import EditingSession
from ave.harness.schema import Scenario


_session_for_registry: EditingSession | None = None


def _shared_registry() -> ToolRegistry:
    """Lazily build a full-registry EditingSession to look up tool schemas."""
    global _session_for_registry
    if _session_for_registry is None:
        _session_for_registry = EditingSession()
    return _session_for_registry.registry


# Minimal Python-type-name -> JSON-Schema-type-name mapping. Unknown types
# fall back to "string" — stubs don't execute anything, so accuracy here
# only matters so far as giving the model a plausible signature.
_PY_TO_JSON_TYPE: dict[str, str] = {
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "str": "string",
    "list": "array",
    "tuple": "array",
    "dict": "object",
    "bytes": "string",
    "None": "null",
    "NoneType": "null",
}


def _json_type_for(param: ParamInfo) -> str:
    # Strip Optional[...] / typing wrappers by taking the leading identifier.
    head = (param.type_str or "").strip().split("[", 1)[0].split("|", 1)[0].strip()
    return _PY_TO_JSON_TYPE.get(head, "string")


def _params_for(schema: ToolSchema | None) -> ToolParams:
    if schema is None or not schema.params:
        return ToolParams()
    props: dict[str, JSONSchema] = {}
    required: list[str] = []
    for p in schema.params:
        props[p.name] = JSONSchema(type=_json_type_for(p), description=p.name)
        if p.required:
            required.append(p.name)
    return ToolParams(properties=props, required=required)


def _make_stub_tool(name: str, registry: ToolRegistry) -> Tool:
    """Build a non-executing stub tool that records invocation.

    Looks up the tool's real schema in ``registry`` to produce a plausible
    parameter list, so a real LLM can bind arguments without confusion.
    Unknown tools (e.g. names only asserted in ``tools_forbidden``) get an
    empty-parameters stub.
    """
    try:
        schema: ToolSchema | None = registry.get_tool_schema(name)
    except RegistryError:
        schema = None

    description = (
        schema.description if schema and schema.description else f"{name} (stub)"
    ).strip()
    parameters = _params_for(schema)

    async def execute(**kwargs: Any) -> str:
        return f"[stub] {name} invoked"

    tool_def = ToolDef(
        execute,
        name=name,
        description=description,
        parameters=parameters,
    )
    return tool_def.as_tool()


def _tools_for_scenario(scenario: Scenario, registry: ToolRegistry | None = None) -> list[Tool]:
    """Build the tool universe for the scenario's plan expectations.

    Union of ``tools_required.all_of``, ``tools_required.any_of`` and
    ``tools_forbidden``. Forbidden tools are included so the model *can*
    call them and the scorer can catch the violation.
    """
    plan = scenario.expected.plan
    if plan is None:
        return []
    names: set[str] = set()
    names.update(plan.tools_required.all_of)
    names.update(plan.tools_required.any_of)
    names.update(plan.tools_forbidden)
    if not names:
        return []
    reg = registry if registry is not None else _shared_registry()
    return [_make_stub_tool(n, reg) for n in sorted(names)]


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

    Builds non-executing stubs for the scenario's declared tool universe
    and wires them via ``use_tools`` before calling ``generate``. On
    return, ``state.metadata["called_tools"]`` holds the invocation
    sequence for the scorer.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        scenario: Scenario | None = None
        if state.metadata:
            scenario = state.metadata.get("scenario")
        tools = _tools_for_scenario(scenario) if scenario is not None else []
        if tools:
            state = await use_tools(tools)(state, generate)
        state = await generate(state)
        state.metadata = dict(state.metadata or {})
        state.metadata["called_tools"] = extract_tool_calls(state)
        return state

    return solve
