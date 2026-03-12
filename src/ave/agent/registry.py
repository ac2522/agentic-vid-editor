"""Tool Registry — progressive discovery for agent tool access.

Provides 3 meta-functions:
- search_tools: find tools by keyword/domain (~20-50 tokens per result)
- get_tool_schema: full parameter schema for one tool
- call_tool: execute a tool with parameter validation
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, get_type_hints

from ave.agent.dependencies import DependencyGraph, SessionState


class RegistryError(Exception):
    """Raised when registry operations fail."""


class PrerequisiteError(Exception):
    """Raised when tool prerequisites are not met."""


@dataclass(frozen=True)
class ToolSummary:
    """Compact tool summary for search results (~20-50 tokens)."""

    name: str
    domain: str
    description: str  # First line of docstring
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParamInfo:
    """Parameter metadata extracted from type hints."""

    name: str
    type_str: str
    required: bool
    default: Any = None


@dataclass(frozen=True)
class ToolSchema:
    """Full tool schema loaded on demand."""

    name: str
    domain: str
    description: str  # Full docstring
    params: list[ParamInfo]
    requires: list[str]
    provides: list[str]


def _first_line(docstring: str | None) -> str:
    """Extract first non-empty line from a docstring."""
    if not docstring:
        return ""
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _is_pydantic_model(annotation: type) -> bool:
    """Check if an annotation is a Pydantic BaseModel subclass."""
    try:
        from pydantic import BaseModel

        return isinstance(annotation, type) and issubclass(annotation, BaseModel)
    except ImportError:
        return False


def _type_name(annotation: Any) -> str:
    """Convert a type annotation to a string name."""
    if annotation is inspect.Parameter.empty:
        return "Any"
    if annotation is Path:
        return "Path"
    if isinstance(annotation, type):
        return annotation.__name__
    # Handle generic types like list[int], dict[str, Any]
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        return getattr(origin, "__name__", str(origin))
    return str(annotation)


def _extract_params(func: Callable) -> list[ParamInfo]:
    """Extract parameter info from function signature and type hints."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    params: list[ParamInfo] = []

    for name, param in sig.parameters.items():
        if name == "return":
            continue

        annotation = hints.get(name, param.annotation)

        # If the parameter is a Pydantic model, expand its fields
        if _is_pydantic_model(annotation):
            return _extract_pydantic_params(annotation)

        type_str = _type_name(annotation)
        has_default = param.default is not inspect.Parameter.empty
        default = param.default if has_default else None

        params.append(
            ParamInfo(
                name=name,
                type_str=type_str,
                required=not has_default,
                default=default,
            )
        )

    return params


def _extract_pydantic_params(model_cls: type) -> list[ParamInfo]:
    """Extract parameter info from a Pydantic model's fields."""
    params: list[ParamInfo] = []
    for field_name, field_info in model_cls.model_fields.items():
        type_str = _type_name(field_info.annotation)
        required = field_info.is_required()
        default = field_info.default if not required else None
        params.append(
            ParamInfo(name=field_name, type_str=type_str, required=required, default=default)
        )
    return params


class ToolRegistry:
    """Registry for progressive tool discovery."""

    def __init__(self) -> None:
        self._tools: dict[str, dict] = {}  # name -> {func, domain, requires, provides}
        self._dep_graph = DependencyGraph()

    def tool(
        self,
        domain: str,
        requires: list[str] | None = None,
        provides: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> Callable:
        """Decorator to register a tool function."""
        req = requires or []
        prov = provides or []
        tag_list = tags or []

        def decorator(func: Callable) -> Callable:
            name = func.__name__
            if name in self._tools:
                raise RegistryError(f"Tool already registered: {name}")
            self._tools[name] = {
                "func": func,
                "domain": domain,
                "requires": req,
                "provides": prov,
                "tags": tag_list,
            }
            self._dep_graph.add_tool(name, requires=req, provides=prov)
            return func

        return decorator

    def search_tools(self, query: str = "", domain: str | None = None) -> list[ToolSummary]:
        """Search tools by keyword and/or domain. Returns compact summaries."""
        results: list[tuple[int, ToolSummary]] = []
        query_lower = query.lower()
        query_words = query_lower.split() if query_lower else []

        for name, info in self._tools.items():
            # Domain filter
            if domain is not None and info["domain"] != domain:
                continue

            func = info["func"]
            first_line = _first_line(func.__doc__)
            tags = info.get("tags", [])
            searchable = f"{name} {first_line} {' '.join(tags)}".lower()
            tags_tuple = tuple(tags)

            # If no query, match all (domain-only search)
            if not query_words:
                results.append((0, ToolSummary(name=name, domain=info["domain"], description=first_line, tags=tags_tuple)))
                continue

            # Score by word matches
            score = sum(1 for w in query_words if w in searchable)
            if score > 0:
                results.append((score, ToolSummary(name=name, domain=info["domain"], description=first_line, tags=tags_tuple)))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [summary for _, summary in results]

    def get_tool_schema(self, tool_name: str) -> ToolSchema:
        """Get full schema for a tool. Raises RegistryError if not found."""
        info = self._tools.get(tool_name)
        if info is None:
            raise RegistryError(f"Tool not found: {tool_name}")

        func = info["func"]
        params = _extract_params(func)
        docstring = inspect.getdoc(func) or ""

        return ToolSchema(
            name=tool_name,
            domain=info["domain"],
            description=docstring,
            params=params,
            requires=info["requires"],
            provides=info["provides"],
        )

    def call_tool(
        self,
        tool_name: str,
        params: dict,
        session_state: SessionState | None = None,
    ) -> Any:
        """Execute a tool. Validates prerequisites if session_state provided."""
        info = self._tools.get(tool_name)
        if info is None:
            raise RegistryError(f"Tool not found: {tool_name}")

        # Check prerequisites
        if session_state is not None:
            missing = self._dep_graph.check_prerequisites(tool_name, session_state.provisions)
            if missing:
                raise PrerequisiteError(
                    f"Prerequisites not met for '{tool_name}': missing {missing}"
                )

        # Execute
        result = info["func"](**params)

        # Track provisions
        if session_state is not None and info["provides"]:
            session_state.add(*info["provides"])

        return result

    def list_domains(self) -> list[dict[str, Any]]:
        """Return list of domains with tool counts."""
        counts: dict[str, int] = {}
        for info in self._tools.values():
            d = info["domain"]
            counts[d] = counts.get(d, 0) + 1
        return [{"domain": d, "count": c} for d, c in sorted(counts.items())]

    def get_tool_provisions(self, tool_name: str) -> list[str]:
        """Get the provisions list for a tool. Raises RegistryError if not found."""
        info = self._tools.get(tool_name)
        if info is None:
            raise RegistryError(f"Tool not found: {tool_name}")
        return list(info["provides"])

    @property
    def tool_count(self) -> int:
        return len(self._tools)
