"""Tool Registry — progressive discovery for agent tool access.

Provides 3 meta-functions:
- search_tools: find tools by keyword/domain (~20-50 tokens per result)
- get_tool_schema: full parameter schema for one tool
- call_tool: execute a tool with parameter validation

Supports namespace scoping (e.g., "ave:editing.trim", "user:vfx.segment").
Backward-compatible: short names ("trim") resolve when unambiguous.
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
    namespace: str = "ave"


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
    """Registry for progressive tool discovery with namespace support."""

    def __init__(self) -> None:
        # Primary storage: short_name -> tool info dict
        self._tools: dict[str, dict] = {}
        # Short name -> list of full namespaced keys that share it
        self._short_names: dict[str, list[str]] = {}
        # Full namespaced key -> short name
        self._ns_to_short: dict[str, str] = {}
        self._dep_graph = DependencyGraph()

    def _resolve_name(self, name: str) -> str:
        """Resolve a tool name to its internal storage key.

        Accepts full namespaced names (e.g., "ave:editing.trim") or
        short names (e.g., "trim"). Raises on ambiguous short names.
        """
        # Try as namespaced key first
        storage = self._ns_to_short.get(name)
        if storage is not None:
            return storage

        # Check short name ambiguity
        if name in self._short_names:
            full_names = self._short_names[name]
            if len(full_names) == 1:
                return self._ns_to_short[full_names[0]]
            raise KeyError(
                f"Tool name '{name}' is ambiguous — matches: {full_names}. "
                "Use the full namespaced name."
            )

        # Direct match on storage key (backward compat edge case)
        if name in self._tools:
            return name

        raise RegistryError(f"Tool not found: {name}")

    def tool(
        self,
        domain: str,
        requires: list[str] | None = None,
        provides: list[str] | None = None,
        tags: list[str] | None = None,
        modifies_timeline: bool = False,
        namespace: str = "ave",
    ) -> Callable:
        """Decorator to register a tool function.

        Args:
            domain: Tool domain (e.g., "editing", "color", "vfx").
            namespace: Namespace scope (default "ave"). Plugins use "user",
                community plugins use "community".
            modifies_timeline: If True, this tool modifies the GES timeline
                and should trigger end-of-turn verification.
        """
        req = requires or []
        prov = provides or []
        tag_list = tags or []

        def decorator(func: Callable) -> Callable:
            short_name = func.__name__
            full_name = f"{namespace}:{domain}.{short_name}"

            if full_name in self._ns_to_short:
                raise RegistryError(f"Tool already registered: {full_name}")
            # For backward compat: if short name exists AND no namespace
            # collision, allow it. Otherwise, use unique short name.
            storage_key = short_name
            if storage_key in self._tools:
                # Short name collision — use full name as storage key
                storage_key = full_name

            self._tools[storage_key] = {
                "func": func,
                "domain": domain,
                "requires": req,
                "provides": prov,
                "tags": tag_list,
                "modifies_timeline": modifies_timeline,
                "namespace": namespace,
                "full_name": full_name,
                "short_name": short_name,
            }
            self._ns_to_short[full_name] = storage_key
            self._short_names.setdefault(short_name, []).append(full_name)
            self._dep_graph.add_tool(storage_key, requires=req, provides=prov)
            return func

        return decorator

    def register_stub(
        self,
        name: str,
        domain: str,
        summary: str,
        namespace: str = "user",
        plugin_name: str | None = None,
    ) -> None:
        """Register a tool stub (searchable, not yet callable).

        Used by PluginLoader to register plugin tools before code is loaded.
        """
        full_name = f"{namespace}:{domain}.{name}"
        storage_key = name if name not in self._tools else full_name

        self._tools[storage_key] = {
            "func": None,
            "domain": domain,
            "requires": [],
            "provides": [],
            "tags": [],
            "modifies_timeline": False,
            "namespace": namespace,
            "full_name": full_name,
            "short_name": name,
            "stub": True,
            "plugin_name": plugin_name,
            "description": summary,
        }
        self._ns_to_short[full_name] = storage_key
        self._short_names.setdefault(name, []).append(full_name)

    def search_tools(self, query: str = "", domain: str | None = None) -> list[ToolSummary]:
        """Search tools by keyword and/or domain. Returns compact summaries."""
        results: list[tuple[int, ToolSummary]] = []
        query_lower = query.lower()
        query_words = query_lower.split() if query_lower else []

        for storage_key, info in self._tools.items():
            # Domain filter
            if domain is not None and info["domain"] != domain:
                continue

            func = info.get("func")
            if func is not None:
                first_line = _first_line(func.__doc__)
            else:
                first_line = info.get("description", "")

            tags = info.get("tags", [])
            short_name = info.get("short_name", storage_key)
            ns = info.get("namespace", "ave")
            searchable = f"{short_name} {first_line} {' '.join(tags)}".lower()
            tags_tuple = tuple(tags)

            summary = ToolSummary(
                name=storage_key,
                domain=info["domain"],
                description=first_line,
                tags=tags_tuple,
                namespace=ns,
            )

            # If no query, match all (domain-only search)
            if not query_words:
                results.append((0, summary))
                continue

            # Score by word matches
            score = sum(1 for w in query_words if w in searchable)
            if score > 0:
                results.append((score, summary))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [summary for _, summary in results]

    def get_tool_schema(self, tool_name: str) -> ToolSchema:
        """Get full schema for a tool. Raises RegistryError if not found."""
        resolved = self._resolve_name(tool_name)
        info = self._tools[resolved]

        func = info.get("func")
        if func is None:
            raise RegistryError(
                f"Tool '{tool_name}' is a stub — plugin not yet loaded"
            )

        params = _extract_params(func)
        docstring = inspect.getdoc(func) or ""

        return ToolSchema(
            name=resolved,
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
        resolved = self._resolve_name(tool_name)
        info = self._tools[resolved]

        func = info.get("func")
        if func is None:
            raise RegistryError(
                f"Tool '{tool_name}' is a stub — plugin not yet loaded"
            )

        # Check prerequisites
        if session_state is not None:
            missing = self._dep_graph.check_prerequisites(
                resolved, session_state.provisions
            )
            if missing:
                raise PrerequisiteError(
                    f"Prerequisites not met for '{tool_name}': missing {missing}"
                )

        # Execute
        result = func(**params)

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
        resolved = self._resolve_name(tool_name)
        return list(self._tools[resolved]["provides"])

    def tool_modifies_timeline(self, tool_name: str) -> bool:
        """Check if a tool modifies the timeline (for verification gating)."""
        resolved = self._resolve_name(tool_name)
        return self._tools[resolved].get("modifies_timeline", False)

    @property
    def tool_count(self) -> int:
        return len(self._tools)
