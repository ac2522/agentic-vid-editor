"""Agent orchestrator — bridges Claude Agent SDK with AVE tool registry.

Provides 3 meta-tools for progressive tool discovery:
- search_tools: find tools by keyword/domain
- get_tool_schema: get full parameter schema
- call_tool: execute a tool

The orchestrator generates a system prompt with domain summaries
and handles tool call routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ave.agent.session import EditingSession


class OrchestratorError(Exception):
    """Raised when orchestrator operations fail."""


@dataclass(frozen=True)
class MetaToolDef:
    """Definition of a meta-tool exposed to the agent."""

    name: str
    description: str
    parameters: dict


class Orchestrator:
    """Bridges the agent LLM with the AVE tool registry.

    Exposes 3 meta-tools for progressive discovery.
    Handles tool call routing and result formatting.
    """

    def __init__(self, session: EditingSession) -> None:
        self._session = session
        self._turn_count = 0

    @property
    def session(self) -> EditingSession:
        return self._session

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def get_system_prompt(self) -> str:
        """Generate system prompt with domain summaries."""
        domains = self._session.registry.list_domains()
        domain_lines = "\n".join(f"- {d['domain']} ({d['count']} tools)" for d in domains)
        return (
            "You are an AI video editor. You have access to editing tools organized by domain.\n\n"
            f"Available domains:\n{domain_lines}\n\n"
            "Use search_tools to find tools, get_tool_schema for details, call_tool to execute.\n"
            "All timestamps are in nanoseconds. 1 second = 1,000,000,000 ns."
        )

    def get_meta_tools(self) -> list[MetaToolDef]:
        """Return the 3 meta-tool definitions for the agent."""
        return [
            MetaToolDef(
                name="search_tools",
                description="Search for available editing tools by keyword and/or domain.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search keyword",
                        },
                        "domain": {
                            "type": "string",
                            "description": "Filter by domain (optional)",
                        },
                    },
                },
            ),
            MetaToolDef(
                name="get_tool_schema",
                description="Get full parameter schema for a specific tool.",
                parameters={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Tool name to look up",
                        },
                    },
                    "required": ["tool_name"],
                },
            ),
            MetaToolDef(
                name="call_tool",
                description="Execute a tool with the given parameters.",
                parameters={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Tool to execute",
                        },
                        "params": {
                            "type": "object",
                            "description": "Tool parameters",
                        },
                    },
                    "required": ["tool_name", "params"],
                },
            ),
        ]

    def handle_tool_call(self, tool_name: str, arguments: dict) -> str:
        """Route a meta-tool call and return formatted result."""
        self._turn_count += 1

        try:
            if tool_name == "search_tools":
                results = self._session.search_tools(
                    query=arguments.get("query", ""),
                    domain=arguments.get("domain"),
                )
                return self._format_search_results(results)

            elif tool_name == "get_tool_schema":
                schema = self._session.get_tool_schema(arguments["tool_name"])
                return self._format_schema(schema)

            elif tool_name == "call_tool":
                result = self._session.call_tool(
                    arguments["tool_name"],
                    arguments.get("params", {}),
                )
                return self._format_tool_result(arguments["tool_name"], result)

            else:
                return f"Error: Unknown meta-tool '{tool_name}'"

        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def _format_search_results(self, results) -> str:
        """Format search results for LLM consumption."""
        if not results:
            return "No tools found."
        lines = [f"- {r.name} [{r.domain}]: {r.description}" for r in results]
        return f"Found {len(results)} tool(s):\n" + "\n".join(lines)

    def _format_schema(self, schema) -> str:
        """Format tool schema for LLM consumption."""
        params_str = "\n".join(
            f"  - {p.name}: {p.type_str} {'(required)' if p.required else f'(default: {p.default})'}"
            for p in schema.params
        )
        deps = ""
        if schema.requires:
            deps = f"\nRequires: {', '.join(schema.requires)}"
        if schema.provides:
            deps += f"\nProvides: {', '.join(schema.provides)}"
        return f"Tool: {schema.name} [{schema.domain}]\n{schema.description}\n\nParameters:\n{params_str}{deps}"

    def _format_tool_result(self, tool_name: str, result: Any) -> str:
        """Format tool execution result for LLM consumption."""
        return f"Tool '{tool_name}' executed successfully.\nResult: {result}"
