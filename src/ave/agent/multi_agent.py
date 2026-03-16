"""Multi-agent orchestrator — role-based delegation for specialized agents.

Wraps the existing Orchestrator with domain-aware agent definitions.
The LLM decides which subagent to use based on AgentDefinition descriptions;
this module does NOT implement explicit routing logic.
"""

from __future__ import annotations

from ave.agent.session import EditingSession
from ave.agent.orchestrator import Orchestrator
from ave.agent.roles import AgentRole, ALL_ROLES


class MultiAgentOrchestrator:
    """Orchestrator that delegates to role-based subagents.

    Wraps the existing Orchestrator with domain-aware agent definitions.
    Does NOT do explicit routing — the LLM decides which subagent to use
    based on AgentDefinition descriptions.
    """

    def __init__(
        self,
        session: EditingSession,
        roles: list[AgentRole] | None = None,
    ) -> None:
        self._session = session
        self._roles = tuple(roles) if roles else ALL_ROLES
        self._base_orchestrator = Orchestrator(session)

    @property
    def session(self) -> EditingSession:
        return self._session

    @property
    def roles(self) -> tuple[AgentRole, ...]:
        return self._roles

    @property
    def base_orchestrator(self) -> Orchestrator:
        return self._base_orchestrator

    def get_agent_definitions(self) -> dict[str, dict]:
        """Generate Claude Agent SDK AgentDefinition-compatible dicts for each role.

        Returns plain dicts (no SDK dependency) with keys: description, prompt, model.
        """
        return {
            role.name: {
                "description": role.description,
                "prompt": role.system_prompt,
            }
            for role in self._roles
        }

    def get_system_prompt(self) -> str:
        """System prompt that describes available specialist agents and the meta-tool workflow."""
        role_lines = "\n".join(f"- {role.name}: {role.description}" for role in self._roles)
        return (
            "You are a supervising AI video editor coordinating specialist agents.\n\n"
            f"Available specialists:\n{role_lines}\n\n"
            "Each specialist has access to domain-specific tools. "
            "Delegate tasks to the appropriate specialist based on the request.\n\n"
            "Tool discovery workflow:\n"
            "1. Use search_tools to find relevant tools by keyword or domain.\n"
            "2. Use get_tool_schema to inspect parameters before calling.\n"
            "3. Use call_tool to execute the tool with validated parameters.\n\n"
            "All timestamps are in nanoseconds. 1 second = 1,000,000,000 ns."
        )

    def get_role_tools(self, role: AgentRole) -> list[str]:
        """Get tool names accessible to a specific role (filtered by role.domains)."""
        tool_names: list[str] = []
        for domain in role.domains:
            summaries = self._session.registry.search_tools(domain=domain)
            for s in summaries:
                if s.name not in tool_names:
                    tool_names.append(s.name)
        return tool_names

    def get_role_for_domain(self, domain: str) -> AgentRole | None:
        """Find which role handles a given domain. Returns None if no role covers it."""
        for role in self._roles:
            if domain in role.domains:
                return role
        return None
