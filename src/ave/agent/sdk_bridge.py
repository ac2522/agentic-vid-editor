"""SDK integration bridge — generates Claude Agent SDK-compatible configuration.

Produces plain dicts compatible with Claude Agent SDK structures
without importing the SDK itself. This allows AVE to prepare agent
configurations that can be consumed by the SDK at runtime.
"""

from __future__ import annotations

from ave.agent.session import EditingSession
from ave.agent.multi_agent import MultiAgentOrchestrator
from ave.agent.roles import AgentRole


def role_to_agent_definition(
    role: AgentRole,
    session: EditingSession,
    model: str = "opus",
) -> dict:
    """Convert an AgentRole to a Claude Agent SDK AgentDefinition-compatible dict.

    Returns a dict with keys: description, prompt, model.
    Does NOT import claude-agent-sdk — produces plain dicts.
    """
    orch = MultiAgentOrchestrator(session, roles=[role])
    tools = orch.get_role_tools(role)

    return {
        "description": role.description,
        "prompt": role.system_prompt,
        "model": model,
        "tools": tools,
    }


def create_ave_agent_options(
    session: EditingSession,
    roles: list[AgentRole] | None = None,
    model: str = "opus",
) -> dict:
    """Create configuration dict compatible with Claude Agent SDK ClaudeAgentOptions.

    Returns a dict with keys: agents, system_prompt, allowed_tools.
    Does NOT import claude-agent-sdk — produces plain dicts.
    """
    orch = MultiAgentOrchestrator(session, roles=roles)

    agents = [role_to_agent_definition(role, session, model=model) for role in orch.roles]

    # Collect all tools across all roles
    all_tools: list[str] = []
    for role in orch.roles:
        for tool_name in orch.get_role_tools(role):
            if tool_name not in all_tools:
                all_tools.append(tool_name)

    return {
        "agents": agents,
        "system_prompt": orch.get_system_prompt(),
        "allowed_tools": all_tools,
    }
