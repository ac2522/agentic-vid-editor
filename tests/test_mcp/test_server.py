"""Tests for MCP server tool registration and execution."""

from __future__ import annotations

import asyncio

import pytest

from ave.mcp.server import create_mcp_server


def _call(server, tool_name: str, args: dict | None = None):
    """Helper: call tool and extract structured content."""
    result = asyncio.run(server.call_tool(tool_name, args or {}))
    # FastMCP 3.x returns ToolResult with structured_content
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content
    return result


class TestMcpServer:
    def test_server_has_six_tools(self):
        server = create_mcp_server()
        tools = asyncio.run(server.list_tools())
        tool_names = {t.name for t in tools}
        expected = {
            "edit_video",
            "get_job_status",
            "get_project_state",
            "render_preview",
            "ingest_asset",
            "search_tools",
            "call_tool",
        }
        assert tool_names == expected

    def test_search_tools_returns_results(self):
        server = create_mcp_server()
        result = _call(server, "search_tools", {"query": "trim"})
        # FastMCP may wrap list returns in {"result": [...]}
        if isinstance(result, dict) and "result" in result:
            result = result["result"]
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_get_project_state_returns_dict(self):
        server = create_mcp_server()
        result = _call(server, "get_project_state")
        assert "clip_count" in result

    def test_edit_video_returns_result(self):
        server = create_mcp_server()
        result = _call(server, "edit_video", {"instruction": "trim clip 1"})
        assert isinstance(result, dict)
        assert "description" in result

    def test_render_preview_returns_result(self):
        server = create_mcp_server()
        result = _call(server, "render_preview")
        assert "format" in result
        assert result["format"] == "jpeg"

    def test_ingest_asset_returns_result(self):
        server = create_mcp_server()
        result = _call(server, "ingest_asset", {"path": "/tmp/test.mov"})
        assert result["path"] == "/tmp/test.mov"

    def test_call_tool_delegates_to_session(self):
        """call_tool passes through to session — prerequisite errors surface."""
        from fastmcp.exceptions import ToolError

        server = create_mcp_server()
        # Most tools require timeline_loaded, so calling without loading
        # should raise a prerequisite error through FastMCP
        with pytest.raises(ToolError, match="Prerequisites"):
            _call(
                server,
                "call_tool",
                {"name": "trim", "params": {"clip_duration_ns": 10_000, "in_ns": 0, "out_ns": 5_000}},
            )
