"""WebSocket chat protocol and ChatSession for the AVE web UI.

Protocol functions handle message serialisation/deserialisation.
ChatSession bridges the WebSocket with the Anthropic streaming API
for an agentic tool-use loop.
"""

from __future__ import annotations

import asyncio
import json
import logging

logger = logging.getLogger(__name__)


def parse_client_message(raw: str) -> dict:
    """Parse a raw JSON string from the client.

    Returns the parsed dict on success, or ``{"type": "error", "message": "..."}``
    on invalid JSON or missing ``type`` field.
    """
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return {"type": "error", "message": f"Invalid JSON: {exc}"}

    if not isinstance(msg, dict) or "type" not in msg:
        return {"type": "error", "message": "Missing required 'type' field"}

    return msg


def format_text_delta(text: str) -> dict:
    """Format a streaming text chunk for the client."""
    return {"type": "text_delta", "text": text}


def format_tool_start(tool_name: str, tool_id: str) -> dict:
    """Format a tool-execution-started event."""
    return {"type": "tool_start", "tool_name": tool_name, "tool_id": tool_id}


def format_tool_done(tool_id: str) -> dict:
    """Format a tool-execution-completed event."""
    return {"type": "tool_done", "tool_id": tool_id}


def format_timeline_updated() -> dict:
    """Format a timeline-state-changed notification."""
    return {"type": "timeline_updated"}


def format_done(turn_id: int) -> dict:
    """Format an end-of-turn message."""
    return {"type": "done", "turn_id": turn_id}


def format_error(message: str) -> dict:
    """Format an error message for the client."""
    return {"type": "error", "message": message}


def format_busy() -> dict:
    """Format a busy/throttle message."""
    return {"type": "busy"}


def format_connected(session_token: str) -> dict:
    """Format a connection-acknowledged message."""
    return {"type": "connected", "session_token": session_token}


# ---------------------------------------------------------------------------
# ChatSession — agentic loop with Anthropic streaming API
# ---------------------------------------------------------------------------


class ChatSession:
    """Bridge between WebSocket and Anthropic SDK for streaming tool-use.

    Manages conversation history and runs an agentic loop: stream LLM
    response, execute tool calls, feed results back, repeat until the
    model stops requesting tools.
    """

    def __init__(self, orchestrator, timeline_model) -> None:
        self._orchestrator = orchestrator
        self._timeline = timeline_model
        self._messages: list[dict] = []
        self._processing = False
        self._cancel_event = asyncio.Event()

    async def handle_message(self, ws, text: str) -> None:
        """Process a user message.  Rejects with busy if already processing."""
        if self._processing:
            await ws.send_json(format_busy())
            return
        self._processing = True
        self._cancel_event.clear()
        try:
            await self._agentic_loop(ws, text)
        except Exception as e:
            logger.exception("Error in agentic loop")
            await ws.send_json(format_error(str(e)))
        finally:
            self._processing = False

    def cancel(self) -> None:
        """Signal the running loop to stop after the current step."""
        self._cancel_event.set()

    # -- Private -------------------------------------------------------------

    async def _agentic_loop(self, ws, text: str) -> None:
        """Run the tool-use loop with Anthropic streaming API."""
        try:
            import anthropic
        except ImportError:
            await ws.send_json(format_error("anthropic package not installed"))
            await ws.send_json(format_done(self._orchestrator.turn_count))
            return

        self._messages.append({"role": "user", "content": text})
        loop = asyncio.get_running_loop()
        client = anthropic.AsyncAnthropic()
        tools = self._get_tools_json()
        system_prompt = self._orchestrator.get_system_prompt()

        while True:
            if self._cancel_event.is_set():
                await ws.send_json(format_done(self._orchestrator.turn_count))
                return

            tool_calls: list[dict] = []
            async with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                tools=tools,
                messages=self._messages,
            ) as stream:
                async for event in stream:
                    if self._cancel_event.is_set():
                        break
                    await self._forward_stream_event(ws, event, tool_calls)

            response = await stream.get_final_message()
            self._messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                await ws.send_json(format_done(self._orchestrator.turn_count))
                return

            # Execute tool calls in executor (they may block)
            tool_results: list[dict] = []
            for tc in tool_calls:
                result_str = await loop.run_in_executor(
                    None,
                    self._orchestrator.handle_tool_call,
                    tc["name"],
                    tc["input"],
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result_str,
                    }
                )
                await ws.send_json(format_tool_done(tc["id"]))

                if self._is_timeline_modifying(tc["name"], tc["input"]):
                    self._timeline.reload_from_xges()
                    await ws.send_json(format_timeline_updated())

            self._messages.append({"role": "user", "content": tool_results})

    async def _forward_stream_event(self, ws, event, tool_calls: list[dict]) -> None:
        """Forward a streaming event to the WebSocket client."""
        if not hasattr(event, "type"):
            return
        if event.type == "text":
            await ws.send_json(format_text_delta(event.text))
        elif event.type == "content_block_start":
            cb = getattr(event, "content_block", None)
            if cb is not None and cb.type == "tool_use":
                tool_calls.append(
                    {"id": cb.id, "name": cb.name, "input": {}}
                )
                await ws.send_json(format_tool_start(cb.name, cb.id))

    def _get_tools_json(self) -> list[dict]:
        """Convert orchestrator meta-tools to Anthropic tool format."""
        return [
            {
                "name": mt.name,
                "description": mt.description,
                "input_schema": mt.parameters,
            }
            for mt in self._orchestrator.get_meta_tools()
        ]

    def _is_timeline_modifying(self, tool_name: str, tool_input: dict) -> bool:
        """Check if a tool call modifies the timeline."""
        if tool_name != "call_tool":
            return False
        inner = tool_input.get("tool_name", "")
        modifying_domains = {"editing", "compositing", "motion_graphics", "scene"}
        try:
            schema = self._orchestrator.session.registry.get_tool_schema(inner)
            return schema.domain in modifying_domains
        except Exception:
            return False
