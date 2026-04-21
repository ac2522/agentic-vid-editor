"""WebSocket chat protocol and ChatSession for the AVE web UI.

Protocol functions handle message serialisation/deserialisation.
ChatSession bridges the WebSocket with the Anthropic streaming API
for an agentic tool-use loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from ave.agent.state_sync import build_state_summary

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


def format_done(turn_id: int, checkpoint_id: str | None = None) -> dict:
    """Format an end-of-turn message.

    ``turn_id`` is the orchestrator's monotonic turn counter (legacy).
    ``checkpoint_id`` is the session-level turn checkpoint identifier the
    client uses to request undo/redo.
    """
    payload: dict = {"type": "done", "turn_id": turn_id}
    if checkpoint_id is not None:
        payload["checkpoint_id"] = checkpoint_id
    return payload


def format_error(message: str) -> dict:
    """Format an error message for the client."""
    return {"type": "error", "message": message}


def format_busy() -> dict:
    """Format a busy/throttle message."""
    return {"type": "busy"}


def format_connected(session_token: str) -> dict:
    """Format a connection-acknowledged message."""
    return {"type": "connected", "session_token": session_token}


def format_timeline_rollback(*, turn_id: str, direction: str) -> dict:
    """Format a state-rollback notification for the client.

    direction: "undo" or "redo"
    """
    return {"type": "timeline_rollback", "turn_id": turn_id, "direction": direction}


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
        self._last_summary_timestamp: float = 0.0
        self._current_checkpoint_id: str | None = None

    def _prepare_user_content(self, text: str) -> str:
        """Prefix the user's message with a state summary covering activity since the last turn.

        The state-sync summary tells the agent what happened (e.g., user undos,
        other agents' edits) between agent turns. When the orchestrator does not
        expose a session/activity_log, the original text is returned unchanged.
        """
        session = getattr(self._orchestrator, "session", None)
        if session is None:
            return text
        activity_log = getattr(session, "_activity_log", None)
        if activity_log is None:
            return text

        summary = build_state_summary(
            session=session,
            activity_log=activity_log,
            since_timestamp=self._last_summary_timestamp,
        )
        self._last_summary_timestamp = summary.generated_at
        return summary.render() + "\n\n" + text

    async def handle_message(self, ws, text: str) -> None:
        """Process a user message.  Rejects with busy if already processing."""
        if self._processing:
            await ws.send_json(format_busy())
            return
        self._processing = True
        self._cancel_event.clear()

        checkpoint_id = "turn-" + uuid.uuid4().hex
        session = getattr(self._orchestrator, "session", None)
        captured = False
        if session is not None:
            try:
                session.begin_turn(checkpoint_id)
                captured = True
            except Exception:
                logger.debug("begin_turn failed — continuing without checkpoint", exc_info=True)

        self._current_checkpoint_id = checkpoint_id if captured else None

        try:
            await self._agentic_loop(ws, text)
            if session is not None and captured:
                try:
                    session.end_turn(checkpoint_id)
                except Exception:
                    logger.debug("end_turn failed", exc_info=True)
        except Exception as e:
            logger.exception("Error in agentic loop")
            await ws.send_json(format_error(str(e)))
        finally:
            self._processing = False
            self._current_checkpoint_id = None

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
            await ws.send_json(format_done(self._orchestrator.turn_count, self._current_checkpoint_id))
            return

        self._messages.append({"role": "user", "content": self._prepare_user_content(text)})
        loop = asyncio.get_running_loop()
        client = anthropic.AsyncAnthropic()
        tools = self._get_tools_json()
        system_prompt = self._orchestrator.get_system_prompt()

        while True:
            if self._cancel_event.is_set():
                await ws.send_json(format_done(self._orchestrator.turn_count, self._current_checkpoint_id))
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
                await ws.send_json(format_done(self._orchestrator.turn_count, self._current_checkpoint_id))
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
                tool_calls.append({"id": cb.id, "name": cb.name, "input": {}})
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
        """Check if a tool call modifies the timeline.

        Uses the modifies_timeline flag from the tool registry rather than
        a hardcoded domain set, so all tools flagged with
        modifies_timeline=True are correctly detected.
        """
        if tool_name != "call_tool":
            return False
        inner = tool_input.get("tool_name", "")
        try:
            return self._orchestrator.session.registry.tool_modifies_timeline(inner)
        except Exception:
            return False
