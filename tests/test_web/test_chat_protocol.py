"""Tests for ave.web.chat pure WebSocket protocol functions."""

import json


from ave.web.chat import (
    format_busy,
    format_connected,
    format_done,
    format_error,
    format_text_delta,
    format_timeline_updated,
    format_tool_done,
    format_tool_start,
    parse_client_message,
)


class TestParseClientMessage:
    def test_valid_json_with_type(self):
        raw = json.dumps({"type": "chat", "message": "hello"})
        result = parse_client_message(raw)
        assert result == {"type": "chat", "message": "hello"}

    def test_invalid_json(self):
        result = parse_client_message("not json {{{")
        assert result["type"] == "error"
        assert "message" in result

    def test_missing_type_field(self):
        raw = json.dumps({"message": "no type here"})
        result = parse_client_message(raw)
        assert result["type"] == "error"
        assert "type" in result["message"].lower()

    def test_empty_string(self):
        result = parse_client_message("")
        assert result["type"] == "error"

    def test_preserves_all_fields(self):
        raw = json.dumps({"type": "chat", "text": "hi", "extra": 42})
        result = parse_client_message(raw)
        assert result["extra"] == 42


class TestFormatTextDelta:
    def test_structure(self):
        result = format_text_delta("hello world")
        assert result == {"type": "text_delta", "text": "hello world"}

    def test_empty_string(self):
        result = format_text_delta("")
        assert result == {"type": "text_delta", "text": ""}


class TestFormatToolStart:
    def test_structure(self):
        result = format_tool_start("trim_clip", "tool_abc")
        assert result == {
            "type": "tool_start",
            "tool_name": "trim_clip",
            "tool_id": "tool_abc",
        }


class TestFormatToolDone:
    def test_structure(self):
        result = format_tool_done("tool_abc")
        assert result == {"type": "tool_done", "tool_id": "tool_abc"}


class TestFormatTimelineUpdated:
    def test_structure(self):
        result = format_timeline_updated()
        assert result == {"type": "timeline_updated"}


class TestFormatDone:
    def test_structure(self):
        result = format_done(42)
        assert result == {"type": "done", "turn_id": 42}

    def test_turn_id_zero(self):
        result = format_done(0)
        assert result["turn_id"] == 0

    def test_with_checkpoint_id(self):
        result = format_done(42, checkpoint_id="turn-abc")
        assert result == {"type": "done", "turn_id": 42, "checkpoint_id": "turn-abc"}

    def test_without_checkpoint_id_omits_field(self):
        result = format_done(42)
        assert "checkpoint_id" not in result


class TestFormatError:
    def test_structure(self):
        result = format_error("something broke")
        assert result == {"type": "error", "message": "something broke"}


class TestFormatBusy:
    def test_structure(self):
        result = format_busy()
        assert result == {"type": "busy"}


class TestFormatConnected:
    def test_structure(self):
        result = format_connected("tok_123")
        assert result == {"type": "connected", "session_token": "tok_123"}
