"""Tests for verification loop — models, compare_metrics, ProbeVerifier, VerifiedSession."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ave.tools.verify import EditIntent, VerificationResult, compare_metrics
from ave.tools.verify_probe import ProbeVerifier
from ave.agent.verification import VerifiedSession
from ave.agent.session import EditingSession
from ave.agent.registry import ToolRegistry
from ave.agent.dependencies import SessionState

from tests.conftest import requires_ffmpeg


# ---------------------------------------------------------------------------
# EditIntent & VerificationResult creation
# ---------------------------------------------------------------------------

class TestEditIntent:
    def test_creation(self):
        intent = EditIntent(
            tool_name="trim",
            description="Trim clip to 2 seconds",
            expected_changes={"duration_seconds": 2.0, "has_audio": True},
        )
        assert intent.tool_name == "trim"
        assert intent.description == "Trim clip to 2 seconds"
        assert intent.expected_changes == {"duration_seconds": 2.0, "has_audio": True}

    def test_frozen(self):
        intent = EditIntent(
            tool_name="trim",
            description="Trim",
            expected_changes={},
        )
        with pytest.raises(AttributeError):
            intent.tool_name = "split"  # type: ignore[misc]


class TestVerificationResult:
    def test_creation(self):
        intent = EditIntent(tool_name="trim", description="t", expected_changes={})
        result = VerificationResult(
            passed=True,
            intent=intent,
            actual_metrics={"duration_seconds": 2.0},
            discrepancies=[],
            confidence=0.95,
        )
        assert result.passed is True
        assert result.confidence == 0.95
        assert result.discrepancies == []
        assert result.intent is intent

    def test_frozen(self):
        intent = EditIntent(tool_name="x", description="x", expected_changes={})
        result = VerificationResult(
            passed=False, intent=intent, actual_metrics={}, discrepancies=["bad"], confidence=0.5,
        )
        with pytest.raises(AttributeError):
            result.passed = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compare_metrics
# ---------------------------------------------------------------------------

class TestCompareMetrics:
    def test_exact_match_passes(self):
        passed, discreps = compare_metrics(
            {"width": 1920, "height": 1080},
            {"width": 1920, "height": 1080},
        )
        assert passed is True
        assert discreps == []

    def test_numeric_within_tolerance_passes(self):
        passed, discreps = compare_metrics(
            {"duration_seconds": 2.0},
            {"duration_seconds": 2.3},
        )
        assert passed is True
        assert discreps == []

    def test_numeric_outside_tolerance_fails(self):
        passed, discreps = compare_metrics(
            {"duration_seconds": 2.0},
            {"duration_seconds": 5.0},
        )
        assert passed is False
        assert len(discreps) == 1
        assert "duration_seconds" in discreps[0]

    def test_missing_key_in_actual_fails(self):
        passed, discreps = compare_metrics(
            {"width": 1920},
            {},
        )
        assert passed is False
        assert len(discreps) == 1
        assert "width" in discreps[0]

    def test_extra_key_in_actual_ignored(self):
        passed, discreps = compare_metrics(
            {"width": 1920},
            {"width": 1920, "height": 1080},
        )
        assert passed is True
        assert discreps == []

    def test_custom_tolerance_per_key(self):
        # duration_seconds with tight tolerance should fail
        passed, discreps = compare_metrics(
            {"duration_seconds": 2.0},
            {"duration_seconds": 2.3},
            tolerances={"duration_seconds": 0.1},
        )
        assert passed is False
        assert len(discreps) == 1

    def test_boolean_comparison(self):
        passed, discreps = compare_metrics(
            {"has_audio": True},
            {"has_audio": False},
        )
        assert passed is False
        assert len(discreps) == 1
        assert "has_audio" in discreps[0]

    def test_boolean_match(self):
        passed, discreps = compare_metrics(
            {"has_audio": True},
            {"has_audio": True},
        )
        assert passed is True

    def test_string_comparison(self):
        passed, discreps = compare_metrics(
            {"video_codec": "h264"},
            {"video_codec": "h264"},
        )
        assert passed is True

    def test_string_mismatch(self):
        passed, discreps = compare_metrics(
            {"video_codec": "h264"},
            {"video_codec": "hevc"},
        )
        assert passed is False


# ---------------------------------------------------------------------------
# ProbeVerifier — protocol conformance
# ---------------------------------------------------------------------------

class TestProbeVerifierProtocol:
    def test_satisfies_verification_backend(self):
        verifier = ProbeVerifier()
        # Structural subtyping check: ProbeVerifier has the verify method
        assert hasattr(verifier, "verify")
        # Check signature compatibility
        import inspect
        sig = inspect.signature(verifier.verify)
        params = list(sig.parameters.keys())
        assert "intent" in params
        assert "segment_path" in params


# ---------------------------------------------------------------------------
# ProbeVerifier — ffprobe integration
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestProbeVerifier:
    def _generate_test_clip(self, path: Path, duration: float = 2.0) -> Path:
        """Generate a short test clip with video and audio."""
        out = path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=24",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", "-shortest",
                str(out),
            ],
            capture_output=True, check=True,
        )
        return out

    def test_probe_segment(self, tmp_path):
        clip = self._generate_test_clip(tmp_path)
        verifier = ProbeVerifier()
        metrics = verifier.probe_segment(clip)
        assert abs(metrics["duration_seconds"] - 2.0) < 0.5
        assert metrics["width"] == 320
        assert metrics["height"] == 240
        assert metrics["has_video"] is True
        assert metrics["has_audio"] is True
        assert isinstance(metrics["video_codec"], str)
        assert isinstance(metrics["audio_codec"], str)

    def test_verify_passes(self, tmp_path):
        clip = self._generate_test_clip(tmp_path)
        intent = EditIntent(
            tool_name="trim",
            description="Trim to 2s",
            expected_changes={"duration_seconds": 2.0, "has_audio": True, "width": 320},
        )
        verifier = ProbeVerifier()
        result = verifier.verify(intent, clip)
        assert result.passed is True
        assert result.discrepancies == []
        assert result.confidence > 0.0

    def test_verify_fails_on_mismatch(self, tmp_path):
        clip = self._generate_test_clip(tmp_path)
        intent = EditIntent(
            tool_name="trim",
            description="Trim to 10s",
            expected_changes={"duration_seconds": 10.0},
        )
        verifier = ProbeVerifier()
        result = verifier.verify(intent, clip)
        assert result.passed is False
        assert len(result.discrepancies) > 0

    def test_probe_nonexistent_file(self, tmp_path):
        verifier = ProbeVerifier()
        with pytest.raises(Exception):
            verifier.probe_segment(tmp_path / "nonexistent.mp4")


# ---------------------------------------------------------------------------
# VerifiedSession
# ---------------------------------------------------------------------------

def _make_session() -> EditingSession:
    """Create a minimal EditingSession without loading tools."""
    s = EditingSession.__new__(EditingSession)
    s._registry = ToolRegistry()
    s._state = SessionState()
    s._history = []
    s._project_path = None
    s._snapshot_manager = None
    s._transition_graph = None
    s._lock = threading.Lock()
    return s


class TestVerifiedSession:
    def test_delegates_call_tool(self):
        session = _make_session()
        session.call_tool = MagicMock(return_value={"ok": True})
        vs = VerifiedSession(session)
        result = vs.call_tool("trim", {"start": 0, "end": 5})
        session.call_tool.assert_called_once_with("trim", {"start": 0, "end": 5})
        assert result == {"ok": True}

    def test_verify_turn_returns_none_without_verifier(self):
        session = _make_session()
        vs = VerifiedSession(session, verifier=None)
        intent = EditIntent(tool_name="trim", description="t", expected_changes={})
        result = vs.verify_turn(intent, Path("/fake/path.mp4"))
        assert result is None

    def test_verify_turn_calls_verifier(self):
        session = _make_session()
        session.call_tool = MagicMock(return_value={})
        mock_verifier = MagicMock()
        intent = EditIntent(tool_name="trim", description="t", expected_changes={})
        expected_result = VerificationResult(
            passed=True, intent=intent, actual_metrics={}, discrepancies=[], confidence=1.0,
        )
        mock_verifier.verify.return_value = expected_result
        vs = VerifiedSession(session, verifier=mock_verifier)
        vs.call_tool("trim", {})
        result = vs.verify_turn(intent, Path("/fake/path.mp4"))
        mock_verifier.verify.assert_called_once_with(intent, Path("/fake/path.mp4"))
        assert result is expected_result

    def test_turn_tools_tracks_calls(self):
        session = _make_session()
        session.call_tool = MagicMock(return_value={})
        vs = VerifiedSession(session)
        vs.call_tool("trim", {})
        vs.call_tool("color_grade", {})
        assert vs.turn_tools == ["trim", "color_grade"]

    def test_reset_turn_clears_tracking(self):
        session = _make_session()
        session.call_tool = MagicMock(return_value={})
        vs = VerifiedSession(session)
        vs.call_tool("trim", {})
        vs.call_tool("split", {})
        assert len(vs.turn_tools) == 2
        vs.reset_turn()
        assert vs.turn_tools == []

    def test_verify_turn_resets_after(self):
        session = _make_session()
        session.call_tool = MagicMock(return_value={})
        mock_verifier = MagicMock()
        intent = EditIntent(tool_name="trim", description="t", expected_changes={})
        mock_verifier.verify.return_value = VerificationResult(
            passed=True, intent=intent, actual_metrics={}, discrepancies=[], confidence=1.0,
        )
        vs = VerifiedSession(session, verifier=mock_verifier)
        vs.call_tool("trim", {})
        vs.verify_turn(intent, Path("/fake"))
        assert vs.turn_tools == []

    def test_session_property(self):
        session = _make_session()
        vs = VerifiedSession(session)
        assert vs.session is session
