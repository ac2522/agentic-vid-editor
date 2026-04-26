"""Tests for Claude VLM judge."""
from pathlib import Path
import pytest


def _has_anthropic():
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


requires_anthropic = pytest.mark.skipif(not _has_anthropic(), reason="anthropic SDK not installed")


@requires_anthropic
def test_claude_vlm_implements_protocol():
    from ave.harness.judges.claude_vlm import ClaudeVlmJudge
    j = ClaudeVlmJudge(model="claude-haiku-4-5-20251001")
    assert "still" in j.supported_dimension_types
    assert "claude" in j.name.lower()


@requires_anthropic
def test_claude_vlm_extracts_n_frames(tmp_path, monkeypatch):
    """The judge should extract N frames at evenly spaced times."""
    from ave.harness.judges.claude_vlm import _extract_frames
    # Use a fake mp4 path; mock ffmpeg invocation
    from unittest.mock import patch
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        # Simulate frames being created
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            (frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8")
        out = _extract_frames(tmp_path / "fake.mp4", n=3, output_dir=frames_dir)
        assert len(out) == 3


@requires_anthropic
def test_claude_vlm_handles_api_error_gracefully(tmp_path, monkeypatch):
    """When the API fails, return a passed=False verdict not an exception."""
    from ave.harness.judges.claude_vlm import ClaudeVlmJudge

    class FakeException(Exception):
        pass

    j = ClaudeVlmJudge(model="claude-haiku-4-5-20251001")

    def boom(*a, **kw):
        raise FakeException("API down")

    monkeypatch.setattr(j, "_call_api", boom)
    # Create a fake mp4 file
    (tmp_path / "x.mp4").write_bytes(b"\x00" * 100)

    v = j.judge_dimension(
        rendered_path=tmp_path / "x.mp4",
        dimension="framing",
        prompt="subject centered",
        pass_threshold=0.5,
    )
    assert v.passed is False
    assert "error" in v.explanation.lower() or "api" in v.explanation.lower()


def test_module_importable_without_anthropic():
    """The module should import even when anthropic SDK is missing."""
    import importlib
    importlib.import_module("ave.harness.judges.claude_vlm")
