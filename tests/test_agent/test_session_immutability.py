"""Source-asset immutability tests."""

import hashlib
from pathlib import Path

import pytest

from ave.agent.domains import Domain
from ave.agent.errors import SourceAssetWriteError
from ave.agent.session import EditingSession


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_call_tool_rejects_path_under_source_media(tmp_path: Path):
    """A tool call whose resolved path falls under assets/media/source/ is rejected."""
    project = tmp_path / "proj"
    source_dir = project / "assets" / "media" / "source"
    source_dir.mkdir(parents=True)
    original = source_dir / "footage.mxf"
    original.write_bytes(b"untouchable")
    pre = _sha(original)

    session = EditingSession(project_root=project)

    def mutate_file(path: str) -> None:
        """Pretend to mutate a file."""
        Path(path).write_bytes(b"tampered")

    session.registry.register(
        "mutate_file",
        mutate_file,
        domain="video",
        domains_touched=(Domain.VIDEO,),
    )

    with pytest.raises(SourceAssetWriteError, match="source"):
        session.call_tool("mutate_file", {"path": str(original)})

    # File unchanged
    assert _sha(original) == pre
    assert original.read_bytes() == b"untouchable"


def test_call_tool_allows_paths_outside_source(tmp_path: Path):
    """Paths not under source/ go through normally."""
    project = tmp_path / "proj"
    working_dir = project / "assets" / "media" / "working"
    working_dir.mkdir(parents=True)
    intermediate = working_dir / "intermediate.mxf"
    intermediate.write_bytes(b"editable")

    session = EditingSession(project_root=project)

    def mutate_file(path: str) -> dict:
        """Mutates the file."""
        Path(path).write_bytes(b"changed")
        return {"path": path}

    session.registry.register(
        "mutate_file",
        mutate_file,
        domain="video",
        domains_touched=(Domain.VIDEO,),
    )

    session.call_tool("mutate_file", {"path": str(intermediate)})
    assert intermediate.read_bytes() == b"changed"


def test_no_project_root_means_no_immutability_check(tmp_path: Path):
    """Without project_root, the check is skipped (backward compat)."""
    session = EditingSession()  # no project_root
    suspicious = tmp_path / "assets" / "media" / "source" / "file.mxf"
    suspicious.parent.mkdir(parents=True)
    suspicious.write_bytes(b"irrelevant")

    def noop(path: str) -> None:
        """Noop."""

    session.registry.register(
        "noop",
        noop,
        domain="video",
        domains_touched=(Domain.VIDEO,),
    )
    # Should NOT raise because project_root not set
    session.call_tool("noop", {"path": str(suspicious)})
