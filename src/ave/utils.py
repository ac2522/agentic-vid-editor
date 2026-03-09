"""Shared utilities for the ave package."""

from pathlib import Path
from urllib.parse import quote


def path_to_uri(path: Path) -> str:
    """Convert a Path to a file URI."""
    abs_path = str(path.resolve())
    return "file://" + quote(abs_path, safe="/")
