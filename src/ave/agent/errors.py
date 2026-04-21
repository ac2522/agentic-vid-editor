"""Shared errors for agent-session safety enforcement."""

from __future__ import annotations


class ScopeViolationError(Exception):
    """Raised when an agent attempts a tool call outside its owned domains."""


class SourceAssetWriteError(Exception):
    """Raised when a tool call would write to a source-asset path."""
