"""Verification loop — models, protocol, and metric comparison.

Defines the EditIntent/VerificationResult data models, the VerificationBackend
protocol, and a pure-function compare_metrics utility for checking expected
vs actual edit outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class EditIntent:
    """Captures what an edit was supposed to accomplish."""

    tool_name: str
    description: str
    expected_changes: dict  # Key metrics that should change
    # e.g., {"duration_seconds": 2.0, "has_audio": True, "width": 1920}


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying an edit against intent."""

    passed: bool
    intent: EditIntent
    actual_metrics: dict
    discrepancies: list[str]  # Human-readable mismatches
    confidence: float  # 0.0-1.0


class VerificationBackend(Protocol):
    """Protocol for edit verification strategies."""

    def verify(self, intent: EditIntent, segment_path: Path) -> VerificationResult: ...


_DEFAULT_FLOAT_TOLERANCE = 0.5


def compare_metrics(
    expected: dict,
    actual: dict,
    tolerances: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    """Compare expected vs actual metrics.

    tolerances: per-key absolute tolerance for numeric values.
    Default tolerance for floats: 0.5 (e.g., duration within 0.5s).

    Returns (passed, list_of_discrepancies).
    """
    discreps: list[str] = []
    tol = tolerances or {}

    for key, exp_val in expected.items():
        if key not in actual:
            discreps.append(f"{key}: expected {exp_val!r} but key missing from actual metrics")
            continue

        act_val = actual[key]

        if isinstance(exp_val, bool):
            # Bool check must come before numeric since bool is subclass of int
            if act_val != exp_val:
                discreps.append(f"{key}: expected {exp_val!r}, got {act_val!r}")
        elif isinstance(exp_val, (int, float)):
            key_tol = tol.get(key, _DEFAULT_FLOAT_TOLERANCE if isinstance(exp_val, float) else 0)
            if abs(act_val - exp_val) > key_tol:
                discreps.append(
                    f"{key}: expected {exp_val!r} (±{key_tol}), got {act_val!r}"
                )
        else:
            # String or other — exact equality
            if act_val != exp_val:
                discreps.append(f"{key}: expected {exp_val!r}, got {act_val!r}")

    passed = len(discreps) == 0
    return passed, discreps
