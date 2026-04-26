"""Pure XGES state-diff evaluator.

Parses XGES XML to extract timeline metrics and compares them against
scenario execute expectations.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Literal

from ave.harness.schema import ExecuteExpected


StateDiffVerdictRule = Literal[
    "state_ok",
    "clip_count_out_of_range",
    "duration_out_of_range",
    "missing_effects",
    "forbidden_effects",
]


@dataclass(frozen=True)
class StateDiffVerdict:
    passed: bool
    reason: str
    rule: StateDiffVerdictRule = "state_ok"


@dataclass(frozen=True)
class TimelineMetrics:
    clip_count: int
    duration_seconds: float
    effect_names: frozenset[str]


def extract_timeline_metrics(xges_content: str) -> TimelineMetrics:
    """Parse XGES XML and return timeline metrics."""
    root = ET.fromstring(xges_content)

    clips = root.findall(".//clip")
    clip_count = len(clips)

    total_ns = sum(int(c.get("duration", "0")) for c in clips)
    duration_seconds = total_ns / 1_000_000_000.0

    effect_names: set[str] = set()
    for effect in root.findall(".//effect"):
        asset_id = effect.get("asset-id")
        if asset_id:
            effect_names.add(asset_id)

    return TimelineMetrics(
        clip_count=clip_count,
        duration_seconds=duration_seconds,
        effect_names=frozenset(effect_names),
    )


def evaluate_execute_state(
    actual: TimelineMetrics,
    expected: ExecuteExpected,
) -> StateDiffVerdict:
    """Compare actual timeline metrics against execute expectations.

    Rules applied in order (first failure wins):
    1. clip_count outside min/max -> clip_count_out_of_range
    2. duration_seconds outside min/max -> duration_out_of_range
    3. required effects missing -> missing_effects
    4. forbidden effects present -> forbidden_effects
    5. pass -> state_ok
    """
    tl = expected.timeline

    cc_min = tl.clip_count.min
    cc_max = tl.clip_count.max
    if cc_min is not None and actual.clip_count < cc_min:
        return StateDiffVerdict(
            False,
            f"clip_count {actual.clip_count} is below minimum {cc_min}",
            "clip_count_out_of_range",
        )
    if cc_max is not None and actual.clip_count > cc_max:
        return StateDiffVerdict(
            False,
            f"clip_count {actual.clip_count} exceeds maximum {cc_max}",
            "clip_count_out_of_range",
        )

    dur_min = tl.duration_seconds.min
    dur_max = tl.duration_seconds.max
    if dur_min is not None and actual.duration_seconds < dur_min:
        return StateDiffVerdict(
            False,
            f"duration {actual.duration_seconds}s is below minimum {dur_min}s",
            "duration_out_of_range",
        )
    if dur_max is not None and actual.duration_seconds > dur_max:
        return StateDiffVerdict(
            False,
            f"duration {actual.duration_seconds}s exceeds maximum {dur_max}s",
            "duration_out_of_range",
        )

    missing = [e for e in tl.effects_applied if e not in actual.effect_names]
    if missing:
        return StateDiffVerdict(
            False,
            f"required effects not found in timeline: {missing}",
            "missing_effects",
        )

    forbidden_hits = [e for e in tl.effects_forbidden if e in actual.effect_names]
    if forbidden_hits:
        return StateDiffVerdict(
            False,
            f"forbidden effects present in timeline: {forbidden_hits}",
            "forbidden_effects",
        )

    return StateDiffVerdict(True, "timeline state satisfies all constraints", "state_ok")
