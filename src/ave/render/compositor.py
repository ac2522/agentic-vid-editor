"""Compositor strategy selection.

Preference: skia > cpu > gl (gl has known crash bugs #728, #786).
Falls back gracefully when GStreamer is not available.
"""

from __future__ import annotations

from dataclasses import dataclass

# Compositor element names in GStreamer
COMPOSITOR_ELEMENTS = {
    "skia": "skiacompositor",
    "cpu": "compositor",
    "gl": "glvideomixer",
}

PREFERENCE_ORDER = ("skia", "cpu", "gl")


@dataclass(frozen=True)
class CompositorSelection:
    """Result of compositor selection."""

    strategy: str  # "skia", "cpu", or "gl"
    element_name: str  # GStreamer element name
    reason: str  # Why this was selected


class CompositorStrategy:
    """Compositor selection logic.

    Preference: skia > cpu > gl (gl has known crash bugs #728, #786).
    Falls back gracefully when GStreamer is not available.
    """

    @staticmethod
    def detect_available() -> list[str]:
        """Detect available compositor strategies.

        Returns ["cpu"] as safe default if GStreamer not importable.
        """
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)

            available = []
            for strategy, element_name in COMPOSITOR_ELEMENTS.items():
                factory = Gst.ElementFactory.find(element_name)
                if factory is not None:
                    available.append(strategy)

            # Always include cpu as fallback
            if "cpu" not in available:
                available.append("cpu")
            return available
        except (ImportError, ValueError):
            return ["cpu"]

    @staticmethod
    def select(
        preference: str = "auto", available: list[str] | None = None
    ) -> CompositorSelection:
        """Select best compositor.

        preference: "auto", "skia", "cpu", or "gl"
        available: override detected capabilities (for testing)
        """
        if available is None:
            available = CompositorStrategy.detect_available()

        if preference != "auto" and preference in available:
            return CompositorSelection(
                strategy=preference,
                element_name=COMPOSITOR_ELEMENTS[preference],
                reason=f"Explicitly requested: {preference}",
            )

        # Auto or fallback: use preference order
        for strategy in PREFERENCE_ORDER:
            if strategy in available:
                reason = (
                    "Best available (preference order)"
                    if preference == "auto"
                    else f"Fallback from unavailable {preference!r}"
                )
                return CompositorSelection(
                    strategy=strategy,
                    element_name=COMPOSITOR_ELEMENTS[strategy],
                    reason=reason,
                )

        # Should never reach here since cpu is always in available
        return CompositorSelection(
            strategy="cpu",
            element_name="compositor",
            reason="Ultimate fallback",
        )

    @staticmethod
    def get_element_name(strategy: str) -> str:
        """Return GStreamer element name for a strategy.

        Raises ValueError for unknown strategy.
        """
        if strategy not in COMPOSITOR_ELEMENTS:
            raise ValueError(
                f"Unknown compositor strategy: {strategy!r}. "
                f"Valid strategies: {list(COMPOSITOR_ELEMENTS.keys())}"
            )
        return COMPOSITOR_ELEMENTS[strategy]
