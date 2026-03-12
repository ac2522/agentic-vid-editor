"""Audio domain tool registration."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_audio_tools(registry: ToolRegistry) -> None:
    """Register audio domain tools."""

    @registry.tool(
        domain="audio",
        requires=["timeline_loaded", "clip_exists"],
        provides=["volume_set"],
        tags=["loudness", "gain", "audio level", "make louder", "make quieter",
              "turn up", "turn down", "mute", "boost audio"],
    )
    def volume(level_db: float):
        """Set audio volume level in decibels (0 dB = unity)."""
        from ave.tools.audio import compute_volume

        return compute_volume(level_db)

    @registry.tool(
        domain="audio",
        requires=["timeline_loaded", "clip_exists"],
        provides=["fade_applied"],
        tags=["fade in", "fade out", "audio ramp", "smooth start", "smooth end",
              "gradual"],
    )
    def fade(clip_duration_ns: int, fade_in_ns: int, fade_out_ns: int):
        """Apply audio fade-in and fade-out to a clip."""
        from ave.tools.audio import compute_fade

        return compute_fade(clip_duration_ns, fade_in_ns, fade_out_ns)

    @registry.tool(
        domain="audio",
        requires=["timeline_loaded", "clip_exists"],
        provides=["audio_normalized"],
        tags=["level audio", "consistent volume", "loudness standard",
              "even out audio", "LUFS", "peak normalize"],
    )
    def normalize(current_peak_db: float, target_peak_db: float):
        """Normalize audio to a target peak level."""
        from ave.tools.audio import compute_normalize

        return compute_normalize(current_peak_db, target_peak_db)
