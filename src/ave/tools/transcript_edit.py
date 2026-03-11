"""Transcript-driven editing — convert text edits to timeline operations.

Pure logic layer. Takes transcript data and produces edit parameters
that the GES execution layer can apply.
"""

from __future__ import annotations

from dataclasses import dataclass

from ave.tools.transcribe import Transcript


class TranscriptEditError(Exception):
    """Raised when transcript editing fails."""


# Default filler words to detect
DEFAULT_FILLERS = {
    "um", "uh", "er", "ah", "like", "you know",
    "basically", "literally", "actually",
}


@dataclass(frozen=True)
class FillerMatch:
    """A detected filler word with timestamp."""

    word: str
    start_ns: int
    end_ns: int
    segment_index: int


@dataclass(frozen=True)
class CutRegion:
    """A region to cut from the timeline."""

    start_ns: int
    end_ns: int
    reason: str  # e.g. "filler: um", "text cut"


@dataclass(frozen=True)
class KeepRegion:
    """A region to keep (trim to)."""

    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class TranscriptMatch:
    """A search result in transcript."""

    word: str
    start_ns: int
    end_ns: int
    segment_index: int
    context: str  # surrounding text


@dataclass(frozen=True)
class EditOp:
    """An edit operation derived from transcript analysis."""

    op_type: str  # "trim" or "split_remove"
    start_ns: int
    end_ns: int


def seconds_to_ns(seconds: float) -> int:
    """Convert seconds to nanoseconds."""
    return int(seconds * 1_000_000_000)


def find_filler_words(
    transcript: Transcript,
    fillers: set[str] | None = None,
) -> list[FillerMatch]:
    """Find filler words in transcript with their timestamps."""
    filler_set = fillers if fillers is not None else DEFAULT_FILLERS
    filler_lower = {f.lower() for f in filler_set}
    matches: list[FillerMatch] = []

    for seg_idx, segment in enumerate(transcript.segments):
        for word_info in segment.words:
            word = word_info["word"]
            if word.lower() in filler_lower:
                matches.append(
                    FillerMatch(
                        word=word.lower(),
                        start_ns=seconds_to_ns(word_info["start"]),
                        end_ns=seconds_to_ns(word_info["end"]),
                        segment_index=seg_idx,
                    )
                )

    return matches


def compute_filler_removal_cuts(
    fillers: list[FillerMatch],
    padding_ns: int = 50_000_000,  # 50ms padding
) -> list[CutRegion]:
    """Convert filler word locations to cut regions with padding."""
    cuts: list[CutRegion] = []

    for filler in fillers:
        start = filler.start_ns - padding_ns
        end = filler.end_ns + padding_ns
        # Clamp start to 0
        if start < 0:
            start = 0
        cuts.append(
            CutRegion(
                start_ns=start,
                end_ns=end,
                reason=f"filler: {filler.word}",
            )
        )

    return cuts


def _collect_all_words(
    transcript: Transcript,
) -> list[tuple[dict, int]]:
    """Collect all words across segments with their segment index."""
    words = []
    for seg_idx, segment in enumerate(transcript.segments):
        for word_info in segment.words:
            words.append((word_info, seg_idx))
    return words


def find_word_range(
    transcript: Transcript,
    start_word: str,
    end_word: str,
) -> tuple[int, int]:
    """Find the time range spanning from start_word to end_word.

    Returns (start_ns, end_ns).
    Raises TranscriptEditError if words not found.
    """
    all_words = _collect_all_words(transcript)

    start_idx = None
    for i, (word_info, _) in enumerate(all_words):
        if word_info["word"].lower() == start_word.lower():
            start_idx = i
            break

    if start_idx is None:
        raise TranscriptEditError(f"Start word '{start_word}' not found in transcript")

    end_idx = None
    for i in range(start_idx, len(all_words)):
        if all_words[i][0]["word"].lower() == end_word.lower():
            end_idx = i

    if end_idx is None:
        raise TranscriptEditError(f"End word '{end_word}' not found after '{start_word}' in transcript")

    start_ns = seconds_to_ns(all_words[start_idx][0]["start"])
    end_ns = seconds_to_ns(all_words[end_idx][0]["end"])

    return start_ns, end_ns


def compute_text_cut(
    transcript: Transcript,
    start_word: str,
    end_word: str,
) -> CutRegion:
    """Compute a cut region from one word to another."""
    start_ns, end_ns = find_word_range(transcript, start_word, end_word)
    return CutRegion(start_ns=start_ns, end_ns=end_ns, reason="text cut")


def compute_text_keep(
    transcript: Transcript,
    start_word: str,
    end_word: str,
) -> KeepRegion:
    """Compute a keep region (everything outside is cut)."""
    start_ns, end_ns = find_word_range(transcript, start_word, end_word)
    return KeepRegion(start_ns=start_ns, end_ns=end_ns)


def search_transcript(
    transcript: Transcript,
    query: str,
) -> list[TranscriptMatch]:
    """Search transcript for matching words. Case insensitive."""
    matches: list[TranscriptMatch] = []
    query_lower = query.lower()

    for seg_idx, segment in enumerate(transcript.segments):
        for word_info in segment.words:
            if word_info["word"].lower() == query_lower:
                matches.append(
                    TranscriptMatch(
                        word=word_info["word"],
                        start_ns=seconds_to_ns(word_info["start"]),
                        end_ns=seconds_to_ns(word_info["end"]),
                        segment_index=seg_idx,
                        context=segment.text,
                    )
                )

    return matches


def compute_cuts_to_edit_ops(
    cuts: list[CutRegion],
    timeline_duration_ns: int,
) -> list[EditOp]:
    """Convert cut regions to concrete edit operations.

    Merges adjacent/overlapping cuts. Produces split_remove ops.
    """
    if not cuts:
        return []

    # Sort by start time
    sorted_cuts = sorted(cuts, key=lambda c: c.start_ns)

    # Merge overlapping/adjacent cuts
    merged: list[tuple[int, int]] = []
    current_start = sorted_cuts[0].start_ns
    current_end = sorted_cuts[0].end_ns

    for cut in sorted_cuts[1:]:
        if cut.start_ns <= current_end:
            # Overlapping or adjacent — extend
            current_end = max(current_end, cut.end_ns)
        else:
            merged.append((current_start, current_end))
            current_start = cut.start_ns
            current_end = cut.end_ns

    merged.append((current_start, current_end))

    # Convert to edit ops
    ops: list[EditOp] = []
    for start_ns, end_ns in merged:
        ops.append(
            EditOp(
                op_type="split_remove",
                start_ns=start_ns,
                end_ns=end_ns,
            )
        )

    return ops
