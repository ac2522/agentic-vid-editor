"""Unit tests for transcript-driven editing — pure logic, no GES required."""

import pytest

from ave.tools.transcribe import Transcript, TranscriptSegment
from ave.tools.transcript_edit import (
    CutRegion,
    EditOp,
    KeepRegion,
    TranscriptEditError,
    compute_cuts_to_edit_ops,
    compute_filler_removal_cuts,
    compute_text_cut,
    compute_text_keep,
    find_filler_words,
    find_word_range,
    search_transcript,
    seconds_to_ns,
)


def make_transcript() -> Transcript:
    """10-second transcript with word-level timing."""
    return Transcript(
        language="en",
        duration=10.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.5,
                text="Hello um world",
                words=[
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "um", "start": 0.7, "end": 0.9},
                    {"word": "world", "start": 1.0, "end": 1.5},
                ],
            ),
            TranscriptSegment(
                start=2.5,
                end=5.0,
                text="This is a test",
                words=[
                    {"word": "This", "start": 2.5, "end": 2.8},
                    {"word": "is", "start": 2.9, "end": 3.0},
                    {"word": "a", "start": 3.1, "end": 3.2},
                    {"word": "test", "start": 3.3, "end": 3.8},
                ],
            ),
            TranscriptSegment(
                start=5.0,
                end=8.0,
                text="uh let me think",
                words=[
                    {"word": "uh", "start": 5.0, "end": 5.3},
                    {"word": "let", "start": 5.5, "end": 5.7},
                    {"word": "me", "start": 5.8, "end": 5.9},
                    {"word": "think", "start": 6.0, "end": 6.5},
                ],
            ),
            TranscriptSegment(
                start=8.0,
                end=10.0,
                text="that is all",
                words=[
                    {"word": "that", "start": 8.0, "end": 8.3},
                    {"word": "is", "start": 8.4, "end": 8.5},
                    {"word": "all", "start": 8.6, "end": 9.0},
                ],
            ),
        ],
    )


def make_clean_transcript() -> Transcript:
    """Transcript with no filler words."""
    return Transcript(
        language="en",
        duration=5.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.5,
                text="Hello world",
                words=[
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 1.0, "end": 1.5},
                ],
            ),
            TranscriptSegment(
                start=2.5,
                end=5.0,
                text="This is great",
                words=[
                    {"word": "This", "start": 2.5, "end": 2.8},
                    {"word": "is", "start": 2.9, "end": 3.0},
                    {"word": "great", "start": 3.1, "end": 3.5},
                ],
            ),
        ],
    )


class TestSecondsToNs:
    """Test time conversion utility."""

    def test_seconds_to_ns(self):
        assert seconds_to_ns(1.5) == 1_500_000_000

    def test_seconds_to_ns_zero(self):
        assert seconds_to_ns(0.0) == 0

    def test_seconds_to_ns_small(self):
        assert seconds_to_ns(0.001) == 1_000_000


class TestFindFillerWords:
    """Test filler word detection."""

    def test_find_filler_words_default(self):
        transcript = make_transcript()
        fillers = find_filler_words(transcript)

        assert len(fillers) == 2
        words = {f.word for f in fillers}
        assert words == {"um", "uh"}

        # Check timestamps are converted to nanoseconds
        um = next(f for f in fillers if f.word == "um")
        assert um.start_ns == seconds_to_ns(0.7)
        assert um.end_ns == seconds_to_ns(0.9)
        assert um.segment_index == 0

        uh = next(f for f in fillers if f.word == "uh")
        assert uh.start_ns == seconds_to_ns(5.0)
        assert uh.end_ns == seconds_to_ns(5.3)
        assert uh.segment_index == 2

    def test_find_filler_words_custom_list(self):
        transcript = make_transcript()
        fillers = find_filler_words(transcript, fillers={"like", "you know"})

        assert len(fillers) == 0

    def test_find_filler_words_none_found(self):
        transcript = make_clean_transcript()
        fillers = find_filler_words(transcript)

        assert fillers == []


class TestComputeFillerRemovalCuts:
    """Test conversion of filler locations to cut regions."""

    def test_compute_filler_removal_cuts(self):
        transcript = make_transcript()
        fillers = find_filler_words(transcript)
        cuts = compute_filler_removal_cuts(fillers)

        assert len(cuts) == 2
        for cut in cuts:
            assert isinstance(cut, CutRegion)
            assert "filler" in cut.reason

    def test_compute_filler_removal_preserves_context(self):
        """Cuts include small padding (default 50ms) around filler."""
        transcript = make_transcript()
        fillers = find_filler_words(transcript)
        padding_ns = 50_000_000  # 50ms

        cuts = compute_filler_removal_cuts(fillers, padding_ns=padding_ns)

        um_cut = next(c for c in cuts if "um" in c.reason)
        # "um" is at 0.7-0.9s, with 50ms padding:
        # start = 0.7s - 50ms = 650ms, but clamped to 0
        assert um_cut.start_ns == seconds_to_ns(0.7) - padding_ns
        assert um_cut.end_ns == seconds_to_ns(0.9) + padding_ns

    def test_compute_filler_removal_empty(self):
        cuts = compute_filler_removal_cuts([])
        assert cuts == []


class TestFindWordRange:
    """Test word range finding."""

    def test_find_word_range(self):
        transcript = make_transcript()
        start_ns, end_ns = find_word_range(transcript, "This", "test")

        assert start_ns == seconds_to_ns(2.5)
        assert end_ns == seconds_to_ns(3.8)

    def test_find_word_range_not_found(self):
        transcript = make_transcript()

        with pytest.raises(TranscriptEditError):
            find_word_range(transcript, "nonexistent", "words")


class TestComputeTextCut:
    """Test text-based cut computation."""

    def test_compute_text_cut(self):
        transcript = make_transcript()
        cut = compute_text_cut(transcript, "um", "world")

        assert isinstance(cut, CutRegion)
        assert cut.start_ns == seconds_to_ns(0.7)
        assert cut.end_ns == seconds_to_ns(1.5)
        assert cut.reason == "text cut"

    def test_compute_text_cut_not_found(self):
        transcript = make_transcript()

        with pytest.raises(TranscriptEditError):
            compute_text_cut(transcript, "nonexistent", "words")


class TestComputeTextKeep:
    """Test text-based keep computation."""

    def test_compute_text_keep(self):
        transcript = make_transcript()
        keep = compute_text_keep(transcript, "This", "test")

        assert isinstance(keep, KeepRegion)
        assert keep.start_ns == seconds_to_ns(2.5)
        assert keep.end_ns == seconds_to_ns(3.8)


class TestSearchTranscript:
    """Test transcript searching."""

    def test_search_transcript(self):
        transcript = make_transcript()
        matches = search_transcript(transcript, "test")

        assert len(matches) == 1
        assert matches[0].word == "test"
        assert matches[0].start_ns == seconds_to_ns(3.3)
        assert matches[0].end_ns == seconds_to_ns(3.8)
        assert matches[0].segment_index == 1

    def test_search_transcript_case_insensitive(self):
        transcript = make_transcript()
        matches = search_transcript(transcript, "TEST")

        assert len(matches) == 1
        assert matches[0].word == "test"

    def test_search_transcript_no_matches(self):
        transcript = make_transcript()
        matches = search_transcript(transcript, "xyz")

        assert matches == []

    def test_search_transcript_multiple_matches(self):
        transcript = make_transcript()
        # "is" appears in segments 1 and 3
        matches = search_transcript(transcript, "is")

        assert len(matches) == 2


class TestComputeCutsToEditOps:
    """Test conversion of cut regions to edit operations."""

    def test_compute_cuts_to_edit_ops(self):
        cuts = [
            CutRegion(
                start_ns=seconds_to_ns(0.7),
                end_ns=seconds_to_ns(0.9),
                reason="filler: um",
            ),
        ]
        timeline_duration_ns = seconds_to_ns(10.0)
        ops = compute_cuts_to_edit_ops(cuts, timeline_duration_ns)

        assert len(ops) >= 1
        for op in ops:
            assert isinstance(op, EditOp)
            assert op.op_type in ("trim", "split_remove")

    def test_compute_cuts_non_overlapping(self):
        """Non-overlapping cuts produce separate edit ops."""
        cuts = [
            CutRegion(start_ns=seconds_to_ns(0.7), end_ns=seconds_to_ns(0.9), reason="filler"),
            CutRegion(start_ns=seconds_to_ns(5.0), end_ns=seconds_to_ns(5.3), reason="filler"),
        ]
        timeline_duration_ns = seconds_to_ns(10.0)
        ops = compute_cuts_to_edit_ops(cuts, timeline_duration_ns)

        assert len(ops) == 2
        # Ops should be sorted by start time
        assert ops[0].start_ns < ops[1].start_ns

    def test_compute_cuts_adjacent(self):
        """Adjacent cuts merge into single operation."""
        cuts = [
            CutRegion(start_ns=seconds_to_ns(1.0), end_ns=seconds_to_ns(2.0), reason="cut1"),
            CutRegion(start_ns=seconds_to_ns(2.0), end_ns=seconds_to_ns(3.0), reason="cut2"),
        ]
        timeline_duration_ns = seconds_to_ns(10.0)
        ops = compute_cuts_to_edit_ops(cuts, timeline_duration_ns)

        # Adjacent cuts should merge
        assert len(ops) == 1
        assert ops[0].start_ns == seconds_to_ns(1.0)
        assert ops[0].end_ns == seconds_to_ns(3.0)

    def test_compute_cuts_overlapping_merge(self):
        """Overlapping cuts merge into single operation."""
        cuts = [
            CutRegion(start_ns=seconds_to_ns(1.0), end_ns=seconds_to_ns(2.5), reason="cut1"),
            CutRegion(start_ns=seconds_to_ns(2.0), end_ns=seconds_to_ns(3.0), reason="cut2"),
        ]
        timeline_duration_ns = seconds_to_ns(10.0)
        ops = compute_cuts_to_edit_ops(cuts, timeline_duration_ns)

        assert len(ops) == 1
        assert ops[0].start_ns == seconds_to_ns(1.0)
        assert ops[0].end_ns == seconds_to_ns(3.0)

    def test_compute_cuts_empty(self):
        ops = compute_cuts_to_edit_ops([], seconds_to_ns(10.0))
        assert ops == []
