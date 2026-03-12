"""Transcription domain tool registration."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_transcription_tools(registry: ToolRegistry) -> None:
    """Register transcription domain tools."""

    @registry.tool(
        domain="transcription",
        requires=["transcript_loaded"],
        provides=["search_results"],
        tags=["find words", "search dialogue", "locate speech", "find quote",
              "search captions", "subtitle search"],
    )
    def search_transcript(transcript_json: str, query: str):
        """Search transcript for matching words. Case insensitive."""
        import json

        from ave.tools.transcribe import Transcript
        from ave.tools.transcript_edit import search_transcript as _search

        transcript = Transcript(**json.loads(transcript_json))
        return _search(transcript, query)

    @registry.tool(
        domain="transcription",
        requires=["transcript_loaded"],
        provides=["fillers_found"],
        tags=["um", "uh", "like", "filler words", "verbal tics", "stammering",
              "clean up speech", "remove ums", "hesitations"],
    )
    def find_fillers(transcript_json: str, fillers: list = None):
        """Find filler words (um, uh, like, etc.) in the transcript."""
        import json

        from ave.tools.transcribe import Transcript
        from ave.tools.transcript_edit import find_filler_words

        transcript = Transcript(**json.loads(transcript_json))
        filler_set = set(fillers) if fillers else None
        return find_filler_words(transcript, filler_set)

    @registry.tool(
        domain="transcription",
        requires=["transcript_loaded"],
        provides=["text_cut_computed"],
        tags=["cut between words", "remove section", "delete dialogue",
              "trim speech", "cut sentence"],
    )
    def text_cut(transcript_json: str, start_word: str, end_word: str):
        """Cut a region of the timeline between two spoken words."""
        import json

        from ave.tools.transcribe import Transcript
        from ave.tools.transcript_edit import compute_text_cut

        transcript = Transcript(**json.loads(transcript_json))
        return compute_text_cut(transcript, start_word, end_word)

    @registry.tool(
        domain="transcription",
        requires=["transcript_loaded"],
        provides=["text_keep_computed"],
        tags=["keep only", "isolate quote", "extract soundbite", "keep section",
              "preserve dialogue"],
    )
    def text_keep(transcript_json: str, start_word: str, end_word: str):
        """Keep only the region between two spoken words, cutting the rest."""
        import json

        from ave.tools.transcribe import Transcript
        from ave.tools.transcript_edit import compute_text_keep

        transcript = Transcript(**json.loads(transcript_json))
        return compute_text_keep(transcript, start_word, end_word)
