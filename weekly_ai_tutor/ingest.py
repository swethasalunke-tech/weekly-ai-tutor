"""Load transcripts from disk into the Transcript schema."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Transcript, TranscriptValidationError, transcript_from_dict

__all__ = ["load_transcript_file", "load_transcripts_from_dir", "TranscriptValidationError"]


def load_transcript_file(path: str | Path) -> Transcript:
    """Load a single transcript JSON file.

    Raises FileNotFoundError if the path doesn't exist, json.JSONDecodeError
    if it isn't valid JSON, and TranscriptValidationError if it's valid JSON
    that doesn't match the transcript schema.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data = json.loads(raw)
    try:
        return transcript_from_dict(data)
    except TranscriptValidationError as e:
        raise TranscriptValidationError(f"{p}: {e}") from e


def load_transcripts_from_dir(dir_path: str | Path) -> list[Transcript]:
    """Load every *.json file in a directory as a transcript.

    Files that fail to parse or validate are skipped, not silently dropped --
    a warning is printed to stderr for each one so a bad file in the input
    directory doesn't just vanish without a trace.
    """
    import sys

    d = Path(dir_path)
    if not d.is_dir():
        raise NotADirectoryError(f"{d} is not a directory")

    transcripts: list[Transcript] = []
    for f in sorted(d.glob("*.json")):
        try:
            transcripts.append(load_transcript_file(f))
        except (json.JSONDecodeError, TranscriptValidationError) as e:
            print(f"weekly-ai-tutor: skipping {f}: {e}", file=sys.stderr)
    return transcripts
