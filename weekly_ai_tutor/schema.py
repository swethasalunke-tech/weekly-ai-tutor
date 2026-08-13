"""Data model for a single AI-assistant transcript.

This schema is deliberately tool-agnostic: it doesn't assume Claude Code's
internal on-disk session format (that format hasn't been verified against
this schema yet -- see BUILD-SCHEDULE.md day 1 note). Instead it defines a
plain JSON shape that any transcript source can be converted into, so the
rest of the pipeline (gap detection, clustering, curriculum, scripting)
never has to know where the data originally came from.

Expected JSON shape (one file per session):

{
  "session_id": "abc123",
  "title": "Fix race condition in worker queue",
  "started_at": "2026-08-10T14:32:00Z",
  "messages": [
    {"role": "user", "content": "...", "timestamp": "2026-08-10T14:32:00Z"},
    {"role": "assistant", "content": "...", "timestamp": "2026-08-10T14:32:05Z"}
  ]
}

`role` must be "user" or "assistant". `timestamp` is optional on individual
messages (falls back to None if missing or unparseable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class TranscriptValidationError(ValueError):
    """Raised when raw transcript JSON doesn't match the expected schema."""


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise TranscriptValidationError(
                f"message role must be 'user' or 'assistant', got {self.role!r}"
            )
        if not isinstance(self.content, str) or not self.content.strip():
            raise TranscriptValidationError("message content must be a non-empty string")


@dataclass(frozen=True)
class Transcript:
    session_id: str
    title: str
    started_at: Optional[datetime]
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise TranscriptValidationError("session_id is required")
        if not self.messages:
            raise TranscriptValidationError(
                f"transcript {self.session_id!r} has no messages"
            )

    @property
    def user_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "user"]

    @property
    def assistant_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "assistant"]


def _parse_timestamp(raw: object) -> Optional[datetime]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    try:
        # Accept both "...Z" and explicit-offset ISO 8601.
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def transcript_from_dict(data: dict) -> Transcript:
    """Build a Transcript from a parsed JSON object matching the schema above.

    Raises TranscriptValidationError with a specific message on any
    structural problem, rather than letting a KeyError/TypeError leak out.
    """
    if not isinstance(data, dict):
        raise TranscriptValidationError("transcript JSON must be an object")

    missing = [k for k in ("session_id", "title", "messages") if k not in data]
    if missing:
        raise TranscriptValidationError(f"missing required field(s): {', '.join(missing)}")

    raw_messages = data["messages"]
    if not isinstance(raw_messages, list):
        raise TranscriptValidationError("'messages' must be a list")

    messages: list[Message] = []
    for i, raw_msg in enumerate(raw_messages):
        if not isinstance(raw_msg, dict):
            raise TranscriptValidationError(f"messages[{i}] must be an object")
        if "role" not in raw_msg or "content" not in raw_msg:
            raise TranscriptValidationError(
                f"messages[{i}] missing required field(s): role/content"
            )
        messages.append(
            Message(
                role=raw_msg["role"],
                content=raw_msg["content"],
                timestamp=_parse_timestamp(raw_msg.get("timestamp")),
            )
        )

    return Transcript(
        session_id=str(data["session_id"]),
        title=str(data.get("title", "")),
        started_at=_parse_timestamp(data.get("started_at")),
        messages=messages,
    )
