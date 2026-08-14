"""Gap-detection: the 3 gating conditions from DESIGN.md, part 1.

This module implements *only* the three gating conditions that decide
whether a moment in a transcript is a "gap candidate" at all:

1. non_trivial_delegation -- the user asked Claude to do something with
   real conceptual weight (not reformat/rename/fetch).
2. no_engagement_signal -- after Claude's output, the user showed no sign
   of trying to understand it (no "why", no follow-up, no manual edit,
   no pushback).
3. immediate_accept -- the output was used as-is, no diff review or
   hesitation visible.

A candidate only "passes the gates" (see `GapCandidate.passes_gates`) when
all three are true. Recurrence weighting, foundational-weight scoring, and
the four non-gate exclusion rules (opted-out delegation, boilerplate,
already-understood, incident firefighting) are explicitly NOT implemented
here -- see BUILD-SCHEDULE.md day 3.

Classifying a transcript against this rubric is a judgment call ("did the
user show an engagement signal?") that a deterministic parser cannot make
reliably -- it requires reading the actual conversation. So this module
asks Claude to do the classification, via a forced tool-use call that
returns structured JSON rather than free text. Everything Claude cannot be
trusted to get right on its own (malformed shape, wrong types, an index
that doesn't point at a real message) is validated in Python before a
GapCandidate is ever constructed.

The Claude call is injected via the `GapDetectionClient` protocol so the
gating/parsing logic can be tested without a live API call -- no
ANTHROPIC_API_KEY is available in the build sandbox this module was
written in, so `AnthropicGapDetectionClient` (the real implementation) has
NOT been exercised against the live API. Its request-construction and
response-extraction logic is written directly against the documented
Anthropic Messages/tool-use API shape, but only `detect_gaps` +
`parse_gap_candidates` (fed by a fake client) are actually verified by the
test suite. Treat `AnthropicGapDetectionClient` as unverified until it's
run somewhere with a real key.

Privacy note: the prompt instructs Claude to paraphrase `description` away
from verbatim proprietary content, but this module does not itself enforce
that -- gap candidates are an internal intermediate representation, not
the final script. The hard privacy enforcement point is the script
generation step (BUILD-SCHEDULE.md day 6), per DESIGN.md's privacy
constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .schema import Transcript

__all__ = [
    "GapCandidate",
    "GapDetectionResponseError",
    "GapDetectionClient",
    "AnthropicGapDetectionClient",
    "build_gap_detection_prompt",
    "parse_gap_candidates",
    "detect_gaps",
    "GAP_CANDIDATE_TOOL_SCHEMA",
    "DEFAULT_MODEL",
]

DEFAULT_MODEL = "claude-sonnet-5"

_REQUIRED_CANDIDATE_FIELDS = (
    "topic",
    "description",
    "user_message_index",
    "non_trivial_delegation",
    "no_engagement_signal",
    "immediate_accept",
    "reasoning",
)

_BOOL_FIELDS = ("non_trivial_delegation", "no_engagement_signal", "immediate_accept")
_STR_FIELDS = ("topic", "description", "reasoning")

# Forced tool-use schema: Claude must call this "tool" with an argument
# matching this shape, which is how we get structured JSON out of a chat
# model instead of parsing free text.
GAP_CANDIDATE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "description": (
                "Every moment in the transcript that involves the user "
                "delegating something to the assistant, evaluated against "
                "the 3 gating conditions. Include moments even if they "
                "fail one or more gates -- gate evaluation happens per "
                "candidate, filtering happens downstream."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Short topic label, e.g. 'race condition debugging'.",
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "1-2 sentence paraphrase of what was delegated, "
                            "with real identifiers, credentials, and "
                            "business-specific details stripped out."
                        ),
                    },
                    "user_message_index": {
                        "type": "integer",
                        "description": (
                            "Index into the transcript's messages list of "
                            "the user message that initiated this delegation."
                        ),
                    },
                    "non_trivial_delegation": {
                        "type": "boolean",
                        "description": (
                            "True if the ask had real conceptual weight "
                            "(implement/fix/explain logic), false if it was "
                            "mechanical (reformat/rename/fetch)."
                        ),
                    },
                    "no_engagement_signal": {
                        "type": "boolean",
                        "description": (
                            "True if the user showed NO sign of trying to "
                            "understand the output (no why-question, no "
                            "follow-up, no manual edit, no pushback)."
                        ),
                    },
                    "immediate_accept": {
                        "type": "boolean",
                        "description": (
                            "True if the output was used as-is with no diff "
                            "review or hesitation visible in the transcript."
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence citing what in the transcript supports each gate value.",
                    },
                },
                "required": list(_REQUIRED_CANDIDATE_FIELDS),
            },
        }
    },
    "required": ["candidates"],
}


class GapDetectionResponseError(ValueError):
    """Raised when a gap-detection API response doesn't match the expected shape."""


@dataclass(frozen=True)
class GapCandidate:
    session_id: str
    topic: str
    description: str
    user_message_index: int
    non_trivial_delegation: bool
    no_engagement_signal: bool
    immediate_accept: bool
    reasoning: str

    def __post_init__(self) -> None:
        if not self.session_id:
            raise GapDetectionResponseError("GapCandidate requires a session_id")
        if not self.topic.strip():
            raise GapDetectionResponseError("GapCandidate requires a non-empty topic")
        if self.user_message_index < 0:
            raise GapDetectionResponseError(
                f"user_message_index must be >= 0, got {self.user_message_index}"
            )

    @property
    def passes_gates(self) -> bool:
        """True only when all 3 gating conditions hold (DESIGN.md rubric items 1-3)."""
        return self.non_trivial_delegation and self.no_engagement_signal and self.immediate_accept


class GapDetectionClient(Protocol):
    """Anything that can classify a transcript against the gate rubric.

    Implementations return the raw candidate dict matching
    GAP_CANDIDATE_TOOL_SCHEMA -- i.e. `{"candidates": [...]}` -- *before*
    validation. `parse_gap_candidates` does the validation, so both the
    real client and any fake/mock used in tests share the same validation
    path.
    """

    def classify(self, transcript: Transcript) -> dict[str, Any]: ...


class AnthropicGapDetectionClient:
    """Real GapDetectionClient backed by the Anthropic Messages API.

    NOT exercised against a live API in this build sandbox (no
    ANTHROPIC_API_KEY available here) -- see module docstring. The
    `anthropic` import is deferred to __init__ so importing this module
    (and running the fake-client-backed tests) doesn't require the
    `anthropic` package to be importable in every environment.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client: Any = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            import anthropic  # deferred import, see docstring

            self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def classify(self, transcript: Transcript) -> dict[str, Any]:
        prompt = build_gap_detection_prompt(transcript)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            tools=[
                {
                    "name": "report_gap_candidates",
                    "description": (
                        "Report every delegation moment found in the transcript, "
                        "each evaluated against the 3 gating conditions."
                    ),
                    "input_schema": GAP_CANDIDATE_TOOL_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "report_gap_candidates"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use" and getattr(block, "name", None) == "report_gap_candidates":
                return block.input
        raise GapDetectionResponseError(
            "Anthropic response did not contain the expected report_gap_candidates tool_use block"
        )


def build_gap_detection_prompt(transcript: Transcript) -> str:
    """Build the user-turn prompt sent to Claude for gap classification.

    Deterministic and side-effect-free so it can be unit tested without
    any API access.
    """
    lines = [
        "You are evaluating a transcript of a user delegating tasks to an "
        "AI assistant, to find moments where the user may have missed a "
        "chance to learn something. Apply exactly these 3 gating "
        "conditions to every delegation moment you find -- do not infer "
        "what the user probably doesn't know, only flag what the "
        "transcript text actually shows:",
        "",
        "1. non_trivial_delegation: the ask had real conceptual weight "
        "(implement/fix/explain logic), not a mechanical edit "
        "(reformat/rename/fetch).",
        "2. no_engagement_signal: after the assistant's output, the user "
        "showed NO sign of trying to understand it -- no why-question, no "
        "follow-up, no manual edit, no pushback or request for an "
        "alternative.",
        "3. immediate_accept: the output was used as-is (run/sent/executed) "
        "with no diff review or hesitation visible.",
        "",
        "Report every delegation moment via the report_gap_candidates tool, "
        "including moments that fail one or more gates -- filtering happens "
        "downstream, not in your judgment. Paraphrase `description` so it "
        "carries no real identifiers, credentials, or business-specific "
        "details.",
        "",
        f"Transcript session_id: {transcript.session_id}",
        f"Transcript title: {transcript.title}",
        "Messages (index: role: content):",
    ]
    for i, msg in enumerate(transcript.messages):
        lines.append(f"{i}: {msg.role}: {msg.content}")
    return "\n".join(lines)


def parse_gap_candidates(raw: dict[str, Any], transcript: Transcript) -> list[GapCandidate]:
    """Validate + convert a raw classify() response into GapCandidate objects.

    Raises GapDetectionResponseError with a specific message for any
    structural problem, rather than letting a KeyError/TypeError leak out
    of a mis-shapen (or hallucinated) API response.
    """
    if not isinstance(raw, dict):
        raise GapDetectionResponseError("gap-detection response must be an object")

    if "candidates" not in raw:
        raise GapDetectionResponseError("gap-detection response missing 'candidates'")

    raw_candidates = raw["candidates"]
    if not isinstance(raw_candidates, list):
        raise GapDetectionResponseError("'candidates' must be a list")

    num_messages = len(transcript.messages)
    candidates: list[GapCandidate] = []
    for i, raw_c in enumerate(raw_candidates):
        if not isinstance(raw_c, dict):
            raise GapDetectionResponseError(f"candidates[{i}] must be an object")

        missing = [f for f in _REQUIRED_CANDIDATE_FIELDS if f not in raw_c]
        if missing:
            raise GapDetectionResponseError(
                f"candidates[{i}] missing required field(s): {', '.join(missing)}"
            )

        for field in _BOOL_FIELDS:
            if not isinstance(raw_c[field], bool):
                raise GapDetectionResponseError(
                    f"candidates[{i}].{field} must be a boolean, got {type(raw_c[field]).__name__}"
                )

        for field in _STR_FIELDS:
            if not isinstance(raw_c[field], str) or not raw_c[field].strip():
                raise GapDetectionResponseError(
                    f"candidates[{i}].{field} must be a non-empty string"
                )

        idx = raw_c["user_message_index"]
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise GapDetectionResponseError(
                f"candidates[{i}].user_message_index must be an integer, got {type(idx).__name__}"
            )
        if idx < 0 or idx >= num_messages:
            raise GapDetectionResponseError(
                f"candidates[{i}].user_message_index {idx} is out of range for a "
                f"transcript with {num_messages} messages"
            )
        if transcript.messages[idx].role != "user":
            raise GapDetectionResponseError(
                f"candidates[{i}].user_message_index {idx} points at a "
                f"{transcript.messages[idx].role!r} message, not a user message"
            )

        candidates.append(
            GapCandidate(
                session_id=transcript.session_id,
                topic=raw_c["topic"],
                description=raw_c["description"],
                user_message_index=idx,
                non_trivial_delegation=raw_c["non_trivial_delegation"],
                no_engagement_signal=raw_c["no_engagement_signal"],
                immediate_accept=raw_c["immediate_accept"],
                reasoning=raw_c["reasoning"],
            )
        )

    return candidates


def detect_gaps(transcript: Transcript, client: GapDetectionClient) -> list[GapCandidate]:
    """Classify a transcript and return all structurally-valid GapCandidates.

    This returns every candidate the client reported, including ones that
    fail one or more gates -- use `[c for c in result if c.passes_gates]`
    to get only the ones that pass all 3 gating conditions. Recurrence and
    foundational-weight scoring across multiple transcripts (rubric items
    4-6) are not applied here -- see BUILD-SCHEDULE.md day 3.
    """
    raw = client.classify(transcript)
    return parse_gap_candidates(raw, transcript)
