"""Script generation: DESIGN.md's "Script template (per topic)" section, BUILD-SCHEDULE.md day 6.

DESIGN.md's 4-part per-topic template:

1. Real hook (30-60s): paraphrase the actual moment from this week, without
   exposing sensitive specifics (no real API keys, customer data, internal
   system names beyond what's needed for context).
2. Concept, first principles (3-5 min): the underlying idea independent of
   the user's specific code.
3. Worked example (3-5 min): a clean, minimal example illustrating the
   concept -- can reuse the *shape* of the real problem without reusing
   sensitive content.
4. Next time (1 min): a concrete prompt/checklist for catching this
   themselves next time it comes up.

## Privacy enforcement -- the hard rule this module exists to satisfy

DESIGN.md's privacy constraint: "The gap-detection and script-generation
steps must never pass verbatim proprietary content ... into the final
script. Extraction should paraphrase to the *concept* before it ever
reaches the script-writing step."

This module enforces that in two independent ways, not just one, because a
single enforcement point is one bug away from a leak:

1. **By API design.** `generate_script` / `build_script_prompt` accept only
   `topic: str` and `source_descriptions: list[str]` -- never a
   `Transcript` or `Message`. There is no code path in this module through
   which a raw transcript message can reach the prompt sent to Claude.
   Callers are expected to pass `GapCandidate.description` strings (already
   paraphrased by gap_detection.py's own prompt instructions) or
   equivalently-paraphrased text -- the type signature makes it structurally
   impossible to accidentally pass raw message content instead, since
   `Transcript`/`Message` objects simply aren't accepted here.
2. **By output scan, as defense in depth.** `enforce_privacy` regex-scans
   every generated section (and, as a sanity check, the source descriptions
   themselves) for patterns that look like secrets or credentials --
   `sk-...`/`ghp_...`/`AKIA...`-style API keys, PEM private key headers,
   and `password=`/`api_key=`-style assignments. This does not replace
   paraphrasing (a regex scan cannot tell whether "the worker queue" is a
   real internal system name), it exists only to catch the specific,
   mechanically-detectable case where a literal credential slipped through
   upstream paraphrasing -- the same "no fabrication, no leak" standing
   discipline already applied elsewhere in this profile (e.g. the
   sk_live/whsec_ grep done before pushing Stripe integration code in a
   sibling repo). A match raises `PrivacyViolationError` and the script is
   never returned to the caller.

Gap-detection's own prompt (gap_detection.py) already instructs Claude to
paraphrase `GapCandidate.description` away from verbatim proprietary
content, but gap_detection.py's docstring is explicit that it does NOT
itself enforce that -- "the hard privacy enforcement point is the script
generation step (BUILD-SCHEDULE.md day 6)." This module is that point.

## Pattern

Same shape as gap_detection.py and clustering.py: a `ScriptGenerationClient`
Protocol so the prompt/parse/validate logic can be tested without a live
API call, a real `AnthropicScriptGenerationClient` implementation written
directly against the documented Anthropic Messages/tool-use API shape but
NOT exercised against a live API in this build sandbox (no
ANTHROPIC_API_KEY available here -- see gap_detection.py's module docstring
for the same caveat), and a forced tool-use call so the response is
structured JSON rather than free text that would need fragile parsing.

## What this module does NOT do

- Decide *which* topics get a script or how many minutes each gets -- that
  is curriculum.py's job (day 5). This module takes `topic` and `minutes`
  as already-decided inputs (e.g. from a `CurriculumSlot`), it doesn't
  select or budget anything itself.
- Render audio -- that's the TTS module (day 7).
- Gather source descriptions from a pool of `GapCandidate`/`ScoredGap`
  objects -- day 8 (CLI wiring) is where this module gets connected to the
  rest of the pipeline, the same deferral pattern curriculum.py used for
  `CurriculumCandidate` vs. `TopicCluster`/`ScoredGap`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "TopicScript",
    "ScriptGenerationResponseError",
    "PrivacyViolationError",
    "ScriptGenerationClient",
    "AnthropicScriptGenerationClient",
    "build_script_prompt",
    "enforce_privacy",
    "parse_topic_script",
    "generate_script",
    "SCRIPT_TOOL_SCHEMA",
    "DEFAULT_MODEL",
]

DEFAULT_MODEL = "claude-sonnet-5"

_SECTION_FIELDS = ("hook", "concept", "example", "next_time")

# Defense-in-depth secret/credential patterns -- see module docstring.
# Deliberately narrow and mechanically-checkable (things a regex can
# actually catch reliably), not an attempt to detect "sensitive business
# context" in general, which requires judgment a regex cannot supply.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("OpenAI-style API key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("AWS access key ID", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("Stripe secret/webhook key", re.compile(r"(sk|whsec)_(live|test)_[A-Za-z0-9]{10,}")),
    ("PEM private key header", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "inline credential assignment",
        re.compile(r"(?i)\b(password|api[_-]?key|secret|token)\s*[:=]\s*\S{6,}"),
    ),
)

# Forced tool-use schema: Claude must call this "tool" with an argument
# matching this shape, mirroring gap_detection.py's/clustering.py's
# forced-tool-use pattern for structured (rather than free-text) output.
SCRIPT_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hook": {
            "type": "string",
            "description": (
                "30-60s real hook: paraphrase the actual moment from this "
                "week using only the provided source descriptions -- do "
                "not invent specifics not present in them, and do not "
                "include real API keys, credentials, customer data, or "
                "internal system names beyond what's needed for context."
            ),
        },
        "concept": {
            "type": "string",
            "description": (
                "3-5 min explanation of the underlying idea from first "
                "principles, independent of the user's specific code."
            ),
        },
        "example": {
            "type": "string",
            "description": (
                "3-5 min clean, minimal worked example illustrating the "
                "concept -- may reuse the shape of the real problem "
                "described in the source descriptions, but must not reuse "
                "any sensitive content from them."
            ),
        },
        "next_time": {
            "type": "string",
            "description": (
                "~1 min concrete prompt/checklist the user can use to catch "
                "this themselves next time it comes up."
            ),
        },
    },
    "required": list(_SECTION_FIELDS),
}


class ScriptGenerationResponseError(ValueError):
    """Raised when a script-generation API response doesn't match the expected shape."""


class PrivacyViolationError(ValueError):
    """Raised when generated (or source) text matches a secret/credential pattern.

    See module docstring -- this is defense-in-depth on top of the
    paraphrase-before-scripting API design, not a replacement for it.
    """


@dataclass(frozen=True)
class TopicScript:
    """A complete 4-part script for one topic, ready to hand to the TTS step."""

    topic: str
    minutes: float
    hook: str
    concept: str
    example: str
    next_time: str

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ScriptGenerationResponseError("TopicScript requires a non-empty topic")
        if self.minutes <= 0:
            raise ScriptGenerationResponseError(
                f"TopicScript {self.topic!r} minutes must be > 0, got {self.minutes}"
            )
        for field in _SECTION_FIELDS:
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ScriptGenerationResponseError(
                    f"TopicScript {self.topic!r} section {field!r} must be a non-empty string"
                )

    @property
    def full_text(self) -> str:
        """The 4 sections concatenated in template order, for show-notes/TTS input."""
        return "\n\n".join(
            [
                f"[Hook]\n{self.hook}",
                f"[Concept]\n{self.concept}",
                f"[Worked example]\n{self.example}",
                f"[Next time]\n{self.next_time}",
            ]
        )


class ScriptGenerationClient(Protocol):
    """Anything that can turn (topic, source descriptions, minutes) into a raw script response.

    Implementations return a raw dict matching SCRIPT_TOOL_SCHEMA --
    `{"hook": ..., "concept": ..., "example": ..., "next_time": ...}` --
    *before* validation. `parse_topic_script` does the validation, so both
    the real client and any fake/mock used in tests share the same
    validation (and privacy-enforcement) path.
    """

    def generate(self, topic: str, source_descriptions: list[str], minutes: float) -> dict[str, Any]: ...


class AnthropicScriptGenerationClient:
    """Real ScriptGenerationClient backed by the Anthropic Messages API.

    NOT exercised against a live API in this build sandbox (no
    ANTHROPIC_API_KEY available here) -- same caveat as
    `gap_detection.AnthropicGapDetectionClient` and
    `clustering.AnthropicClusteringClient`. The `anthropic` import is
    deferred to __init__ so importing this module (and running the
    fake-client-backed tests) doesn't require the `anthropic` package to be
    importable in every environment.
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

    def generate(self, topic: str, source_descriptions: list[str], minutes: float) -> dict[str, Any]:
        prompt = build_script_prompt(topic, source_descriptions, minutes)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            tools=[
                {
                    "name": "report_topic_script",
                    "description": (
                        "Report the 4-part lesson script (hook, concept, "
                        "example, next_time) for this topic."
                    ),
                    "input_schema": SCRIPT_TOOL_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "report_topic_script"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use" and getattr(block, "name", None) == "report_topic_script":
                return block.input
        raise ScriptGenerationResponseError(
            "Anthropic response did not contain the expected report_topic_script tool_use block"
        )


def build_script_prompt(topic: str, source_descriptions: list[str], minutes: float) -> str:
    """Build the user-turn prompt sent to Claude for script generation.

    Deterministic and side-effect-free so it can be unit tested without any
    API access. Takes only paraphrased `source_descriptions` (strings), not
    a `Transcript`/`Message` -- see module docstring's "by API design"
    enforcement point. Raises ValueError if `topic` is empty,
    `source_descriptions` is empty, or `minutes` <= 0 -- there is nothing to
    ground a script in without at least one source description, and no
    lesson without a topic or a positive time budget.
    """
    if not topic.strip():
        raise ValueError("build_script_prompt requires a non-empty topic")
    if not source_descriptions:
        raise ValueError(
            f"build_script_prompt for topic {topic!r} requires at least one source description"
        )
    if minutes <= 0:
        raise ValueError(f"build_script_prompt minutes must be > 0, got {minutes}")

    lines = [
        "You are writing a short spoken-audio lesson script for exactly one "
        "topic, to be narrated as a podcast segment. Follow this 4-part "
        "template exactly, per the design doc:",
        "",
        "1. hook (30-60s): paraphrase the actual moment from this week that "
        "prompted this lesson -- use ONLY the source descriptions provided "
        "below, do not invent additional specifics. Do not include real API "
        "keys, credentials, customer data, or internal system names beyond "
        "what's needed for context.",
        "2. concept (3-5 min): explain the underlying idea from first "
        "principles, independent of the user's specific code.",
        "3. example (3-5 min): a clean, minimal worked example illustrating "
        "the concept -- you may reuse the *shape* of the real problem from "
        "the source descriptions, but must not reuse any sensitive content "
        "from them.",
        "4. next_time (~1 min): a concrete prompt or checklist the listener "
        "can use to catch this themselves next time it comes up.",
        "",
        "The source descriptions below are already paraphrased -- they "
        "contain no real credentials or verbatim proprietary content. Do "
        "not attempt to reconstruct or guess at anything more specific than "
        "what they state.",
        "",
        f"Topic: {topic}",
        f"Target total depth: {minutes:.1f} minutes",
        "Source descriptions (already paraphrased, from this week's usage):",
    ]
    for i, desc in enumerate(source_descriptions, start=1):
        lines.append(f"{i}. {desc}")
    lines.append("")
    lines.append(
        "Report the script via the report_topic_script tool, with exactly "
        "the 4 fields hook, concept, example, next_time."
    )
    return "\n".join(lines)


def enforce_privacy(sections: dict[str, str]) -> None:
    """Scan a dict of section-name -> text for secret/credential-like patterns.

    Raises `PrivacyViolationError` naming the offending section and pattern
    on the first match found. This is defense-in-depth, applied to every
    generated section before a `TopicScript` is ever constructed -- see
    module docstring for why this exists alongside (not instead of) the
    paraphrase-before-scripting API design.
    """
    for section_name, text in sections.items():
        for pattern_name, pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                raise PrivacyViolationError(
                    f"generated script section {section_name!r} matched a "
                    f"{pattern_name!r} pattern -- refusing to return a script "
                    f"that may contain a leaked credential"
                )


def parse_topic_script(raw: dict[str, Any], topic: str, minutes: float) -> TopicScript:
    """Validate + convert a raw generate() response into a TopicScript.

    Raises ScriptGenerationResponseError with a specific message for any
    structural problem (missing/wrong-typed field), and
    PrivacyViolationError if any section matches a secret/credential
    pattern (via `enforce_privacy`) -- rather than letting a
    KeyError/TypeError leak out of a malformed response, or letting a
    plausible-looking but credential-containing response through.
    """
    if not isinstance(raw, dict):
        raise ScriptGenerationResponseError("script-generation response must be an object")

    missing = [f for f in _SECTION_FIELDS if f not in raw]
    if missing:
        raise ScriptGenerationResponseError(
            f"script-generation response missing required field(s): {', '.join(missing)}"
        )

    sections: dict[str, str] = {}
    for field in _SECTION_FIELDS:
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise ScriptGenerationResponseError(
                f"script-generation response field {field!r} must be a non-empty string"
            )
        sections[field] = value

    # Defense-in-depth privacy scan BEFORE constructing the TopicScript --
    # see module docstring. A privacy violation must never reach the
    # caller as a valid TopicScript.
    enforce_privacy(sections)

    return TopicScript(
        topic=topic,
        minutes=minutes,
        hook=sections["hook"],
        concept=sections["concept"],
        example=sections["example"],
        next_time=sections["next_time"],
    )


def generate_script(
    topic: str,
    source_descriptions: list[str],
    minutes: float,
    client: ScriptGenerationClient,
) -> TopicScript:
    """Generate and validate a complete TopicScript for one topic.

    `source_descriptions` must already be paraphrased (e.g.
    `GapCandidate.description` strings) -- this function has no code path
    that accepts raw transcript content, by design (see module docstring).
    Also runs `enforce_privacy` on the source descriptions themselves
    before ever building the prompt, as a sanity check that upstream
    paraphrasing didn't already fail -- if a source description itself
    matches a secret pattern, this raises before Claude is even called,
    rather than trusting the model to strip it out downstream.
    """
    enforce_privacy({f"source_description[{i}]": d for i, d in enumerate(source_descriptions)})
    raw = client.generate(topic, source_descriptions, minutes)
    return parse_topic_script(raw, topic, minutes)
