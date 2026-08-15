"""Recurrence + foundational-weight scoring (DESIGN.md rubric items 4-5).

`gap_detection.py` classifies one transcript at a time and has no view of
the rest of the week -- it can't know that "git rebase" showed up as a
zero-engagement gap in three different sessions. This module takes the
pool of `GapCandidate`s produced across *all* of a week's transcripts
(one or more `detect_gaps` calls) and applies:

- The three hard non-gate exclusions (opted_out, boilerplate,
  already_understood) via `GapCandidate.should_flag`.
- Rubric item 4, recurrence: the same topic appearing 2+ times with zero
  engagement is weighted "roughly by count."
- Rubric item 5, foundational vs. niche: concepts that generalize outrank
  one-off library trivia at equal recurrence.
- The fourth non-gate, incident_context: DESIGN.md says "note it, don't
  rank it the same as leisure-time exploration" -- a down-weight, not an
  exclude, so it's applied here as a severity multiplier rather than a
  filter.

DESIGN.md's scoring sketch is `severity = complexity_weight x
recurrence_count x foundational_weight`. `complexity_weight` isn't
otherwise defined anywhere in DESIGN.md -- there's no rubric item that
specifies how it should be derived from a candidate. Rather than invent an
undocumented formula for it, this module fixes it at `BASE_COMPLEXITY_WEIGHT
= 1.0` (a no-op multiplier) and implements exactly the two components
BUILD-SCHEDULE.md day 3 scopes: recurrence_count and foundational_weight,
plus the incident-context down-weight. If a later day's plan defines
complexity_weight concretely, `score_gap_candidates` is the place to wire
it in.

NOT implemented here (explicitly out of scope for day 3, see
BUILD-SCHEDULE.md and gap_detection.py's module docstring):

- Rework-cost weighting (rubric item 6).
- Real topic clustering (DESIGN.md step 3 "dedup + cluster", day 4) --
  this module groups by `GapCandidate.topic_key`, an exact-match
  normalization (stripped + lowercased), not fuzzy/semantic grouping.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gap_detection import GapCandidate

__all__ = [
    "ScoredGap",
    "score_gap_candidates",
    "BASE_COMPLEXITY_WEIGHT",
    "FOUNDATIONAL_WEIGHT",
    "NICHE_WEIGHT",
    "INCIDENT_CONTEXT_MULTIPLIER",
]

# See module docstring: complexity_weight has no defined derivation in
# DESIGN.md yet, so it's a fixed no-op multiplier pending further design.
BASE_COMPLEXITY_WEIGHT = 1.0

# DESIGN.md rubric item 5: "Concepts that generalize ... outrank one-off
# library-specific trivia, even at equal recurrence." 2x is a deliberate,
# documented starting point -- strong enough to outrank a niche topic with
# equal recurrence, not so strong it swamps a highly recurrent niche topic.
FOUNDATIONAL_WEIGHT = 2.0
NICHE_WEIGHT = 1.0

# DESIGN.md: incident-context delegation is "appropriate ... note it,
# don't rank it the same as leisure-time exploration" -- a down-weight,
# not a hard exclude. 0.5 halves severity rather than zeroing it, since
# the underlying gap is still real, just lower-priority.
INCIDENT_CONTEXT_MULTIPLIER = 0.5


@dataclass(frozen=True)
class ScoredGap:
    """One topic's aggregated score across every occurrence that reached scoring.

    `candidates` holds every `GapCandidate` that contributed to this
    group (all share the same `topic_key`, i.e. matched after
    strip+lowercase normalization) -- kept around so callers (e.g. day 4's
    clustering, day 5's curriculum fitting) can trace a score back to its
    source transcripts/messages rather than just seeing a number.
    """

    topic_key: str
    display_topic: str
    candidates: tuple[GapCandidate, ...]
    recurrence_count: int
    foundational: bool
    all_incident_context: bool
    severity: float

    def __post_init__(self) -> None:
        if not self.topic_key:
            raise ValueError("ScoredGap requires a non-empty topic_key")
        if not self.candidates:
            raise ValueError(f"ScoredGap {self.topic_key!r} requires at least one candidate")
        if self.recurrence_count != len(self.candidates):
            raise ValueError(
                f"ScoredGap {self.topic_key!r} recurrence_count "
                f"({self.recurrence_count}) must equal len(candidates) "
                f"({len(self.candidates)})"
            )
        if self.recurrence_count < 1:
            raise ValueError(
                f"ScoredGap {self.topic_key!r} recurrence_count must be >= 1, "
                f"got {self.recurrence_count}"
            )
        if self.severity < 0:
            raise ValueError(
                f"ScoredGap {self.topic_key!r} severity must be >= 0, got {self.severity}"
            )


def score_gap_candidates(candidates: list[GapCandidate]) -> list[ScoredGap]:
    """Group, filter, and score a pool of GapCandidates spanning one or more transcripts.

    Steps:
    1. Keep only candidates where `should_flag` is True -- passes all 3
       gates AND isn't hard-excluded by opted_out/boilerplate/
       already_understood.
    2. Group survivors by `topic_key` (exact match after strip+lowercase;
       see module docstring re: day 4 clustering).
    3. Per group: `recurrence_count` = number of occurrences,
       `foundational` = True if ANY occurrence was tagged foundational
       (favors recall -- a topic worth teaching once is worth teaching
       even if not every occurrence got tagged that way),
       `all_incident_context` = True only if EVERY occurrence in the group
       was incident-context (a topic seen once during an incident and
       once during unpressured exploration isn't purely a firefighting
       artifact, so it doesn't get down-weighted).
    4. `severity = BASE_COMPLEXITY_WEIGHT * recurrence_count *
       (FOUNDATIONAL_WEIGHT if foundational else NICHE_WEIGHT) *
       (INCIDENT_CONTEXT_MULTIPLIER if all_incident_context else 1.0)`.
    5. Return groups sorted by severity descending, ties broken
       alphabetically by topic_key for deterministic output.

    `display_topic` on each ScoredGap is the topic string (original
    casing/whitespace) from the group's first candidate -- purely
    cosmetic, grouping itself uses `topic_key`.

    Empty input returns an empty list. A pool where nothing survives
    should_flag filtering also returns an empty list (not an error) --
    "no gaps found this week" is a valid, expected outcome, not a failure.
    """
    flagged = [c for c in candidates if c.should_flag]

    groups: dict[str, list[GapCandidate]] = {}
    for c in flagged:
        groups.setdefault(c.topic_key, []).append(c)

    scored: list[ScoredGap] = []
    for topic_key, group in groups.items():
        recurrence_count = len(group)
        foundational = any(c.foundational for c in group)
        all_incident_context = all(c.incident_context for c in group)

        foundational_weight = FOUNDATIONAL_WEIGHT if foundational else NICHE_WEIGHT
        incident_multiplier = INCIDENT_CONTEXT_MULTIPLIER if all_incident_context else 1.0
        severity = BASE_COMPLEXITY_WEIGHT * recurrence_count * foundational_weight * incident_multiplier

        scored.append(
            ScoredGap(
                topic_key=topic_key,
                display_topic=group[0].topic,
                candidates=tuple(group),
                recurrence_count=recurrence_count,
                foundational=foundational,
                all_incident_context=all_incident_context,
                severity=severity,
            )
        )

    scored.sort(key=lambda s: (-s.severity, s.topic_key))
    return scored
