"""Curriculum fitting: DESIGN.md's "Curriculum fitting" section, BUILD-SCHEDULE.md day 5.

DESIGN.md's spec: given a ranked list of gap clusters and a time budget T
minutes, reserve ~10% for intro/outro, greedily select topics
highest-severity-first until remaining time can't fit another topic at a
~8-min floor, and cap at 4-5 topics per episode regardless of budget.

This module deliberately does NOT import `TopicCluster` (clustering.py) or
`ScoredGap` (scoring.py). Neither upstream type carries what this module
needs in one place: `TopicCluster` has no severity/priority field --
clustering.py's own docstring defers "recomputing scoring.py's severity
formula on top of clusters instead of exact-match topic_key" to
BUILD-SCHEDULE.md day 8 (CLI wiring), and `ScoredGap` is keyed by exact-match
`topic_key`, not a semantic cluster. Rather than reach ahead and invent that
wiring now, this module accepts a small, self-contained `CurriculumCandidate`
(label + priority + recurrence_count) that a day-8 caller can construct from
either upstream type -- the allocator itself doesn't need to know where the
priority number came from, only that higher means "teach this first."

Two numeric decisions from DESIGN.md needed a concrete value where the doc
only gives a qualitative range or no value at all -- both are named
constants below with the reasoning attached, not buried magic numbers:

- `MAX_TOPICS_CAP = 5`: DESIGN.md says "cap at a max of 4-5 topics... regardless
  of budget." 5 is the outer cap; the 8-min floor plus limited budgets is
  what naturally produces episodes with fewer topics (e.g. a 30-min budget
  can fit at most 3 topics at floor depth after the intro/outro reserve),
  so a single fixed cap of 5 covers the stated "4-5" range without a second,
  redundant knob.
- `MAX_TOPIC_MINUTES = 12.0`: DESIGN.md's per-topic script template (hook
  30-60s + concept 3-5min + example 3-5min + next-time 1min) sums to a
  ceiling of 1 + 5 + 5 + 1 = 12 minutes at the upper bound of every part.
  Without a per-topic ceiling, a generous time budget with few high-priority
  candidates would dump the entire remainder into one topic, which
  contradicts the script template's own shape.

What this module does NOT do: merge related topics that don't fit (DESIGN.md:
"a topic gets merged with a related one or dropped this week" -- the merge
option describes what clustering.py already does *before* this module runs,
or what a future day/human curator could do with `carried_over`; this
module's only decision is allocate-or-drop). It also doesn't compute
severity/priority itself -- see above.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CurriculumCandidate",
    "CurriculumSlot",
    "CurriculumPlan",
    "fit_curriculum",
    "INTRO_OUTRO_RESERVE_FRACTION",
    "MIN_TOPIC_MINUTES",
    "MAX_TOPIC_MINUTES",
    "MAX_TOPICS_CAP",
]

# DESIGN.md: "Reserve ~10% for intro/outro."
INTRO_OUTRO_RESERVE_FRACTION = 0.10

# DESIGN.md: "~8 min floor per topic -- below that, a topic gets merged with
# a related one or dropped this week and carried to next week's backlog."
MIN_TOPIC_MINUTES = 8.0

# See module docstring: derived from the script template's upper bounds
# (1 + 5 + 5 + 1 minutes).
MAX_TOPIC_MINUTES = 12.0

# See module docstring: DESIGN.md's "4-5 topics" range, expressed as a
# single outer cap.
MAX_TOPICS_CAP = 5


@dataclass(frozen=True)
class CurriculumCandidate:
    """One gap cluster competing for a slot in this week's episode.

    Deliberately decoupled from `TopicCluster`/`ScoredGap` -- see module
    docstring. `label` should already be a human-readable topic title (e.g.
    `TopicCluster.canonical_topic` or `ScoredGap.display_topic`); `priority`
    is whatever ranking score the caller computed (e.g. `ScoredGap.severity`);
    `recurrence_count` is carried through only for reporting/show-notes use,
    not used in the allocation decision itself (priority already reflects it
    upstream).
    """

    label: str
    priority: float
    recurrence_count: int = 1

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("CurriculumCandidate requires a non-empty label")
        if self.priority < 0:
            raise ValueError(
                f"CurriculumCandidate {self.label!r} priority must be >= 0, got {self.priority}"
            )
        if self.recurrence_count < 1:
            raise ValueError(
                f"CurriculumCandidate {self.label!r} recurrence_count must be >= 1, "
                f"got {self.recurrence_count}"
            )


@dataclass(frozen=True)
class CurriculumSlot:
    """One selected topic and the depth (minutes) it was allocated."""

    candidate: CurriculumCandidate
    minutes: float

    def __post_init__(self) -> None:
        if not (MIN_TOPIC_MINUTES - 1e-9 <= self.minutes <= MAX_TOPIC_MINUTES + 1e-9):
            raise ValueError(
                f"CurriculumSlot {self.candidate.label!r} minutes ({self.minutes}) "
                f"must be within [{MIN_TOPIC_MINUTES}, {MAX_TOPIC_MINUTES}]"
            )


@dataclass(frozen=True)
class CurriculumPlan:
    """The full result of fitting a ranked candidate pool to a time budget.

    `topics` is selected-and-allocated, highest priority first.
    `carried_over` is every candidate that didn't make it in, in the same
    priority order, so a caller can report "N topics carried to next week"
    or feed them back in as next week's starting pool (DESIGN.md: "carried
    to next week's backlog").

    Invariant checked here (not just assumed): `intro_outro_minutes` +
    sum(slot.minutes) + `unallocated_minutes` == `total_minutes` (within
    floating-point tolerance), and `len(topics) <= MAX_TOPICS_CAP`.
    """

    total_minutes: float
    intro_outro_minutes: float
    topics: tuple[CurriculumSlot, ...]
    carried_over: tuple[CurriculumCandidate, ...]
    unallocated_minutes: float

    def __post_init__(self) -> None:
        if self.total_minutes <= 0:
            raise ValueError(f"CurriculumPlan total_minutes must be > 0, got {self.total_minutes}")
        if self.intro_outro_minutes < 0:
            raise ValueError(
                f"CurriculumPlan intro_outro_minutes must be >= 0, got {self.intro_outro_minutes}"
            )
        if self.unallocated_minutes < -1e-9:
            raise ValueError(
                f"CurriculumPlan unallocated_minutes must be >= 0, got {self.unallocated_minutes}"
            )
        if len(self.topics) > MAX_TOPICS_CAP:
            raise ValueError(
                f"CurriculumPlan has {len(self.topics)} topics, exceeds MAX_TOPICS_CAP "
                f"({MAX_TOPICS_CAP})"
            )
        accounted = (
            self.intro_outro_minutes
            + sum(slot.minutes for slot in self.topics)
            + self.unallocated_minutes
        )
        if abs(accounted - self.total_minutes) > 1e-6:
            raise ValueError(
                f"CurriculumPlan minutes don't reconcile: intro_outro "
                f"({self.intro_outro_minutes}) + topics "
                f"({sum(slot.minutes for slot in self.topics)}) + unallocated "
                f"({self.unallocated_minutes}) = {accounted}, expected total_minutes "
                f"({self.total_minutes})"
            )

    @property
    def topic_count(self) -> int:
        return len(self.topics)


def fit_curriculum(
    candidates: list[CurriculumCandidate], total_minutes: float
) -> CurriculumPlan:
    """Greedily fit a ranked pool of candidates into a total_minutes time budget.

    Steps (mirrors DESIGN.md's "Curriculum fitting" section exactly):

    1. Reserve `INTRO_OUTRO_RESERVE_FRACTION` of `total_minutes` for
       intro/outro.
    2. Sort candidates by `priority` descending; ties broken alphabetically
       by `label` for deterministic output (same tie-break convention as
       `scoring.score_gap_candidates`).
    3. Walk the sorted list, giving each candidate `MIN_TOPIC_MINUTES` (the
       floor) as long as: the remaining budget can still afford the floor,
       AND the selected count hasn't hit `MAX_TOPICS_CAP`. The first
       candidate that fails either check, and everything after it in
       priority order, goes to `carried_over` -- this is a greedy
       highest-priority-first fill, not a knapsack optimization, matching
       DESIGN.md's "greedily select" language.
    4. Whatever budget remains after every selected topic has its floor is
       distributed back across the selected topics in priority order, each
       topping up to at most `MAX_TOPIC_MINUTES`, until either the leftover
       runs out or every topic is maxed. Depth over breadth: the
       highest-priority topics get the extra depth first.
    5. Any leftover that remains even after every selected topic is maxed
       out (a generous budget with few/short candidate pools) is reported
       as `unallocated_minutes` rather than silently dropped or force-fed
       into one topic past `MAX_TOPIC_MINUTES`.

    Raises ValueError if `total_minutes` <= 0, or if two candidates share
    the same `label` (a caller bug -- candidates are expected to already be
    distinct clusters by the time they reach this module).

    Empty `candidates` returns a plan with no topics, the full budget in
    `unallocated_minutes` (after the intro/outro reserve), and an empty
    `carried_over` -- "nothing to teach this week" is a valid outcome, not
    an error, matching the rest of this pipeline's handling of empty pools.
    """
    if total_minutes <= 0:
        raise ValueError(f"fit_curriculum total_minutes must be > 0, got {total_minutes}")

    labels_seen: dict[str, CurriculumCandidate] = {}
    for c in candidates:
        if c.label in labels_seen:
            raise ValueError(f"duplicate CurriculumCandidate label: {c.label!r}")
        labels_seen[c.label] = c

    ranked = sorted(candidates, key=lambda c: (-c.priority, c.label))

    intro_outro_minutes = total_minutes * INTRO_OUTRO_RESERVE_FRACTION
    remaining = total_minutes - intro_outro_minutes

    selected: list[CurriculumCandidate] = []
    carried_over: list[CurriculumCandidate] = []
    for c in ranked:
        if len(selected) < MAX_TOPICS_CAP and remaining >= MIN_TOPIC_MINUTES - 1e-9:
            selected.append(c)
            remaining -= MIN_TOPIC_MINUTES
        else:
            carried_over.append(c)

    # remaining may have gone slightly negative due to floating point on the
    # last floor allocation; clamp it to 0 before distributing leftover.
    leftover = max(remaining, 0.0)

    depths: dict[str, float] = {c.label: MIN_TOPIC_MINUTES for c in selected}
    for c in selected:
        if leftover <= 1e-9:
            break
        headroom = MAX_TOPIC_MINUTES - depths[c.label]
        top_up = min(headroom, leftover)
        if top_up > 0:
            depths[c.label] += top_up
            leftover -= top_up

    topics = tuple(CurriculumSlot(candidate=c, minutes=depths[c.label]) for c in selected)

    return CurriculumPlan(
        total_minutes=total_minutes,
        intro_outro_minutes=intro_outro_minutes,
        topics=topics,
        carried_over=tuple(carried_over),
        unallocated_minutes=leftover,
    )
