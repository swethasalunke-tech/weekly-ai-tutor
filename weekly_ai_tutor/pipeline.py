"""Full pipeline wiring: DESIGN.md pipeline steps 1-5, BUILD-SCHEDULE.md day 8 "CLI wiring".

Ties together, in order: `ingest.load_transcripts_from_dir` -> for each
transcript, `gap_detection.detect_gaps` -> `clustering.cluster_gap_candidates`
over the pooled candidates -> `curriculum.fit_curriculum` against a time
budget -> `script_generation.generate_script` per selected topic.

## The one piece of wiring logic that lives here (not in an earlier module)

`clustering.py`'s own docstring explicitly defers this: "Recomputing
scoring.py's severity formula on top of clusters instead of exact-match
topic_key is left for whichever day actually wires the full pipeline
together (BUILD-SCHEDULE.md day 8, CLI wiring)." `scoring.py`'s
`score_gap_candidates` only groups by exact-match `topic_key` -- it has no
notion of a `TopicCluster`, which may merge several different topic_key
strings that clustering decided describe the same concept. `score_topic_cluster`
below is that missing piece: the exact same severity formula from
`scoring.py` (`BASE_COMPLEXITY_WEIGHT * recurrence_count *
foundational_weight * incident_multiplier`), applied over a cluster's full
candidate pool instead of a single exact-match group.

## What this module does NOT do

- Render audio (TTS, day 7) or assemble/deliver a final episode (mp3 +
  show-notes, DESIGN.md steps 6-7). BUILD-SCHEDULE.md day 8 scopes CLI
  wiring to "ties ingestion through script generation into one command" --
  audio rendering on top of this module's `EpisodePlan.scripts` is left for
  a later day, same deferral pattern already used between curriculum.py and
  script_generation.py.
- Construct real Anthropic-backed clients. `run_pipeline` takes
  `GapDetectionClient` / `ClusteringClient` / `ScriptGenerationClient`
  instances as required arguments -- it has no code path that reaches for
  `ANTHROPIC_API_KEY` itself, so it can be fully exercised in tests (and in
  this build sandbox, which has no live key) with fakes. `cli.py` is where
  the real `Anthropic*Client` instances get constructed for an actual run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .clustering import ClusteringClient, TopicCluster, cluster_gap_candidates
from .curriculum import CurriculumCandidate, fit_curriculum
from .gap_detection import GapCandidate, GapDetectionClient, detect_gaps
from .ingest import load_transcripts_from_dir
from .schema import Transcript
from .script_generation import ScriptGenerationClient, TopicScript, generate_script
from .scoring import (
    BASE_COMPLEXITY_WEIGHT,
    FOUNDATIONAL_WEIGHT,
    INCIDENT_CONTEXT_MULTIPLIER,
    NICHE_WEIGHT,
)

__all__ = ["EpisodePlan", "score_topic_cluster", "run_pipeline"]


@dataclass(frozen=True)
class EpisodePlan:
    """The end-to-end result of running the pipeline once: ingest through scripts.

    `scripts` is one `TopicScript` per selected/allocated topic, in the same
    priority order as `CurriculumPlan.topics` (highest severity first).
    `carried_over_topics` is the `canonical_topic` label of every cluster
    that didn't make it into this week's episode (DESIGN.md: "carried to
    next week's backlog") -- kept as plain labels rather than full
    `TopicCluster`/`CurriculumCandidate` objects, since nothing downstream of
    this function needs more than the label to report what got bumped.

    `transcripts_loaded` and `gap_candidates_found` are diagnostic counts
    (not used in any allocation decision) so a caller/CLI can report "found
    0 gap candidates" vs. "found 12, but all were filtered out" as distinct,
    honest outcomes rather than collapsing both into "no topics this week."
    """

    total_minutes: float
    intro_outro_minutes: float
    scripts: tuple[TopicScript, ...]
    carried_over_topics: tuple[str, ...]
    unallocated_minutes: float
    transcripts_loaded: int
    gap_candidates_found: int


def score_topic_cluster(cluster: TopicCluster) -> float:
    """Apply scoring.py's severity formula to a TopicCluster.

    `recurrence_count` is `len(cluster.candidates)` (may span what used to
    be several separate exact-match `topic_key` groups, now merged by
    clustering.py's semantic grouping). `foundational` is True if ANY
    candidate in the cluster was tagged foundational (same "favors recall"
    reasoning as `scoring.score_gap_candidates`). `all_incident_context` is
    True only if EVERY candidate in the cluster was incident-context (same
    reasoning: a topic seen once during an incident and once during
    unpressured exploration isn't purely a firefighting artifact).

    Returns `BASE_COMPLEXITY_WEIGHT * recurrence_count *
    (FOUNDATIONAL_WEIGHT if foundational else NICHE_WEIGHT) *
    (INCIDENT_CONTEXT_MULTIPLIER if all_incident_context else 1.0)` -- the
    identical formula `scoring.score_gap_candidates` uses, just computed
    over a cluster's pooled candidates instead of an exact-match group.
    """
    recurrence_count = cluster.recurrence_count
    foundational = any(c.foundational for c in cluster.candidates)
    all_incident_context = all(c.incident_context for c in cluster.candidates)

    foundational_weight = FOUNDATIONAL_WEIGHT if foundational else NICHE_WEIGHT
    incident_multiplier = INCIDENT_CONTEXT_MULTIPLIER if all_incident_context else 1.0
    return BASE_COMPLEXITY_WEIGHT * recurrence_count * foundational_weight * incident_multiplier


def run_pipeline(
    transcripts_dir: str | Path,
    total_minutes: float,
    gap_client: GapDetectionClient,
    clustering_client: ClusteringClient,
    script_client: ScriptGenerationClient,
) -> EpisodePlan:
    """Run DESIGN.md pipeline steps 1-5 end to end against a directory of transcripts.

    Steps, in order:

    1. `ingest.load_transcripts_from_dir(transcripts_dir)` -- bad/invalid
       files are skipped with a stderr warning, not fatal (see ingest.py).
    2. `gap_detection.detect_gaps(transcript, gap_client)` for every loaded
       transcript, pooling all returned candidates (flagged and not --
       filtering happens downstream, same contract `detect_gaps` documents).
    3. `clustering.cluster_gap_candidates(all_candidates, clustering_client)`
       over the full pool -- this applies `should_flag` filtering internally,
       so gate-failing/non-gate-excluded candidates never reach clustering.
    4. For each resulting `TopicCluster`, `score_topic_cluster` computes a
       priority, and `curriculum.fit_curriculum` greedily allocates minutes
       against `total_minutes`.
    5. `script_generation.generate_script` for each selected
       `CurriculumSlot`, using its cluster's candidate `description` strings
       (already paraphrased by gap_detection.py's own prompt) as
       `source_descriptions` -- never raw transcript content, preserving the
       privacy-by-API-design guarantee `script_generation.py` documents.

    Raises ValueError if `total_minutes` <= 0 (same validation
    `fit_curriculum` applies, checked here too so the error surfaces before
    any transcripts are even loaded).

    Returns an `EpisodePlan` with empty `scripts` (and `carried_over_topics`
    accordingly empty) if no transcripts are found, no candidate survives
    gating, or clustering finds nothing to cluster -- "nothing to teach this
    week" is a valid outcome at every stage of this pipeline (see
    ingest.py/scoring.py/clustering.py/curriculum.py's own handling of empty
    input), never an error.
    """
    if total_minutes <= 0:
        raise ValueError(f"run_pipeline total_minutes must be > 0, got {total_minutes}")

    transcripts: list[Transcript] = load_transcripts_from_dir(transcripts_dir)

    all_candidates: list[GapCandidate] = []
    for transcript in transcripts:
        all_candidates.extend(detect_gaps(transcript, gap_client))

    clusters = cluster_gap_candidates(all_candidates, clustering_client)

    curriculum_candidates = [
        CurriculumCandidate(
            label=cluster.canonical_topic,
            priority=score_topic_cluster(cluster),
            recurrence_count=cluster.recurrence_count,
        )
        for cluster in clusters
    ]
    plan = fit_curriculum(curriculum_candidates, total_minutes)

    clusters_by_label = {cluster.canonical_topic: cluster for cluster in clusters}
    scripts: list[TopicScript] = []
    for slot in plan.topics:
        cluster = clusters_by_label[slot.candidate.label]
        source_descriptions = [c.description for c in cluster.candidates]
        scripts.append(
            generate_script(
                topic=cluster.canonical_topic,
                source_descriptions=source_descriptions,
                minutes=slot.minutes,
                client=script_client,
            )
        )

    return EpisodePlan(
        total_minutes=plan.total_minutes,
        intro_outro_minutes=plan.intro_outro_minutes,
        scripts=tuple(scripts),
        carried_over_topics=tuple(c.label for c in plan.carried_over),
        unallocated_minutes=plan.unallocated_minutes,
        transcripts_loaded=len(transcripts),
        gap_candidates_found=len(all_candidates),
    )
