"""Topic clustering: DESIGN.md pipeline step 3, "dedup + cluster".

`scoring.py` groups `GapCandidate`s by `GapCandidate.topic_key`, an
exact-match normalization (stripped + lowercased) -- documented there as a
deliberate simplification ahead of this module. Exact match fragments
recurrence: "SQL joins" and "joining tables in SQL" are the same underlying
gap but don't share a topic_key, so a topic a user hit twice under two
different phrasings looks like two single-occurrence gaps instead of one
recurring one. DESIGN.md step 3 asks for real grouping by topic, not just
string equality.

Deciding whether two topic strings describe the same underlying concept is
a judgment call a deterministic parser can't make reliably (same category
of problem as gap_detection.py's gating decisions), so -- following the
pattern already established there -- this module asks Claude to do the
clustering, via a forced tool-use call, and validates the structure of
whatever comes back before trusting it.

Scope for this module: given a pool of `GapCandidate`s spanning one or more
transcripts, filter to the ones that survive `should_flag` (same filter
`scoring.py` applies), cluster their *distinct* topic_key values by
underlying concept, and regroup the original candidates by cluster. This
module does NOT compute severity/recurrence scoring itself -- `TopicCluster`
exposes the grouped candidates so a caller can run them back through
`scoring.py`-style aggregation, or a later day can wire the two together.
Recomputing scoring.py's severity formula on top of clusters instead of
exact-match topic_key is left for whichever day actually wires the full
pipeline together (BUILD-SCHEDULE.md day 8, CLI wiring) rather than done
speculatively here.

Same caveat as gap_detection.py: no ANTHROPIC_API_KEY is available in the
build sandbox this module was written in, so `AnthropicClusteringClient`'s
request/response plumbing is verified against a mocked SDK client only,
not a live call. `build_clustering_prompt` + `parse_topic_clusters` +
`cluster_gap_candidates` (fed by a fake client) are the parts actually
exercised by the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .gap_detection import GapCandidate

__all__ = [
    "TopicCluster",
    "ClusteringResponseError",
    "ClusteringClient",
    "AnthropicClusteringClient",
    "build_clustering_prompt",
    "parse_topic_clusters",
    "cluster_gap_candidates",
    "CLUSTER_TOOL_SCHEMA",
    "DEFAULT_MODEL",
]

DEFAULT_MODEL = "claude-sonnet-5"

# Forced tool-use schema: Claude must call this "tool" with an argument
# matching this shape. Every input topic_key must appear in exactly one
# cluster's member_topics -- parse_topic_clusters enforces that as a
# structural check, it isn't just a hopeful description in the prompt.
CLUSTER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "clusters": {
            "type": "array",
            "description": (
                "Every input topic grouped into clusters of topics that "
                "describe the same underlying concept. Every input topic "
                "must appear in exactly one cluster's member_topics -- "
                "don't drop any, don't invent new ones, don't put the same "
                "topic in two clusters. A topic with no match to any other "
                "still gets its own single-member cluster."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "canonical_topic": {
                        "type": "string",
                        "description": (
                            "Short label for the underlying concept this "
                            "cluster represents, e.g. 'SQL joins'. Should "
                            "read naturally as a lesson topic title."
                        ),
                    },
                    "member_topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Every input topic string that belongs to this "
                            "cluster, copied verbatim from the input list."
                        ),
                    },
                },
                "required": ["canonical_topic", "member_topics"],
            },
        }
    },
    "required": ["clusters"],
}


class ClusteringResponseError(ValueError):
    """Raised when a clustering API response doesn't match the expected shape."""


@dataclass(frozen=True)
class TopicCluster:
    """One underlying concept and every GapCandidate that maps to it.

    `member_topic_keys` holds the (normalized) topic_key strings the
    clustering step decided belong together -- may be a single topic_key
    if nothing else matched it. `candidates` holds every GapCandidate
    across the input pool whose topic_key is in `member_topic_keys`, so a
    cluster's total recurrence is `len(candidates)`, potentially summed
    across what used to be several separate exact-match groups in
    scoring.py.
    """

    canonical_topic: str
    member_topic_keys: tuple[str, ...]
    candidates: tuple[GapCandidate, ...]

    def __post_init__(self) -> None:
        if not self.canonical_topic.strip():
            raise ValueError("TopicCluster requires a non-empty canonical_topic")
        if not self.member_topic_keys:
            raise ValueError(
                f"TopicCluster {self.canonical_topic!r} requires at least one member_topic_key"
            )
        if not self.candidates:
            raise ValueError(
                f"TopicCluster {self.canonical_topic!r} requires at least one candidate"
            )
        member_set = set(self.member_topic_keys)
        stray = [c for c in self.candidates if c.topic_key not in member_set]
        if stray:
            raise ValueError(
                f"TopicCluster {self.canonical_topic!r} has candidate(s) with "
                f"topic_key(s) not in member_topic_keys: "
                f"{sorted({c.topic_key for c in stray})}"
            )

    @property
    def recurrence_count(self) -> int:
        return len(self.candidates)


class ClusteringClient(Protocol):
    """Anything that can cluster a list of distinct topic strings by concept.

    Implementations return the raw dict matching CLUSTER_TOOL_SCHEMA --
    i.e. `{"clusters": [...]}` -- *before* validation.
    `parse_topic_clusters` does the validation, so both the real client
    and any fake/mock used in tests share the same validation path.
    """

    def cluster(self, topics: list[str]) -> dict[str, Any]: ...


class AnthropicClusteringClient:
    """Real ClusteringClient backed by the Anthropic Messages API.

    NOT exercised against a live API in this build sandbox (no
    ANTHROPIC_API_KEY available here) -- see module docstring. The
    `anthropic` import is deferred to __init__, same reasoning as
    AnthropicGapDetectionClient in gap_detection.py.
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

    def cluster(self, topics: list[str]) -> dict[str, Any]:
        prompt = build_clustering_prompt(topics)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            tools=[
                {
                    "name": "report_topic_clusters",
                    "description": (
                        "Report every input topic grouped into clusters of "
                        "topics describing the same underlying concept."
                    ),
                    "input_schema": CLUSTER_TOOL_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "report_topic_clusters"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use" and getattr(block, "name", None) == "report_topic_clusters":
                return block.input
        raise ClusteringResponseError(
            "Anthropic response did not contain the expected report_topic_clusters tool_use block"
        )


def build_clustering_prompt(topics: list[str]) -> str:
    """Build the user-turn prompt sent to Claude for topic clustering.

    Deterministic and side-effect-free so it can be unit tested without any
    API access. `topics` is expected to already be a list of distinct
    strings (deduped topic_key values) -- this function doesn't dedupe.
    """
    lines = [
        "You are grouping a list of short topic labels by underlying "
        "concept. Each label came from a different moment where a user "
        "delegated a task to an AI assistant without engaging with it. "
        "Some labels describe the same underlying concept in different "
        "words (e.g. 'SQL joins' and 'joining tables in SQL' are the same "
        "concept; 'git rebase' and 'git merge conflicts' are NOT the same "
        "concept, even though both are git). Group conservatively -- only "
        "merge topics you're confident describe the same underlying idea, "
        "not just the same general area.",
        "",
        "Every topic below must end up in exactly one cluster's "
        "member_topics list, copied verbatim. A topic with nothing else "
        "like it still gets its own single-member cluster. Do not invent "
        "topics that aren't in the list below, and do not drop any.",
        "",
        "Topics:",
    ]
    for t in topics:
        lines.append(f"- {t}")
    return "\n".join(lines)


def parse_topic_clusters(raw: dict[str, Any], input_topics: list[str]) -> list[dict[str, Any]]:
    """Validate a raw cluster() response against the set of topics it was asked to cluster.

    Returns a list of `{"canonical_topic": str, "member_topics": list[str]}`
    dicts (not yet joined against GapCandidates -- see
    `cluster_gap_candidates` for that). Raises ClusteringResponseError with
    a specific message for any structural problem: wrong shape, a member
    topic that wasn't in `input_topics` (hallucinated), an input topic
    missing from every cluster (dropped), or an input topic appearing in
    more than one cluster (double-counted).
    """
    if not isinstance(raw, dict):
        raise ClusteringResponseError("clustering response must be an object")

    if "clusters" not in raw:
        raise ClusteringResponseError("clustering response missing 'clusters'")

    raw_clusters = raw["clusters"]
    if not isinstance(raw_clusters, list):
        raise ClusteringResponseError("'clusters' must be a list")

    input_set = set(input_topics)
    seen_topics: set[str] = set()
    clusters: list[dict[str, Any]] = []

    for i, raw_cluster in enumerate(raw_clusters):
        if not isinstance(raw_cluster, dict):
            raise ClusteringResponseError(f"clusters[{i}] must be an object")

        missing = [f for f in ("canonical_topic", "member_topics") if f not in raw_cluster]
        if missing:
            raise ClusteringResponseError(
                f"clusters[{i}] missing required field(s): {', '.join(missing)}"
            )

        canonical_topic = raw_cluster["canonical_topic"]
        if not isinstance(canonical_topic, str) or not canonical_topic.strip():
            raise ClusteringResponseError(
                f"clusters[{i}].canonical_topic must be a non-empty string"
            )

        member_topics = raw_cluster["member_topics"]
        if not isinstance(member_topics, list) or not member_topics:
            raise ClusteringResponseError(
                f"clusters[{i}].member_topics must be a non-empty list"
            )
        for j, member in enumerate(member_topics):
            if not isinstance(member, str) or not member.strip():
                raise ClusteringResponseError(
                    f"clusters[{i}].member_topics[{j}] must be a non-empty string"
                )
            if member not in input_set:
                raise ClusteringResponseError(
                    f"clusters[{i}].member_topics[{j}] {member!r} was not in the "
                    f"input topic list -- clustering must not invent topics"
                )
            if member in seen_topics:
                raise ClusteringResponseError(
                    f"topic {member!r} appears in more than one cluster"
                )
            seen_topics.add(member)

        clusters.append({"canonical_topic": canonical_topic, "member_topics": list(member_topics)})

    missing_topics = input_set - seen_topics
    if missing_topics:
        raise ClusteringResponseError(
            f"input topic(s) missing from every cluster: {sorted(missing_topics)}"
        )

    return clusters


def cluster_gap_candidates(
    candidates: list[GapCandidate], client: ClusteringClient
) -> list[TopicCluster]:
    """Cluster a pool of GapCandidates (spanning one or more transcripts) by topic.

    Steps:
    1. Filter to `should_flag` candidates only -- same gate `scoring.py`
       applies. Clustering non-gaps (opted-out, boilerplate, etc.) has no
       purpose since they'll never be scored or scripted.
    2. Collect distinct `topic_key` values across the survivors, in first-
       seen order (deterministic prompt, and a stable tie-break for
       `parse_topic_clusters`' error messages).
    3. Call `client.cluster(...)` with that list and validate the response
       via `parse_topic_clusters`.
    4. Regroup the original candidates: every candidate whose topic_key is
       a member of a cluster becomes part of that cluster's `TopicCluster`.

    Returns an empty list if no candidates survive the should_flag filter
    -- "nothing to cluster" is a valid outcome, not an error, matching
    `scoring.score_gap_candidates`'s handling of the same case. When
    exactly one distinct topic_key survives, `client.cluster` is still
    called (a single-topic pool can't be assumed to need no clustering --
    the client is the one source of truth for what counts as "the same
    topic", not this function).
    """
    flagged = [c for c in candidates if c.should_flag]
    if not flagged:
        return []

    topic_keys: list[str] = []
    seen: set[str] = set()
    for c in flagged:
        if c.topic_key not in seen:
            seen.add(c.topic_key)
            topic_keys.append(c.topic_key)

    raw = client.cluster(topic_keys)
    parsed = parse_topic_clusters(raw, topic_keys)

    candidates_by_topic_key: dict[str, list[GapCandidate]] = {}
    for c in flagged:
        candidates_by_topic_key.setdefault(c.topic_key, []).append(c)

    result: list[TopicCluster] = []
    for entry in parsed:
        member_topic_keys = tuple(entry["member_topics"])
        cluster_candidates: list[GapCandidate] = []
        for key in member_topic_keys:
            cluster_candidates.extend(candidates_by_topic_key[key])
        result.append(
            TopicCluster(
                canonical_topic=entry["canonical_topic"],
                member_topic_keys=member_topic_keys,
                candidates=tuple(cluster_candidates),
            )
        )

    return result
