import json
from pathlib import Path

import pytest

from weekly_ai_tutor.clustering import TopicCluster
from weekly_ai_tutor.gap_detection import GapCandidate
from weekly_ai_tutor.pipeline import EpisodePlan, run_pipeline, score_topic_cluster
from weekly_ai_tutor.scoring import (
    BASE_COMPLEXITY_WEIGHT,
    FOUNDATIONAL_WEIGHT,
    INCIDENT_CONTEXT_MULTIPLIER,
    NICHE_WEIGHT,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _candidate(**overrides) -> GapCandidate:
    base = dict(
        session_id="s1",
        topic="git rebase",
        description="User hit a rebase conflict and accepted the fix without asking why.",
        user_message_index=0,
        non_trivial_delegation=True,
        no_engagement_signal=True,
        immediate_accept=True,
        reasoning="r",
    )
    base.update(overrides)
    return GapCandidate(**base)


def _cluster(canonical_topic="git rebase", **candidate_overrides_list) -> TopicCluster:
    """Build a single-member-topic-key TopicCluster from one or more candidates.

    `candidate_overrides_list` isn't used directly -- see _cluster_from for
    the multi-candidate version. This convenience builds a one-candidate
    cluster from **overrides applied to a single default candidate.
    """
    candidate = _candidate(topic=canonical_topic, **candidate_overrides_list)
    return TopicCluster(
        canonical_topic=canonical_topic,
        member_topic_keys=(candidate.topic_key,),
        candidates=(candidate,),
    )


def _write_transcript(path, session_id, title, messages):
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "title": title,
                "started_at": "2026-08-17T10:00:00Z",
                "messages": messages,
            }
        ),
        encoding="utf-8",
    )


class FakeGapDetectionClient:
    """Test double for GapDetectionClient -- returns a canned response per session_id."""

    def __init__(self, responses_by_session_id):
        self.responses_by_session_id = responses_by_session_id
        self.calls = []

    def classify(self, transcript):
        self.calls.append(transcript.session_id)
        return self.responses_by_session_id[transcript.session_id]


class FakeClusteringClient:
    """Test double for ClusteringClient -- returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def cluster(self, topics):
        self.calls.append(list(topics))
        return self.response


class FakeScriptGenerationClient:
    """Test double for ScriptGenerationClient -- returns a fixed valid section set per topic."""

    def __init__(self):
        self.calls = []

    def generate(self, topic, source_descriptions, minutes):
        self.calls.append((topic, list(source_descriptions), minutes))
        return {
            "hook": f"On a recent day you asked Claude about {topic} and accepted the fix as-is.",
            "concept": f"{topic}, explained from first principles independent of your specific code.",
            "example": f"A clean, minimal worked example illustrating {topic}.",
            "next_time": f"Next time, watch for {topic} and ask Claude to walk you through it.",
        }


def _gap_response(topic, **overrides):
    base = dict(
        topic=topic,
        description=f"User delegated a fix involving {topic} and accepted it without engaging.",
        user_message_index=0,
        non_trivial_delegation=True,
        no_engagement_signal=True,
        immediate_accept=True,
        reasoning="matches all 3 gates",
    )
    base.update(overrides)
    return {"candidates": [base]}


def _no_candidates_response():
    return {"candidates": []}


# ---------------------------------------------------------------------------
# score_topic_cluster
# ---------------------------------------------------------------------------


def test_score_topic_cluster_matches_scoring_formula_niche_no_incident():
    cluster = _cluster("regex lookahead", foundational=False, incident_context=False)
    expected = BASE_COMPLEXITY_WEIGHT * 1 * NICHE_WEIGHT * 1.0
    assert score_topic_cluster(cluster) == expected


def test_score_topic_cluster_foundational_weight_applied():
    cluster = _cluster("SQL joins", foundational=True, incident_context=False)
    expected = BASE_COMPLEXITY_WEIGHT * 1 * FOUNDATIONAL_WEIGHT * 1.0
    assert score_topic_cluster(cluster) == expected


def test_score_topic_cluster_foundational_true_if_any_candidate_is():
    c1 = _candidate(topic="git rebase", session_id="s1", foundational=True)
    c2 = _candidate(topic="git rebase", session_id="s2", foundational=False)
    cluster = TopicCluster(
        canonical_topic="git rebase",
        member_topic_keys=("git rebase",),
        candidates=(c1, c2),
    )
    expected = BASE_COMPLEXITY_WEIGHT * 2 * FOUNDATIONAL_WEIGHT * 1.0
    assert score_topic_cluster(cluster) == expected


def test_score_topic_cluster_incident_multiplier_needs_all_candidates():
    c1 = _candidate(topic="oauth refresh", session_id="s1", incident_context=True)
    c2 = _candidate(topic="oauth refresh", session_id="s2", incident_context=False)
    mixed_cluster = TopicCluster(
        canonical_topic="oauth refresh",
        member_topic_keys=("oauth refresh",),
        candidates=(c1, c2),
    )
    # Not every candidate was incident-context -> no down-weight.
    assert score_topic_cluster(mixed_cluster) == BASE_COMPLEXITY_WEIGHT * 2 * NICHE_WEIGHT * 1.0

    c3 = _candidate(topic="oauth refresh", session_id="s3", incident_context=True)
    all_incident_cluster = TopicCluster(
        canonical_topic="oauth refresh",
        member_topic_keys=("oauth refresh",),
        candidates=(c1, c3),
    )
    assert score_topic_cluster(all_incident_cluster) == (
        BASE_COMPLEXITY_WEIGHT * 2 * NICHE_WEIGHT * INCIDENT_CONTEXT_MULTIPLIER
    )


def test_score_topic_cluster_recurrence_scales_linearly():
    single = _cluster("deadlocks", foundational=True, incident_context=False)
    c1 = _candidate(topic="deadlocks", session_id="s1", foundational=True)
    c2 = _candidate(topic="deadlocks", session_id="s2", foundational=True)
    c3 = _candidate(topic="deadlocks", session_id="s3", foundational=True)
    triple = TopicCluster(
        canonical_topic="deadlocks", member_topic_keys=("deadlocks",), candidates=(c1, c2, c3)
    )
    assert score_topic_cluster(triple) == score_topic_cluster(single) * 3


# ---------------------------------------------------------------------------
# run_pipeline -- validation
# ---------------------------------------------------------------------------


def test_run_pipeline_rejects_non_positive_minutes_before_touching_disk(tmp_path):
    # Deliberately a nonexistent directory -- if this raised NotADirectoryError
    # instead of the expected ValueError, that would mean total_minutes wasn't
    # actually checked first.
    bogus_dir = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="total_minutes must be > 0"):
        run_pipeline(
            bogus_dir,
            0,
            FakeGapDetectionClient({}),
            FakeClusteringClient({"clusters": []}),
            FakeScriptGenerationClient(),
        )
    with pytest.raises(ValueError, match="total_minutes must be > 0"):
        run_pipeline(
            bogus_dir,
            -5,
            FakeGapDetectionClient({}),
            FakeClusteringClient({"clusters": []}),
            FakeScriptGenerationClient(),
        )


# ---------------------------------------------------------------------------
# run_pipeline -- empty inputs are valid outcomes, not errors
# ---------------------------------------------------------------------------


def test_run_pipeline_empty_transcripts_dir(tmp_path):
    gap_client = FakeGapDetectionClient({})
    clustering_client = FakeClusteringClient({"clusters": []})
    script_client = FakeScriptGenerationClient()

    plan = run_pipeline(tmp_path, 60.0, gap_client, clustering_client, script_client)

    assert isinstance(plan, EpisodePlan)
    assert plan.transcripts_loaded == 0
    assert plan.gap_candidates_found == 0
    assert plan.scripts == ()
    assert plan.carried_over_topics == ()
    assert plan.total_minutes == 60.0
    assert plan.intro_outro_minutes == pytest.approx(6.0)
    # Nothing to teach -- the whole post-reserve budget is unallocated.
    assert plan.unallocated_minutes == pytest.approx(54.0)
    # clustering is never called when there is nothing to cluster.
    assert clustering_client.calls == []
    assert script_client.calls == []


def test_run_pipeline_no_candidate_survives_gating(tmp_path):
    _write_transcript(
        tmp_path / "01.json",
        "s1",
        "Rename a variable",
        [
            {"role": "user", "content": "rename usrId to userId", "timestamp": "2026-08-20T11:00:00Z"},
            {"role": "assistant", "content": "Done.", "timestamp": "2026-08-20T11:00:05Z"},
        ],
    )
    gap_client = FakeGapDetectionClient({"s1": _no_candidates_response()})
    clustering_client = FakeClusteringClient({"clusters": []})
    script_client = FakeScriptGenerationClient()

    plan = run_pipeline(tmp_path, 30.0, gap_client, clustering_client, script_client)

    assert plan.transcripts_loaded == 1
    assert plan.gap_candidates_found == 0
    assert plan.scripts == ()
    assert clustering_client.calls == []
    assert script_client.calls == []


def test_run_pipeline_skips_invalid_transcript_files(tmp_path, capsys):
    (tmp_path / "bad.json").write_text("{not valid json", encoding="utf-8")
    _write_transcript(
        tmp_path / "good.json",
        "s1",
        "Fix a race condition",
        [
            {"role": "user", "content": "fix the race condition", "timestamp": "2026-08-17T10:00:00Z"},
            {"role": "assistant", "content": "Added a lock.", "timestamp": "2026-08-17T10:00:05Z"},
        ],
    )
    gap_client = FakeGapDetectionClient({"s1": _no_candidates_response()})
    clustering_client = FakeClusteringClient({"clusters": []})
    script_client = FakeScriptGenerationClient()

    plan = run_pipeline(tmp_path, 30.0, gap_client, clustering_client, script_client)

    # The bad file is skipped (ingest.py's own contract), not fatal.
    assert plan.transcripts_loaded == 1
    assert gap_client.calls == ["s1"]


# ---------------------------------------------------------------------------
# run_pipeline -- end to end happy path
# ---------------------------------------------------------------------------


def test_run_pipeline_end_to_end_selects_and_scripts_topics(tmp_path):
    _write_transcript(
        tmp_path / "01_race_condition.json",
        "s101",
        "Fix race condition in payment worker",
        [
            {"role": "user", "content": "can you fix this race condition", "timestamp": "2026-08-17T10:00:00Z"},
            {"role": "assistant", "content": "Added a lock around the shared counter.", "timestamp": "2026-08-17T10:00:05Z"},
            {"role": "user", "content": "thanks, that works", "timestamp": "2026-08-17T10:01:00Z"},
        ],
    )
    _write_transcript(
        tmp_path / "02_deadlock.json",
        "s102",
        "Fix deadlock in job scheduler",
        [
            {"role": "user", "content": "my job scheduler is deadlocking, fix it", "timestamp": "2026-08-18T10:00:00Z"},
            {"role": "assistant", "content": "Reordered lock acquisition.", "timestamp": "2026-08-18T10:00:05Z"},
            {"role": "user", "content": "great, deployed", "timestamp": "2026-08-18T10:01:00Z"},
        ],
    )
    _write_transcript(
        tmp_path / "03_engaged_flaky_test.json",
        "s103",
        "Fix flaky test in CI",
        [
            {"role": "user", "content": "this test is flaky, fix it", "timestamp": "2026-08-19T09:00:00Z"},
            {"role": "assistant", "content": "Added a wait-for-condition.", "timestamp": "2026-08-19T09:00:10Z"},
            {"role": "user", "content": "why does that fix it -- walk me through it", "timestamp": "2026-08-19T09:02:00Z"},
            {"role": "assistant", "content": "Sure -- the sleep assumed...", "timestamp": "2026-08-19T09:02:15Z"},
        ],
    )

    gap_client = FakeGapDetectionClient(
        {
            # Flagged: no engagement, immediate accept.
            "s101": _gap_response("race conditions", foundational=True),
            "s102": _gap_response("deadlocks", foundational=True, incident_context=True),
            # NOT flagged: the user asked "why does that fix it", an engagement signal.
            "s103": _gap_response("flaky tests", no_engagement_signal=False),
        }
    )
    clustering_client = FakeClusteringClient(
        {
            "clusters": [
                {"canonical_topic": "race conditions", "member_topics": ["race conditions"]},
                {"canonical_topic": "deadlocks", "member_topics": ["deadlocks"]},
            ]
        }
    )
    script_client = FakeScriptGenerationClient()

    plan = run_pipeline(tmp_path, 30.0, gap_client, clustering_client, script_client)

    assert plan.transcripts_loaded == 3
    # All 3 classify() calls return a candidate; only 2 pass should_flag,
    # but detect_gaps returns every structurally-valid candidate.
    assert plan.gap_candidates_found == 3
    assert sorted(gap_client.calls) == ["s101", "s102", "s103"]

    # Only the 2 flagged topics were sent to clustering -- the engaged
    # flaky-test candidate never reaches it.
    assert clustering_client.calls == [["race conditions", "deadlocks"]]

    assert len(plan.scripts) == 2
    scripted_topics = {s.topic for s in plan.scripts}
    assert scripted_topics == {"race conditions", "deadlocks"}
    assert plan.carried_over_topics == ()

    # deadlocks is foundational + all-incident-context (severity =
    # 1 * 1 * 2.0 * 0.5 = 1.0); race conditions is foundational, no
    # incident context (severity = 1 * 1 * 2.0 * 1.0 = 2.0) -- higher
    # priority, so it's scripted first and gets top-up depth first.
    race_script = next(s for s in plan.scripts if s.topic == "race conditions")
    deadlock_script = next(s for s in plan.scripts if s.topic == "deadlocks")
    assert race_script.minutes >= deadlock_script.minutes

    # Script generation received the cluster's (paraphrased) candidate
    # descriptions, never raw transcript content.
    race_call = next(c for c in script_client.calls if c[0] == "race conditions")
    assert "delegated a fix involving race conditions" in race_call[1][0]

    total_accounted = (
        plan.intro_outro_minutes
        + sum(s.minutes for s in plan.scripts)
        + plan.unallocated_minutes
    )
    assert total_accounted == pytest.approx(plan.total_minutes)


def test_run_pipeline_reports_carried_over_topics_when_budget_is_tight(tmp_path):
    # 3 distinct flagged topics but a budget that (after the 10% reserve and
    # the 8-min floor) can only fit 2 at MAX_TOPICS_CAP-independent floor math.
    for i, (session_id, topic) in enumerate(
        [("s1", "topic a"), ("s2", "topic b"), ("s3", "topic c")]
    ):
        _write_transcript(
            tmp_path / f"{i}.json",
            session_id,
            f"Session about {topic}",
            [
                {"role": "user", "content": f"fix this {topic} issue", "timestamp": "2026-08-17T10:00:00Z"},
                {"role": "assistant", "content": "Fixed.", "timestamp": "2026-08-17T10:00:05Z"},
            ],
        )

    gap_client = FakeGapDetectionClient(
        {
            "s1": _gap_response("topic a", foundational=True),
            "s2": _gap_response("topic b", foundational=False),
            "s3": _gap_response("topic c", foundational=False),
        }
    )
    clustering_client = FakeClusteringClient(
        {
            "clusters": [
                {"canonical_topic": "topic a", "member_topics": ["topic a"]},
                {"canonical_topic": "topic b", "member_topics": ["topic b"]},
                {"canonical_topic": "topic c", "member_topics": ["topic c"]},
            ]
        }
    )
    script_client = FakeScriptGenerationClient()

    # 20 min budget: 10% reserve = 2 min, 18 min remaining -- fits exactly
    # 2 topics at the 8-min floor (16 min), not a 3rd.
    plan = run_pipeline(tmp_path, 20.0, gap_client, clustering_client, script_client)

    assert len(plan.scripts) == 2
    assert plan.carried_over_topics == ("topic c",)
    # "topic a" is foundational (severity 2.0) and ranks ahead of the two
    # equal-severity niche topics ("topic b"/"topic c", tie-broken
    # alphabetically by curriculum.fit_curriculum) -- "topic c" is the one
    # bumped.
    scripted_topics = {s.topic for s in plan.scripts}
    assert scripted_topics == {"topic a", "topic b"}


# ---------------------------------------------------------------------------
# run_pipeline -- integration test against the repo's real fixture files
# (BUILD-SCHEDULE.md day 8: "integration test against fixtures")
# ---------------------------------------------------------------------------

_PIPELINE_RUN_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pipeline_run"


def test_run_pipeline_against_real_fixture_transcripts():
    """Runs the full pipeline against the 6 real transcript files on disk.

    Unlike the synthetic-transcript tests above, this loads the actual
    `tests/fixtures/pipeline_run/*.json` files from ingest.py's real
    `load_transcripts_from_dir`, so it also serves as a check that those
    fixture files are valid transcripts. The 3 Claude-backed steps
    (gap-detection, clustering, script-generation) are still fakes, keyed
    by each fixture's real `session_id` -- BUILD-SCHEDULE.md's TTS/day-9
    notes are explicit that no live ANTHROPIC_API_KEY is available in this
    build sandbox, so this cannot (and does not claim to) exercise the
    real Claude-backed classification of these transcripts' content.
    """
    assert _PIPELINE_RUN_FIXTURES_DIR.is_dir(), (
        f"expected fixture directory at {_PIPELINE_RUN_FIXTURES_DIR}"
    )

    gap_client = FakeGapDetectionClient(
        {
            # 01_race_condition.json -- immediate accept, no follow-up question.
            "sess-101": _gap_response("race conditions", foundational=True),
            # 02_deadlock.json -- immediate accept under production pressure.
            "sess-102": _gap_response(
                "deadlocks", foundational=True, incident_context=True
            ),
            # 03_engaged_flaky_test.json -- user asked "why does that fix it",
            # an explicit engagement signal -- fails gate 2.
            "sess-103": _gap_response("flaky tests", no_engagement_signal=False),
            # 04_trivial_rename.json -- mechanical rename, no conceptual weight
            # -- fails gate 1.
            "sess-104": _gap_response(
                "variable renaming", non_trivial_delegation=False
            ),
            # 05_regex_lookahead.json -- immediate accept; regex is called out
            # by name in DESIGN.md's foundational-concepts list.
            "sess-105": _gap_response("regex lookahead", foundational=True),
            # 06_oauth_refresh.json -- immediate accept; auth/session flows are
            # also called out by name as foundational.
            "sess-107": _gap_response("oauth token refresh", foundational=True),
        }
    )
    clustering_client = FakeClusteringClient(
        {
            "clusters": [
                {"canonical_topic": "race conditions", "member_topics": ["race conditions"]},
                {"canonical_topic": "deadlocks", "member_topics": ["deadlocks"]},
                {"canonical_topic": "regex lookahead", "member_topics": ["regex lookahead"]},
                {
                    "canonical_topic": "oauth token refresh",
                    "member_topics": ["oauth token refresh"],
                },
            ]
        }
    )
    script_client = FakeScriptGenerationClient()

    plan = run_pipeline(
        _PIPELINE_RUN_FIXTURES_DIR, 60.0, gap_client, clustering_client, script_client
    )

    # All 6 fixture files load and get classified.
    assert plan.transcripts_loaded == 6
    assert plan.gap_candidates_found == 6
    assert sorted(gap_client.calls) == [
        "sess-101",
        "sess-102",
        "sess-103",
        "sess-104",
        "sess-105",
        "sess-107",
    ]

    # Only the 4 candidates that pass all 3 gates (not the engaged flaky-test
    # fix, not the trivial rename) reach clustering.
    assert clustering_client.calls == [
        ["race conditions", "deadlocks", "regex lookahead", "oauth token refresh"]
    ]

    # 60 min is generous enough (4 topics x 8 min floor = 32 <= 54 min
    # post-reserve) that all 4 flagged/clustered topics get scripted and
    # nothing is carried over.
    scripted_topics = {s.topic for s in plan.scripts}
    assert scripted_topics == {
        "race conditions",
        "deadlocks",
        "regex lookahead",
        "oauth token refresh",
    }
    assert plan.carried_over_topics == ()

    total_accounted = (
        plan.intro_outro_minutes
        + sum(s.minutes for s in plan.scripts)
        + plan.unallocated_minutes
    )
    assert total_accounted == pytest.approx(plan.total_minutes)
