import pytest

from weekly_ai_tutor.gap_detection import GapCandidate
from weekly_ai_tutor.scoring import (
    BASE_COMPLEXITY_WEIGHT,
    FOUNDATIONAL_WEIGHT,
    INCIDENT_CONTEXT_MULTIPLIER,
    NICHE_WEIGHT,
    ScoredGap,
    score_gap_candidates,
)


def _candidate(**overrides) -> GapCandidate:
    base = dict(
        session_id="s1",
        topic="git rebase",
        description="d",
        user_message_index=0,
        non_trivial_delegation=True,
        no_engagement_signal=True,
        immediate_accept=True,
        reasoning="r",
    )
    base.update(overrides)
    return GapCandidate(**base)


# ---------------------------------------------------------------------------
# filtering: only should_flag candidates get scored
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_list():
    assert score_gap_candidates([]) == []


def test_candidate_failing_a_gate_is_dropped():
    c = _candidate(immediate_accept=False)
    assert score_gap_candidates([c]) == []


def test_candidate_hard_excluded_is_dropped():
    c = _candidate(opted_out=True)
    assert score_gap_candidates([c]) == []


def test_all_dropped_returns_empty_list_not_error():
    candidates = [
        _candidate(session_id="s1", boilerplate=True),
        _candidate(session_id="s2", already_understood=True),
    ]
    assert score_gap_candidates(candidates) == []


# ---------------------------------------------------------------------------
# grouping by topic_key
# ---------------------------------------------------------------------------


def test_single_candidate_recurrence_count_is_one():
    c = _candidate()
    [scored] = score_gap_candidates([c])
    assert scored.recurrence_count == 1
    assert scored.topic_key == "git rebase"
    assert scored.display_topic == "git rebase"
    assert scored.candidates == (c,)


def test_topic_grouping_is_case_and_whitespace_insensitive():
    c1 = _candidate(session_id="s1", topic="Git Rebase")
    c2 = _candidate(session_id="s2", topic="  git rebase  ")
    [scored] = score_gap_candidates([c1, c2])
    assert scored.recurrence_count == 2
    assert set(scored.candidates) == {c1, c2}


def test_different_topics_produce_separate_groups():
    c1 = _candidate(session_id="s1", topic="git rebase")
    c2 = _candidate(session_id="s2", topic="sql joins")
    result = score_gap_candidates([c1, c2])
    assert len(result) == 2
    assert {s.topic_key for s in result} == {"git rebase", "sql joins"}


# ---------------------------------------------------------------------------
# severity formula
# ---------------------------------------------------------------------------


def test_severity_niche_single_occurrence():
    c = _candidate(foundational=False)
    [scored] = score_gap_candidates([c])
    assert scored.severity == pytest.approx(BASE_COMPLEXITY_WEIGHT * 1 * NICHE_WEIGHT)


def test_severity_foundational_weighted_higher_at_equal_recurrence():
    niche = _candidate(session_id="s1", topic="niche topic", foundational=False)
    foundational = _candidate(session_id="s2", topic="foundational topic", foundational=True)
    result = score_gap_candidates([niche, foundational])
    by_topic = {s.topic_key: s for s in result}
    assert by_topic["foundational topic"].severity > by_topic["niche topic"].severity
    assert by_topic["foundational topic"].severity == pytest.approx(
        BASE_COMPLEXITY_WEIGHT * 1 * FOUNDATIONAL_WEIGHT
    )


def test_severity_scales_with_recurrence_count():
    once = [_candidate(session_id="s1", topic="topic a")]
    three_times = [
        _candidate(session_id="s1", topic="topic b"),
        _candidate(session_id="s2", topic="topic b"),
        _candidate(session_id="s3", topic="topic b"),
    ]
    [scored_once] = score_gap_candidates(once)
    [scored_thrice] = score_gap_candidates(three_times)
    assert scored_thrice.recurrence_count == 3
    assert scored_thrice.severity == pytest.approx(scored_once.severity * 3)


def test_foundational_true_if_any_occurrence_tagged_foundational():
    c1 = _candidate(session_id="s1", topic="t", foundational=True)
    c2 = _candidate(session_id="s2", topic="t", foundational=False)
    [scored] = score_gap_candidates([c1, c2])
    assert scored.foundational is True


# ---------------------------------------------------------------------------
# incident_context down-weighting
# ---------------------------------------------------------------------------


def test_incident_context_down_weights_when_every_occurrence_is_incident():
    c = _candidate(incident_context=True)
    [scored] = score_gap_candidates([c])
    assert scored.all_incident_context is True
    assert scored.severity == pytest.approx(
        BASE_COMPLEXITY_WEIGHT * 1 * NICHE_WEIGHT * INCIDENT_CONTEXT_MULTIPLIER
    )


def test_incident_context_not_applied_when_only_some_occurrences_are_incident():
    c1 = _candidate(session_id="s1", topic="t", incident_context=True)
    c2 = _candidate(session_id="s2", topic="t", incident_context=False)
    [scored] = score_gap_candidates([c1, c2])
    assert scored.all_incident_context is False
    assert scored.severity == pytest.approx(BASE_COMPLEXITY_WEIGHT * 2 * NICHE_WEIGHT)


def test_incident_context_and_foundational_combine_multiplicatively():
    c = _candidate(foundational=True, incident_context=True)
    [scored] = score_gap_candidates([c])
    expected = BASE_COMPLEXITY_WEIGHT * 1 * FOUNDATIONAL_WEIGHT * INCIDENT_CONTEXT_MULTIPLIER
    assert scored.severity == pytest.approx(expected)


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------


def test_results_sorted_by_severity_descending():
    low = _candidate(session_id="s1", topic="low", foundational=False)
    high = [
        _candidate(session_id="s2", topic="high", foundational=True),
        _candidate(session_id="s3", topic="high", foundational=True),
    ]
    result = score_gap_candidates([low, *high])
    assert [s.topic_key for s in result] == ["high", "low"]


def test_ties_broken_alphabetically_by_topic_key():
    c1 = _candidate(session_id="s1", topic="zeta")
    c2 = _candidate(session_id="s2", topic="alpha")
    result = score_gap_candidates([c1, c2])
    assert [s.topic_key for s in result] == ["alpha", "zeta"]


# ---------------------------------------------------------------------------
# ScoredGap validation
# ---------------------------------------------------------------------------


def test_scored_gap_rejects_empty_topic_key():
    with pytest.raises(ValueError, match="non-empty topic_key"):
        ScoredGap(
            topic_key="",
            display_topic="t",
            candidates=(_candidate(),),
            recurrence_count=1,
            foundational=False,
            all_incident_context=False,
            severity=1.0,
        )


def test_scored_gap_rejects_empty_candidates():
    with pytest.raises(ValueError, match="at least one candidate"):
        ScoredGap(
            topic_key="t",
            display_topic="t",
            candidates=(),
            recurrence_count=1,
            foundational=False,
            all_incident_context=False,
            severity=1.0,
        )


def test_scored_gap_rejects_mismatched_recurrence_count():
    with pytest.raises(ValueError, match="must equal len\\(candidates\\)"):
        ScoredGap(
            topic_key="t",
            display_topic="t",
            candidates=(_candidate(),),
            recurrence_count=2,
            foundational=False,
            all_incident_context=False,
            severity=1.0,
        )


def test_scored_gap_rejects_negative_severity():
    with pytest.raises(ValueError, match="severity must be >= 0"):
        ScoredGap(
            topic_key="t",
            display_topic="t",
            candidates=(_candidate(),),
            recurrence_count=1,
            foundational=False,
            all_incident_context=False,
            severity=-1.0,
        )
