import pytest

from weekly_ai_tutor.curriculum import (
    INTRO_OUTRO_RESERVE_FRACTION,
    MAX_TOPIC_MINUTES,
    MAX_TOPICS_CAP,
    MIN_TOPIC_MINUTES,
    CurriculumCandidate,
    CurriculumPlan,
    CurriculumSlot,
    fit_curriculum,
)


def _candidate(label="topic", priority=1.0, recurrence_count=1) -> CurriculumCandidate:
    return CurriculumCandidate(label=label, priority=priority, recurrence_count=recurrence_count)


# ---------------------------------------------------------------------------
# CurriculumCandidate validation
# ---------------------------------------------------------------------------


def test_candidate_rejects_empty_label():
    with pytest.raises(ValueError, match="non-empty label"):
        CurriculumCandidate(label="  ", priority=1.0)


def test_candidate_rejects_negative_priority():
    with pytest.raises(ValueError, match="priority must be >= 0"):
        CurriculumCandidate(label="t", priority=-1.0)


def test_candidate_rejects_recurrence_count_below_one():
    with pytest.raises(ValueError, match="recurrence_count must be >= 1"):
        CurriculumCandidate(label="t", priority=1.0, recurrence_count=0)


def test_candidate_accepts_zero_priority():
    # zero is a valid (if unlikely) priority -- only negative is rejected
    CurriculumCandidate(label="t", priority=0.0)


# ---------------------------------------------------------------------------
# CurriculumSlot validation
# ---------------------------------------------------------------------------


def test_slot_rejects_minutes_below_floor():
    with pytest.raises(ValueError, match="must be within"):
        CurriculumSlot(candidate=_candidate(), minutes=MIN_TOPIC_MINUTES - 1)


def test_slot_rejects_minutes_above_ceiling():
    with pytest.raises(ValueError, match="must be within"):
        CurriculumSlot(candidate=_candidate(), minutes=MAX_TOPIC_MINUTES + 1)


def test_slot_accepts_floor_and_ceiling_bounds():
    CurriculumSlot(candidate=_candidate(), minutes=MIN_TOPIC_MINUTES)
    CurriculumSlot(candidate=_candidate(), minutes=MAX_TOPIC_MINUTES)


# ---------------------------------------------------------------------------
# CurriculumPlan validation
# ---------------------------------------------------------------------------


def test_plan_rejects_non_positive_total_minutes():
    with pytest.raises(ValueError, match="total_minutes must be > 0"):
        CurriculumPlan(
            total_minutes=0,
            intro_outro_minutes=0,
            topics=(),
            carried_over=(),
            unallocated_minutes=0,
        )


def test_plan_rejects_mismatched_minute_accounting():
    with pytest.raises(ValueError, match="don't reconcile"):
        CurriculumPlan(
            total_minutes=100,
            intro_outro_minutes=10,
            topics=(),
            carried_over=(),
            unallocated_minutes=50,  # should be 90
        )


def test_plan_rejects_exceeding_max_topics_cap():
    topics = tuple(
        CurriculumSlot(candidate=_candidate(label=f"t{i}"), minutes=MIN_TOPIC_MINUTES)
        for i in range(MAX_TOPICS_CAP + 1)
    )
    total = sum(t.minutes for t in topics)
    with pytest.raises(ValueError, match="exceeds MAX_TOPICS_CAP"):
        CurriculumPlan(
            total_minutes=total,
            intro_outro_minutes=0,
            topics=topics,
            carried_over=(),
            unallocated_minutes=0,
        )


# ---------------------------------------------------------------------------
# fit_curriculum: input validation
# ---------------------------------------------------------------------------


def test_fit_rejects_non_positive_total_minutes():
    with pytest.raises(ValueError, match="total_minutes must be > 0"):
        fit_curriculum([_candidate()], 0)


def test_fit_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="duplicate CurriculumCandidate label"):
        fit_curriculum([_candidate(label="dup"), _candidate(label="dup")], 60)


# ---------------------------------------------------------------------------
# fit_curriculum: empty input
# ---------------------------------------------------------------------------


def test_fit_empty_candidates_returns_no_topics():
    plan = fit_curriculum([], 60)
    assert plan.topics == ()
    assert plan.carried_over == ()
    assert plan.intro_outro_minutes == pytest.approx(6.0)
    assert plan.unallocated_minutes == pytest.approx(54.0)


# ---------------------------------------------------------------------------
# fit_curriculum: intro/outro reserve
# ---------------------------------------------------------------------------


def test_fit_reserves_ten_percent_for_intro_outro():
    plan = fit_curriculum([], 100)
    assert plan.intro_outro_minutes == pytest.approx(100 * INTRO_OUTRO_RESERVE_FRACTION)


# ---------------------------------------------------------------------------
# fit_curriculum: floor selection and carry-over
# ---------------------------------------------------------------------------


def test_fit_selects_single_topic_at_floor_when_budget_is_tight():
    # 20 min budget: 2 min reserve, 18 remaining -- fits one topic at the
    # 8-min floor with leftover, not a second (would need 16 min for two).
    plan = fit_curriculum([_candidate(label="a", priority=1.0)], 20)
    assert plan.topic_count == 1
    assert plan.topics[0].candidate.label == "a"
    assert plan.topics[0].minutes >= MIN_TOPIC_MINUTES


def test_fit_carries_over_when_budget_cannot_fit_even_one_topic():
    # 5 min budget: 0.5 min reserve, 4.5 remaining -- can't afford an 8-min floor.
    plan = fit_curriculum([_candidate(label="a")], 5)
    assert plan.topics == ()
    assert [c.label for c in plan.carried_over] == ["a"]


def test_fit_selects_highest_priority_first_when_not_all_fit():
    # 25 min budget: 2.5 reserve, 22.5 remaining -- fits 2 at floor (16 min)
    # but not a 3rd (would need 24 min).
    candidates = [
        _candidate(label="low", priority=1.0),
        _candidate(label="high", priority=3.0),
        _candidate(label="mid", priority=2.0),
    ]
    plan = fit_curriculum(candidates, 25)
    assert [t.candidate.label for t in plan.topics] == ["high", "mid"]
    assert [c.label for c in plan.carried_over] == ["low"]


def test_fit_ties_broken_alphabetically_by_label():
    candidates = [
        _candidate(label="zeta", priority=1.0),
        _candidate(label="alpha", priority=1.0),
    ]
    plan = fit_curriculum(candidates, 100)
    assert [t.candidate.label for t in plan.topics] == ["alpha", "zeta"]


# ---------------------------------------------------------------------------
# fit_curriculum: MAX_TOPICS_CAP enforcement regardless of budget
# ---------------------------------------------------------------------------


def test_fit_caps_at_max_topics_even_with_huge_budget():
    candidates = [_candidate(label=f"t{i}", priority=float(i)) for i in range(10)]
    plan = fit_curriculum(candidates, 1000)
    assert plan.topic_count == MAX_TOPICS_CAP
    assert len(plan.carried_over) == 10 - MAX_TOPICS_CAP
    # highest-priority 5 are the ones selected (t9..t5), carried is t4..t0
    assert {t.candidate.label for t in plan.topics} == {"t9", "t8", "t7", "t6", "t5"}


# ---------------------------------------------------------------------------
# fit_curriculum: leftover distribution up to MAX_TOPIC_MINUTES
# ---------------------------------------------------------------------------


def test_fit_distributes_leftover_to_highest_priority_topics_first():
    # 100 min budget: 10 reserve, 90 remaining. One candidate: floor is 8,
    # leftover after floor is 82, capped at MAX_TOPIC_MINUTES (12), so this
    # topic gets 12 and the rest (78) is unallocated.
    plan = fit_curriculum([_candidate(label="solo")], 100)
    assert plan.topic_count == 1
    assert plan.topics[0].minutes == pytest.approx(MAX_TOPIC_MINUTES)
    assert plan.unallocated_minutes == pytest.approx(90 - MAX_TOPIC_MINUTES)


def test_fit_leftover_fills_highest_priority_before_lower():
    # 5 candidates cap fills 5 slots at floor (40 min), reserve for 200
    # budget is 20, remaining 180, leftover after floor = 140. Each of 5
    # topics can take up to (12-8)=4 more minutes = 20 min max total top-up,
    # so all 5 hit MAX_TOPIC_MINUTES and there's still leftover unallocated.
    candidates = [_candidate(label=f"t{i}", priority=float(i)) for i in range(5)]
    plan = fit_curriculum(candidates, 200)
    assert plan.topic_count == 5
    assert all(t.minutes == pytest.approx(MAX_TOPIC_MINUTES) for t in plan.topics)
    expected_unallocated = 200 * (1 - INTRO_OUTRO_RESERVE_FRACTION) - 5 * MAX_TOPIC_MINUTES
    assert plan.unallocated_minutes == pytest.approx(expected_unallocated)


def test_fit_leftover_partially_tops_up_when_not_enough_for_full_max():
    # 2 candidates, reserve leaves exactly 20 min remaining: floor takes 16,
    # leftover 4 -- not enough to max both out (would need 8 total), so
    # the higher-priority one gets the full top-up first.
    candidates = [
        _candidate(label="high", priority=2.0),
        _candidate(label="low", priority=1.0),
    ]
    # total_minutes chosen so remaining after reserve is exactly 20:
    # total * 0.9 = 20 -> total = 22.222...
    total = 20 / (1 - INTRO_OUTRO_RESERVE_FRACTION)
    plan = fit_curriculum(candidates, total)
    by_label = {t.candidate.label: t.minutes for t in plan.topics}
    assert by_label["high"] == pytest.approx(MIN_TOPIC_MINUTES + 4)
    assert by_label["low"] == pytest.approx(MIN_TOPIC_MINUTES)
    assert plan.unallocated_minutes == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# fit_curriculum: minute accounting always reconciles (via CurriculumPlan's
# own __post_init__ check -- if fit_curriculum's arithmetic were wrong,
# constructing the returned CurriculumPlan would itself raise)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("total_minutes", [5, 20, 22.5, 30, 45, 60, 90, 100, 200, 500])
def test_fit_accounting_always_reconciles(total_minutes):
    candidates = [_candidate(label=f"t{i}", priority=float(i)) for i in range(7)]
    plan = fit_curriculum(candidates, total_minutes)  # would raise if unbalanced
    assert plan.total_minutes == total_minutes
