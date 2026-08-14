from pathlib import Path

import pytest

from weekly_ai_tutor.gap_detection import (
    AnthropicGapDetectionClient,
    GapCandidate,
    GapDetectionResponseError,
    build_gap_detection_prompt,
    detect_gaps,
    parse_gap_candidates,
)
from weekly_ai_tutor.ingest import load_transcript_file

FIXTURES = Path(__file__).parent / "fixtures"


def _clean_gap_transcript():
    # sess-001: race-condition fix, user says "thanks, that works" -- no
    # why-question, no edit, used as-is. Plausible "passes all 3 gates" case.
    return load_transcript_file(FIXTURES / "sample_transcript.json")


def _engaged_transcript():
    # sess-002: user explicitly asks "why does that fix it" -- engagement
    # signal present, should fail gate 2 regardless of gates 1/3.
    return load_transcript_file(FIXTURES / "gap_engaged_transcript.json")


def _trivial_transcript():
    # sess-003: mechanical rename -- should fail gate 1.
    return load_transcript_file(FIXTURES / "gap_trivial_transcript.json")


class FakeGapDetectionClient:
    """Test double for GapDetectionClient -- returns a canned response."""

    def __init__(self, response: dict):
        self.response = response
        self.calls = []

    def classify(self, transcript):
        self.calls.append(transcript)
        return self.response


# ---------------------------------------------------------------------------
# build_gap_detection_prompt
# ---------------------------------------------------------------------------


def test_prompt_includes_session_metadata_and_indexed_messages():
    t = _clean_gap_transcript()
    prompt = build_gap_detection_prompt(t)
    assert "sess-001" in prompt
    assert "Fix race condition in worker queue" in prompt
    assert "0: user: can you fix this race condition in my worker queue" in prompt
    assert "1: assistant: I added a lock around the queue pop to fix it." in prompt
    assert "2: user: thanks, that works" in prompt


def test_prompt_mentions_all_three_gates():
    prompt = build_gap_detection_prompt(_clean_gap_transcript())
    assert "non_trivial_delegation" in prompt
    assert "no_engagement_signal" in prompt
    assert "immediate_accept" in prompt


# ---------------------------------------------------------------------------
# GapCandidate
# ---------------------------------------------------------------------------


def test_gap_candidate_passes_gates_only_when_all_three_true():
    base = dict(
        session_id="s",
        topic="t",
        description="d",
        user_message_index=0,
        reasoning="r",
    )
    assert GapCandidate(
        **base, non_trivial_delegation=True, no_engagement_signal=True, immediate_accept=True
    ).passes_gates
    assert not GapCandidate(
        **base, non_trivial_delegation=True, no_engagement_signal=True, immediate_accept=False
    ).passes_gates
    assert not GapCandidate(
        **base, non_trivial_delegation=False, no_engagement_signal=True, immediate_accept=True
    ).passes_gates
    assert not GapCandidate(
        **base, non_trivial_delegation=True, no_engagement_signal=False, immediate_accept=True
    ).passes_gates


def test_gap_candidate_requires_session_id():
    with pytest.raises(GapDetectionResponseError, match="session_id"):
        GapCandidate(
            session_id="",
            topic="t",
            description="d",
            user_message_index=0,
            non_trivial_delegation=True,
            no_engagement_signal=True,
            immediate_accept=True,
            reasoning="r",
        )


def test_gap_candidate_requires_non_negative_index():
    with pytest.raises(GapDetectionResponseError, match="user_message_index"):
        GapCandidate(
            session_id="s",
            topic="t",
            description="d",
            user_message_index=-1,
            non_trivial_delegation=True,
            no_engagement_signal=True,
            immediate_accept=True,
            reasoning="r",
        )


# ---------------------------------------------------------------------------
# parse_gap_candidates
# ---------------------------------------------------------------------------


def test_parse_valid_response():
    t = _clean_gap_transcript()
    raw = {
        "candidates": [
            {
                "topic": "race condition debugging",
                "description": "User asked for a fix to a queue race condition and accepted it without discussion.",
                "user_message_index": 0,
                "non_trivial_delegation": True,
                "no_engagement_signal": True,
                "immediate_accept": True,
                "reasoning": "User said 'thanks, that works' with no follow-up question.",
            }
        ]
    }
    candidates = parse_gap_candidates(raw, t)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.session_id == "sess-001"
    assert c.topic == "race condition debugging"
    assert c.passes_gates is True


def test_parse_missing_candidates_key_raises():
    with pytest.raises(GapDetectionResponseError, match="candidates"):
        parse_gap_candidates({}, _clean_gap_transcript())


def test_parse_candidates_not_a_list_raises():
    with pytest.raises(GapDetectionResponseError, match="must be a list"):
        parse_gap_candidates({"candidates": "nope"}, _clean_gap_transcript())


def test_parse_candidate_not_an_object_raises():
    with pytest.raises(GapDetectionResponseError, match="must be an object"):
        parse_gap_candidates({"candidates": ["nope"]}, _clean_gap_transcript())


def test_parse_candidate_missing_field_raises():
    raw = {
        "candidates": [
            {
                "topic": "t",
                "description": "d",
                "user_message_index": 0,
                "non_trivial_delegation": True,
                "no_engagement_signal": True,
                # immediate_accept missing
                "reasoning": "r",
            }
        ]
    }
    with pytest.raises(GapDetectionResponseError, match="missing required field"):
        parse_gap_candidates(raw, _clean_gap_transcript())


def test_parse_candidate_non_bool_gate_raises():
    raw = {
        "candidates": [
            {
                "topic": "t",
                "description": "d",
                "user_message_index": 0,
                "non_trivial_delegation": "yes",  # should be bool
                "no_engagement_signal": True,
                "immediate_accept": True,
                "reasoning": "r",
            }
        ]
    }
    with pytest.raises(GapDetectionResponseError, match="non_trivial_delegation must be a boolean"):
        parse_gap_candidates(raw, _clean_gap_transcript())


def test_parse_candidate_empty_string_field_raises():
    raw = {
        "candidates": [
            {
                "topic": "  ",
                "description": "d",
                "user_message_index": 0,
                "non_trivial_delegation": True,
                "no_engagement_signal": True,
                "immediate_accept": True,
                "reasoning": "r",
            }
        ]
    }
    with pytest.raises(GapDetectionResponseError, match="topic must be a non-empty string"):
        parse_gap_candidates(raw, _clean_gap_transcript())


def test_parse_candidate_bool_index_rejected():
    # bool is a subclass of int in Python -- True/False must not slip
    # through the "is an integer" check.
    raw = {
        "candidates": [
            {
                "topic": "t",
                "description": "d",
                "user_message_index": True,
                "non_trivial_delegation": True,
                "no_engagement_signal": True,
                "immediate_accept": True,
                "reasoning": "r",
            }
        ]
    }
    with pytest.raises(GapDetectionResponseError, match="user_message_index must be an integer"):
        parse_gap_candidates(raw, _clean_gap_transcript())


def test_parse_candidate_index_out_of_range_raises():
    raw = {
        "candidates": [
            {
                "topic": "t",
                "description": "d",
                "user_message_index": 99,
                "non_trivial_delegation": True,
                "no_engagement_signal": True,
                "immediate_accept": True,
                "reasoning": "r",
            }
        ]
    }
    with pytest.raises(GapDetectionResponseError, match="out of range"):
        parse_gap_candidates(raw, _clean_gap_transcript())


def test_parse_candidate_index_pointing_at_assistant_message_raises():
    raw = {
        "candidates": [
            {
                "topic": "t",
                "description": "d",
                "user_message_index": 1,  # message 1 is the assistant reply
                "non_trivial_delegation": True,
                "no_engagement_signal": True,
                "immediate_accept": True,
                "reasoning": "r",
            }
        ]
    }
    with pytest.raises(GapDetectionResponseError, match="not a user message"):
        parse_gap_candidates(raw, _clean_gap_transcript())


# ---------------------------------------------------------------------------
# detect_gaps (fake client, end-to-end orchestration)
# ---------------------------------------------------------------------------


def test_detect_gaps_clean_transcript_passes_all_gates():
    t = _clean_gap_transcript()
    fake = FakeGapDetectionClient(
        {
            "candidates": [
                {
                    "topic": "race condition debugging",
                    "description": "User asked for a queue race-condition fix and accepted it without discussion.",
                    "user_message_index": 0,
                    "non_trivial_delegation": True,
                    "no_engagement_signal": True,
                    "immediate_accept": True,
                    "reasoning": "Conceptual fix, no follow-up question, immediate 'thanks, that works'.",
                }
            ]
        }
    )
    result = detect_gaps(t, fake)
    assert len(result) == 1
    assert result[0].passes_gates
    assert fake.calls == [t]


def test_detect_gaps_engaged_transcript_fails_gate_2():
    t = _engaged_transcript()
    fake = FakeGapDetectionClient(
        {
            "candidates": [
                {
                    "topic": "flaky CI test",
                    "description": "User asked for a flaky test fix, then asked why the fix worked.",
                    "user_message_index": 0,
                    "non_trivial_delegation": True,
                    "no_engagement_signal": False,  # user asked "why does that fix it"
                    "immediate_accept": True,
                    "reasoning": "User explicitly asked to be walked through the fix -- engagement signal present.",
                }
            ]
        }
    )
    result = detect_gaps(t, fake)
    assert len(result) == 1
    assert result[0].no_engagement_signal is False
    assert not result[0].passes_gates


def test_detect_gaps_trivial_transcript_fails_gate_1():
    t = _trivial_transcript()
    fake = FakeGapDetectionClient(
        {
            "candidates": [
                {
                    "topic": "variable rename",
                    "description": "User asked for a mechanical variable rename across a file.",
                    "user_message_index": 0,
                    "non_trivial_delegation": False,  # mechanical, not conceptual
                    "no_engagement_signal": True,
                    "immediate_accept": True,
                    "reasoning": "Pure rename, no conceptual weight, so gate 1 fails regardless of gates 2/3.",
                }
            ]
        }
    )
    result = detect_gaps(t, fake)
    assert len(result) == 1
    assert not result[0].passes_gates


def test_detect_gaps_mixed_candidates_filters_correctly():
    t = _clean_gap_transcript()
    fake = FakeGapDetectionClient(
        {
            "candidates": [
                {
                    "topic": "real gap",
                    "description": "d1",
                    "user_message_index": 0,
                    "non_trivial_delegation": True,
                    "no_engagement_signal": True,
                    "immediate_accept": True,
                    "reasoning": "r1",
                },
                {
                    "topic": "not a gap",
                    "description": "d2",
                    "user_message_index": 2,
                    "non_trivial_delegation": True,
                    "no_engagement_signal": False,
                    "immediate_accept": True,
                    "reasoning": "r2",
                },
            ]
        }
    )
    result = detect_gaps(t, fake)
    passing = [c for c in result if c.passes_gates]
    assert len(result) == 2
    assert len(passing) == 1
    assert passing[0].topic == "real gap"


def test_detect_gaps_no_candidates_returns_empty_list():
    t = _clean_gap_transcript()
    fake = FakeGapDetectionClient({"candidates": []})
    assert detect_gaps(t, fake) == []


def test_detect_gaps_propagates_malformed_response():
    t = _clean_gap_transcript()
    fake = FakeGapDetectionClient({"not_candidates": []})
    with pytest.raises(GapDetectionResponseError):
        detect_gaps(t, fake)


# ---------------------------------------------------------------------------
# AnthropicGapDetectionClient -- request/response plumbing only, no live API
# ---------------------------------------------------------------------------


class _FakeToolUseBlock:
    def __init__(self, input_):
        self.type = "tool_use"
        self.name = "report_gap_candidates"
        self.input = input_


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeMessages:
    def __init__(self, response, expected_kwargs_check=None):
        self._response = response
        self._expected_kwargs_check = expected_kwargs_check
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self._expected_kwargs_check:
            self._expected_kwargs_check(kwargs)
        return self._response


class _FakeAnthropicSDKClient:
    def __init__(self, response, expected_kwargs_check=None):
        self.messages = _FakeMessages(response, expected_kwargs_check)


def test_anthropic_client_extracts_tool_use_input():
    expected_input = {"candidates": []}
    fake_response = _FakeResponse([_FakeToolUseBlock(expected_input)])
    fake_sdk_client = _FakeAnthropicSDKClient(fake_response)

    client = AnthropicGapDetectionClient(client=fake_sdk_client)
    result = client.classify(_clean_gap_transcript())

    assert result == expected_input
    assert len(fake_sdk_client.messages.create_calls) == 1
    call_kwargs = fake_sdk_client.messages.create_calls[0]
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "report_gap_candidates"}
    assert call_kwargs["tools"][0]["name"] == "report_gap_candidates"
    assert "sess-001" in call_kwargs["messages"][0]["content"]


def test_anthropic_client_ignores_text_blocks_before_tool_use():
    expected_input = {"candidates": []}
    fake_response = _FakeResponse(
        [_FakeTextBlock("thinking out loud"), _FakeToolUseBlock(expected_input)]
    )
    fake_sdk_client = _FakeAnthropicSDKClient(fake_response)

    client = AnthropicGapDetectionClient(client=fake_sdk_client)
    result = client.classify(_clean_gap_transcript())

    assert result == expected_input


def test_anthropic_client_raises_when_no_tool_use_block_present():
    fake_response = _FakeResponse([_FakeTextBlock("I refuse to use the tool")])
    fake_sdk_client = _FakeAnthropicSDKClient(fake_response)

    client = AnthropicGapDetectionClient(client=fake_sdk_client)
    with pytest.raises(GapDetectionResponseError, match="tool_use"):
        client.classify(_clean_gap_transcript())


def test_anthropic_client_uses_default_model_unless_overridden():
    fake_response = _FakeResponse([_FakeToolUseBlock({"candidates": []})])
    fake_sdk_client = _FakeAnthropicSDKClient(fake_response)

    client = AnthropicGapDetectionClient(client=fake_sdk_client)
    client.classify(_clean_gap_transcript())
    assert fake_sdk_client.messages.create_calls[0]["model"] == "claude-sonnet-5"

    fake_sdk_client2 = _FakeAnthropicSDKClient(fake_response)
    client2 = AnthropicGapDetectionClient(client=fake_sdk_client2, model="claude-opus-5")
    client2.classify(_clean_gap_transcript())
    assert fake_sdk_client2.messages.create_calls[0]["model"] == "claude-opus-5"
