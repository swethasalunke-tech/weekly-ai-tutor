import pytest

from weekly_ai_tutor.schema import Message, TranscriptValidationError, transcript_from_dict


def test_transcript_from_dict_valid():
    data = {
        "session_id": "abc",
        "title": "Test session",
        "started_at": "2026-08-10T14:32:00Z",
        "messages": [
            {"role": "user", "content": "hello", "timestamp": "2026-08-10T14:32:00Z"},
            {"role": "assistant", "content": "hi there"},
        ],
    }
    t = transcript_from_dict(data)
    assert t.session_id == "abc"
    assert t.title == "Test session"
    assert len(t.messages) == 2
    assert len(t.user_messages) == 1
    assert len(t.assistant_messages) == 1
    assert t.messages[1].timestamp is None  # missing timestamp -> None, not an error


def test_missing_session_id_raises():
    with pytest.raises(TranscriptValidationError, match="session_id"):
        transcript_from_dict({"title": "x", "messages": [{"role": "user", "content": "hi"}]})


def test_missing_messages_field_raises():
    with pytest.raises(TranscriptValidationError, match="missing required field"):
        transcript_from_dict({"session_id": "a", "title": "x"})


def test_empty_messages_list_raises():
    with pytest.raises(TranscriptValidationError, match="no messages"):
        transcript_from_dict({"session_id": "a", "title": "x", "messages": []})


def test_invalid_role_raises():
    with pytest.raises(TranscriptValidationError, match="role"):
        transcript_from_dict(
            {
                "session_id": "a",
                "title": "x",
                "messages": [{"role": "system", "content": "hi"}],
            }
        )


def test_empty_content_raises():
    with pytest.raises(TranscriptValidationError, match="content"):
        transcript_from_dict(
            {"session_id": "a", "title": "x", "messages": [{"role": "user", "content": "   "}]}
        )


def test_top_level_not_object_raises():
    with pytest.raises(TranscriptValidationError, match="must be an object"):
        transcript_from_dict(["not", "an", "object"])


def test_message_missing_role_raises():
    with pytest.raises(TranscriptValidationError, match="role/content"):
        transcript_from_dict(
            {"session_id": "a", "title": "x", "messages": [{"content": "hi"}]}
        )


def test_unparseable_timestamp_falls_back_to_none():
    data = {
        "session_id": "a",
        "title": "x",
        "messages": [{"role": "user", "content": "hi", "timestamp": "not-a-date"}],
    }
    t = transcript_from_dict(data)
    assert t.messages[0].timestamp is None


def test_message_direct_construction_validates_role():
    with pytest.raises(TranscriptValidationError):
        Message(role="bot", content="hi")
