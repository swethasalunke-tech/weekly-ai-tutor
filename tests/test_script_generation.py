import pytest

from weekly_ai_tutor.script_generation import (
    AnthropicScriptGenerationClient,
    PrivacyViolationError,
    ScriptGenerationResponseError,
    TopicScript,
    build_script_prompt,
    enforce_privacy,
    generate_script,
    parse_topic_script,
)

# ---------------------------------------------------------------------------
# build_script_prompt
# ---------------------------------------------------------------------------


def test_prompt_includes_topic_minutes_and_descriptions():
    prompt = build_script_prompt(
        "git rebase", ["User asked Claude to fix a rebase conflict and accepted it as-is."], 10.0
    )
    assert "Topic: git rebase" in prompt
    assert "10.0 minutes" in prompt
    assert "User asked Claude to fix a rebase conflict and accepted it as-is." in prompt


def test_prompt_mentions_all_four_template_parts():
    prompt = build_script_prompt("git rebase", ["d1"], 10.0)
    assert "hook" in prompt
    assert "concept" in prompt
    assert "example" in prompt
    assert "next_time" in prompt


def test_prompt_numbers_multiple_descriptions():
    prompt = build_script_prompt("git rebase", ["first thing", "second thing"], 10.0)
    assert "1. first thing" in prompt
    assert "2. second thing" in prompt


def test_prompt_instructs_paraphrase_only_no_invention():
    prompt = build_script_prompt("git rebase", ["d1"], 10.0)
    assert "already paraphrased" in prompt
    assert "do not invent" in prompt.lower() or "do not attempt to reconstruct" in prompt.lower()


def test_prompt_requires_non_empty_topic():
    with pytest.raises(ValueError, match="non-empty topic"):
        build_script_prompt("", ["d1"], 10.0)


def test_prompt_requires_at_least_one_description():
    with pytest.raises(ValueError, match="at least one source description"):
        build_script_prompt("git rebase", [], 10.0)


def test_prompt_requires_positive_minutes():
    with pytest.raises(ValueError, match="minutes must be > 0"):
        build_script_prompt("git rebase", ["d1"], 0)
    with pytest.raises(ValueError, match="minutes must be > 0"):
        build_script_prompt("git rebase", ["d1"], -5)


# ---------------------------------------------------------------------------
# enforce_privacy
# ---------------------------------------------------------------------------


def test_enforce_privacy_passes_clean_text():
    enforce_privacy({"hook": "On Tuesday the user asked for a rebase conflict fix."})  # no raise


@pytest.mark.parametrize(
    "text",
    [
        "here's the key: sk-abcdefghijklmnopqrstuvwxyz123456",
        "token ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "AKIAABCDEFGHIJKLMNOP",
        "whsec_test_abcdefghij1234567890",
        "-----BEGIN RSA PRIVATE KEY-----",
        "password: hunter22222",
        "api_key=abcdef123456",
    ],
)
def test_enforce_privacy_raises_on_each_secret_pattern(text):
    with pytest.raises(PrivacyViolationError):
        enforce_privacy({"hook": text})


def test_enforce_privacy_names_offending_section():
    with pytest.raises(PrivacyViolationError, match="concept"):
        enforce_privacy(
            {
                "hook": "clean text",
                "concept": "leaked: sk-abcdefghijklmnopqrstuvwxyz123456",
            }
        )


# ---------------------------------------------------------------------------
# TopicScript
# ---------------------------------------------------------------------------


def _valid_sections(**overrides):
    base = dict(
        topic="git rebase",
        minutes=10.0,
        hook="On Tuesday you asked Claude to resolve a rebase conflict.",
        concept="A rebase replays commits onto a new base...",
        example="Imagine two branches that both edited the same line...",
        next_time="Next time, run `git status` before rebasing and read the conflict markers.",
    )
    base.update(overrides)
    return base


def test_topic_script_valid_construction():
    s = TopicScript(**_valid_sections())
    assert s.topic == "git rebase"
    assert "[Hook]" in s.full_text
    assert "[Concept]" in s.full_text
    assert "[Worked example]" in s.full_text
    assert "[Next time]" in s.full_text


def test_topic_script_requires_non_empty_topic():
    with pytest.raises(ScriptGenerationResponseError, match="non-empty topic"):
        TopicScript(**_valid_sections(topic=""))


def test_topic_script_requires_positive_minutes():
    with pytest.raises(ScriptGenerationResponseError, match="minutes must be > 0"):
        TopicScript(**_valid_sections(minutes=0))


@pytest.mark.parametrize("field", ["hook", "concept", "example", "next_time"])
def test_topic_script_requires_non_empty_sections(field):
    with pytest.raises(ScriptGenerationResponseError, match=f"{field!r} must be a non-empty string"):
        TopicScript(**_valid_sections(**{field: "   "}))


# ---------------------------------------------------------------------------
# parse_topic_script
# ---------------------------------------------------------------------------


def _valid_raw(**overrides):
    base = dict(
        hook="On Tuesday you asked Claude to resolve a rebase conflict.",
        concept="A rebase replays commits onto a new base...",
        example="Imagine two branches that both edited the same line...",
        next_time="Next time, run `git status` before rebasing.",
    )
    base.update(overrides)
    return base


def test_parse_valid_response():
    s = parse_topic_script(_valid_raw(), "git rebase", 10.0)
    assert isinstance(s, TopicScript)
    assert s.topic == "git rebase"
    assert s.minutes == 10.0


def test_parse_not_a_dict_raises():
    with pytest.raises(ScriptGenerationResponseError, match="must be an object"):
        parse_topic_script("nope", "git rebase", 10.0)


def test_parse_missing_field_raises():
    raw = _valid_raw()
    del raw["concept"]
    with pytest.raises(ScriptGenerationResponseError, match="missing required field.*concept"):
        parse_topic_script(raw, "git rebase", 10.0)


def test_parse_non_string_field_raises():
    with pytest.raises(ScriptGenerationResponseError, match="'hook' must be a non-empty string"):
        parse_topic_script(_valid_raw(hook=123), "git rebase", 10.0)


def test_parse_empty_field_raises():
    with pytest.raises(ScriptGenerationResponseError, match="'hook' must be a non-empty string"):
        parse_topic_script(_valid_raw(hook="   "), "git rebase", 10.0)


def test_parse_raises_privacy_violation_before_constructing_script():
    raw = _valid_raw(example="use this key: sk-abcdefghijklmnopqrstuvwxyz123456")
    with pytest.raises(PrivacyViolationError, match="example"):
        parse_topic_script(raw, "git rebase", 10.0)


# ---------------------------------------------------------------------------
# generate_script (fake client, end-to-end orchestration)
# ---------------------------------------------------------------------------


class FakeScriptGenerationClient:
    """Test double for ScriptGenerationClient -- returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, topic, source_descriptions, minutes):
        self.calls.append((topic, source_descriptions, minutes))
        return self.response


def test_generate_script_happy_path():
    fake = FakeScriptGenerationClient(_valid_raw())
    result = generate_script("git rebase", ["User hit a rebase conflict."], 10.0, fake)
    assert isinstance(result, TopicScript)
    assert result.topic == "git rebase"
    assert result.minutes == 10.0
    assert fake.calls == [("git rebase", ["User hit a rebase conflict."], 10.0)]


def test_generate_script_propagates_malformed_response():
    fake = FakeScriptGenerationClient({"hook": "h"})  # missing 3 fields
    with pytest.raises(ScriptGenerationResponseError):
        generate_script("git rebase", ["d1"], 10.0, fake)


def test_generate_script_rejects_leaked_source_description_before_calling_client():
    fake = FakeScriptGenerationClient(_valid_raw())
    with pytest.raises(PrivacyViolationError):
        generate_script(
            "git rebase", ["here's the key: sk-abcdefghijklmnopqrstuvwxyz123456"], 10.0, fake
        )
    # Privacy is checked before the client is ever called.
    assert fake.calls == []


def test_generate_script_rejects_leaked_output_section():
    fake = FakeScriptGenerationClient(_valid_raw(concept="secret: sk-abcdefghijklmnopqrstuvwxyz123456"))
    with pytest.raises(PrivacyViolationError, match="concept"):
        generate_script("git rebase", ["clean description"], 10.0, fake)
    # Client WAS called -- the leak was in its response, not the input.
    assert len(fake.calls) == 1


# ---------------------------------------------------------------------------
# AnthropicScriptGenerationClient -- request/response plumbing only, no live API
# ---------------------------------------------------------------------------


class _FakeToolUseBlock:
    def __init__(self, input_):
        self.type = "tool_use"
        self.name = "report_topic_script"
        self.input = input_


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self._response


class _FakeAnthropicSDKClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def test_anthropic_client_extracts_tool_use_input():
    expected_input = _valid_raw()
    fake_response = _FakeResponse([_FakeToolUseBlock(expected_input)])
    fake_sdk_client = _FakeAnthropicSDKClient(fake_response)

    client = AnthropicScriptGenerationClient(client=fake_sdk_client)
    result = client.generate("git rebase", ["d1"], 10.0)

    assert result == expected_input
    call_kwargs = fake_sdk_client.messages.create_calls[0]
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "report_topic_script"}
    assert call_kwargs["tools"][0]["name"] == "report_topic_script"
    assert "git rebase" in call_kwargs["messages"][0]["content"]


def test_anthropic_client_ignores_text_blocks_before_tool_use():
    expected_input = _valid_raw()
    fake_response = _FakeResponse(
        [_FakeTextBlock("thinking out loud"), _FakeToolUseBlock(expected_input)]
    )
    fake_sdk_client = _FakeAnthropicSDKClient(fake_response)

    client = AnthropicScriptGenerationClient(client=fake_sdk_client)
    result = client.generate("git rebase", ["d1"], 10.0)

    assert result == expected_input


def test_anthropic_client_raises_when_no_tool_use_block_present():
    fake_response = _FakeResponse([_FakeTextBlock("I refuse to use the tool")])
    fake_sdk_client = _FakeAnthropicSDKClient(fake_response)

    client = AnthropicScriptGenerationClient(client=fake_sdk_client)
    with pytest.raises(ScriptGenerationResponseError, match="tool_use"):
        client.generate("git rebase", ["d1"], 10.0)


def test_anthropic_client_uses_default_model_unless_overridden():
    fake_response = _FakeResponse([_FakeToolUseBlock(_valid_raw())])
    fake_sdk_client = _FakeAnthropicSDKClient(fake_response)

    client = AnthropicScriptGenerationClient(client=fake_sdk_client)
    client.generate("git rebase", ["d1"], 10.0)
    assert fake_sdk_client.messages.create_calls[0]["model"] == "claude-sonnet-5"

    fake_sdk_client2 = _FakeAnthropicSDKClient(fake_response)
    client2 = AnthropicScriptGenerationClient(client=fake_sdk_client2, model="claude-opus-5")
    client2.generate("git rebase", ["d1"], 10.0)
    assert fake_sdk_client2.messages.create_calls[0]["model"] == "claude-opus-5"
