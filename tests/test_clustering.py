import pytest

from weekly_ai_tutor.clustering import (
    AnthropicClusteringClient,
    ClusteringResponseError,
    TopicCluster,
    build_clustering_prompt,
    cluster_gap_candidates,
    parse_topic_clusters,
)
from weekly_ai_tutor.gap_detection import GapCandidate


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


class FakeClusteringClient:
    """Test double for ClusteringClient -- returns a canned response."""

    def __init__(self, response: dict):
        self.response = response
        self.calls = []

    def cluster(self, topics):
        self.calls.append(list(topics))
        return self.response


# ---------------------------------------------------------------------------
# build_clustering_prompt
# ---------------------------------------------------------------------------


def test_prompt_lists_every_topic():
    prompt = build_clustering_prompt(["sql joins", "git rebase"])
    assert "- sql joins" in prompt
    assert "- git rebase" in prompt


def test_prompt_instructs_conservative_merging_and_no_invention():
    prompt = build_clustering_prompt(["sql joins"])
    assert "same underlying concept" in prompt
    assert "not invent" in prompt or "do not invent" in prompt.lower()


def test_prompt_deterministic_for_same_input():
    assert build_clustering_prompt(["a", "b"]) == build_clustering_prompt(["a", "b"])


# ---------------------------------------------------------------------------
# parse_topic_clusters
# ---------------------------------------------------------------------------


def test_parse_valid_single_cluster():
    raw = {
        "clusters": [
            {"canonical_topic": "SQL joins", "member_topics": ["sql joins", "joining tables in sql"]}
        ]
    }
    result = parse_topic_clusters(raw, ["sql joins", "joining tables in sql"])
    assert result == [
        {"canonical_topic": "SQL joins", "member_topics": ["sql joins", "joining tables in sql"]}
    ]


def test_parse_valid_multiple_clusters_including_singletons():
    raw = {
        "clusters": [
            {"canonical_topic": "SQL joins", "member_topics": ["sql joins", "joining tables in sql"]},
            {"canonical_topic": "git rebase", "member_topics": ["git rebase"]},
        ]
    }
    result = parse_topic_clusters(raw, ["sql joins", "joining tables in sql", "git rebase"])
    assert len(result) == 2


def test_parse_missing_clusters_key_raises():
    with pytest.raises(ClusteringResponseError, match="clusters"):
        parse_topic_clusters({}, ["a"])


def test_parse_clusters_not_a_list_raises():
    with pytest.raises(ClusteringResponseError, match="must be a list"):
        parse_topic_clusters({"clusters": "nope"}, ["a"])


def test_parse_cluster_not_an_object_raises():
    with pytest.raises(ClusteringResponseError, match="must be an object"):
        parse_topic_clusters({"clusters": ["nope"]}, ["a"])


def test_parse_cluster_missing_field_raises():
    raw = {"clusters": [{"canonical_topic": "t"}]}
    with pytest.raises(ClusteringResponseError, match="missing required field"):
        parse_topic_clusters(raw, ["a"])


def test_parse_cluster_empty_canonical_topic_raises():
    raw = {"clusters": [{"canonical_topic": "  ", "member_topics": ["a"]}]}
    with pytest.raises(ClusteringResponseError, match="canonical_topic must be a non-empty string"):
        parse_topic_clusters(raw, ["a"])


def test_parse_cluster_empty_member_topics_raises():
    raw = {"clusters": [{"canonical_topic": "t", "member_topics": []}]}
    with pytest.raises(ClusteringResponseError, match="member_topics must be a non-empty list"):
        parse_topic_clusters(raw, ["a"])


def test_parse_cluster_member_topics_not_a_list_raises():
    raw = {"clusters": [{"canonical_topic": "t", "member_topics": "a"}]}
    with pytest.raises(ClusteringResponseError, match="member_topics must be a non-empty list"):
        parse_topic_clusters(raw, ["a"])


def test_parse_cluster_non_string_member_raises():
    raw = {"clusters": [{"canonical_topic": "t", "member_topics": [123]}]}
    with pytest.raises(ClusteringResponseError, match="must be a non-empty string"):
        parse_topic_clusters(raw, ["a"])


def test_parse_hallucinated_member_topic_raises():
    raw = {"clusters": [{"canonical_topic": "t", "member_topics": ["not in input"]}]}
    with pytest.raises(ClusteringResponseError, match="was not in the input topic list"):
        parse_topic_clusters(raw, ["a"])


def test_parse_dropped_input_topic_raises():
    raw = {"clusters": [{"canonical_topic": "t", "member_topics": ["a"]}]}
    with pytest.raises(ClusteringResponseError, match="missing from every cluster"):
        parse_topic_clusters(raw, ["a", "b"])


def test_parse_duplicate_assignment_raises():
    raw = {
        "clusters": [
            {"canonical_topic": "t1", "member_topics": ["a"]},
            {"canonical_topic": "t2", "member_topics": ["a"]},
        ]
    }
    with pytest.raises(ClusteringResponseError, match="more than one cluster"):
        parse_topic_clusters(raw, ["a"])


def test_parse_empty_input_topics_and_empty_clusters_is_valid():
    assert parse_topic_clusters({"clusters": []}, []) == []


# ---------------------------------------------------------------------------
# TopicCluster
# ---------------------------------------------------------------------------


def test_topic_cluster_rejects_empty_canonical_topic():
    with pytest.raises(ValueError, match="non-empty canonical_topic"):
        TopicCluster(canonical_topic="  ", member_topic_keys=("a",), candidates=(_candidate(),))


def test_topic_cluster_rejects_empty_member_topic_keys():
    with pytest.raises(ValueError, match="at least one member_topic_key"):
        TopicCluster(canonical_topic="t", member_topic_keys=(), candidates=(_candidate(),))


def test_topic_cluster_rejects_empty_candidates():
    with pytest.raises(ValueError, match="at least one candidate"):
        TopicCluster(canonical_topic="t", member_topic_keys=("a",), candidates=())


def test_topic_cluster_rejects_stray_candidate_topic_key():
    c = _candidate(topic="git rebase")
    with pytest.raises(ValueError, match="not in member_topic_keys"):
        TopicCluster(canonical_topic="t", member_topic_keys=("sql joins",), candidates=(c,))


def test_topic_cluster_recurrence_count_is_len_candidates():
    c1 = _candidate(session_id="s1")
    c2 = _candidate(session_id="s2")
    cluster = TopicCluster(
        canonical_topic="git rebase", member_topic_keys=("git rebase",), candidates=(c1, c2)
    )
    assert cluster.recurrence_count == 2


# ---------------------------------------------------------------------------
# cluster_gap_candidates (fake client, end-to-end orchestration)
# ---------------------------------------------------------------------------


def test_cluster_gap_candidates_empty_input_returns_empty_list():
    fake = FakeClusteringClient({"clusters": []})
    assert cluster_gap_candidates([], fake) == []
    assert fake.calls == []  # never even called the client


def test_cluster_gap_candidates_all_filtered_out_returns_empty_list_without_calling_client():
    c = _candidate(opted_out=True)
    fake = FakeClusteringClient({"clusters": []})
    assert cluster_gap_candidates([c], fake) == []
    assert fake.calls == []


def test_cluster_gap_candidates_merges_differently_phrased_same_concept():
    c1 = _candidate(session_id="s1", topic="sql joins")
    c2 = _candidate(session_id="s2", topic="joining tables in sql")
    fake = FakeClusteringClient(
        {
            "clusters": [
                {
                    "canonical_topic": "SQL joins",
                    "member_topics": ["sql joins", "joining tables in sql"],
                }
            ]
        }
    )
    [cluster] = cluster_gap_candidates([c1, c2], fake)
    assert cluster.canonical_topic == "SQL joins"
    assert cluster.recurrence_count == 2
    assert set(cluster.candidates) == {c1, c2}
    assert fake.calls == [["sql joins", "joining tables in sql"]]


def test_cluster_gap_candidates_keeps_unrelated_topics_separate():
    c1 = _candidate(session_id="s1", topic="git rebase")
    c2 = _candidate(session_id="s2", topic="sql joins")
    fake = FakeClusteringClient(
        {
            "clusters": [
                {"canonical_topic": "git rebase", "member_topics": ["git rebase"]},
                {"canonical_topic": "sql joins", "member_topics": ["sql joins"]},
            ]
        }
    )
    result = cluster_gap_candidates([c1, c2], fake)
    assert len(result) == 2
    by_topic = {cl.canonical_topic: cl for cl in result}
    assert by_topic["git rebase"].candidates == (c1,)
    assert by_topic["sql joins"].candidates == (c2,)


def test_cluster_gap_candidates_filters_should_flag_before_clustering():
    real_gap = _candidate(session_id="s1", topic="git rebase")
    opted_out = _candidate(session_id="s2", topic="git rebase", opted_out=True)
    fake = FakeClusteringClient(
        {"clusters": [{"canonical_topic": "git rebase", "member_topics": ["git rebase"]}]}
    )
    [cluster] = cluster_gap_candidates([real_gap, opted_out], fake)
    assert cluster.candidates == (real_gap,)
    # only the flagged candidate's topic_key reaches the client
    assert fake.calls == [["git rebase"]]


def test_cluster_gap_candidates_deduplicates_topic_keys_before_calling_client():
    c1 = _candidate(session_id="s1", topic="git rebase")
    c2 = _candidate(session_id="s2", topic="Git Rebase")  # same topic_key, different casing
    fake = FakeClusteringClient(
        {"clusters": [{"canonical_topic": "git rebase", "member_topics": ["git rebase"]}]}
    )
    [cluster] = cluster_gap_candidates([c1, c2], fake)
    assert cluster.recurrence_count == 2
    assert fake.calls == [["git rebase"]]  # deduped, called once with one entry


def test_cluster_gap_candidates_propagates_malformed_response():
    c = _candidate()
    fake = FakeClusteringClient({"not_clusters": []})
    with pytest.raises(ClusteringResponseError):
        cluster_gap_candidates([c], fake)


def test_cluster_gap_candidates_propagates_dropped_topic_error():
    c1 = _candidate(session_id="s1", topic="git rebase")
    c2 = _candidate(session_id="s2", topic="sql joins")
    fake = FakeClusteringClient(
        {"clusters": [{"canonical_topic": "git rebase", "member_topics": ["git rebase"]}]}
    )
    with pytest.raises(ClusteringResponseError, match="missing from every cluster"):
        cluster_gap_candidates([c1, c2], fake)


# ---------------------------------------------------------------------------
# AnthropicClusteringClient -- request/response plumbing only, no live API
# ---------------------------------------------------------------------------


class _FakeToolUseBlock:
    def __init__(self, input_):
        self.type = "tool_use"
        self.name = "report_topic_clusters"
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
    expected_input = {"clusters": []}
    fake_response = _FakeResponse([_FakeToolUseBlock(expected_input)])
    fake_sdk_client = _FakeAnthropicSDKClient(fake_response)

    client = AnthropicClusteringClient(client=fake_sdk_client)
    result = client.cluster(["git rebase", "sql joins"])

    assert result == expected_input
    call_kwargs = fake_sdk_client.messages.create_calls[0]
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "report_topic_clusters"}
    assert call_kwargs["tools"][0]["name"] == "report_topic_clusters"
    assert "git rebase" in call_kwargs["messages"][0]["content"]


def test_anthropic_client_ignores_text_blocks_before_tool_use():
    expected_input = {"clusters": []}
    fake_response = _FakeResponse(
        [_FakeTextBlock("thinking out loud"), _FakeToolUseBlock(expected_input)]
    )
    fake_sdk_client = _FakeAnthropicSDKClient(fake_response)

    client = AnthropicClusteringClient(client=fake_sdk_client)
    result = client.cluster(["a"])

    assert result == expected_input


def test_anthropic_client_raises_when_no_tool_use_block_present():
    fake_response = _FakeResponse([_FakeTextBlock("I refuse to use the tool")])
    fake_sdk_client = _FakeAnthropicSDKClient(fake_response)

    client = AnthropicClusteringClient(client=fake_sdk_client)
    with pytest.raises(ClusteringResponseError, match="tool_use"):
        client.cluster(["a"])


def test_anthropic_client_uses_default_model_unless_overridden():
    fake_response = _FakeResponse([_FakeToolUseBlock({"clusters": []})])
    fake_sdk_client = _FakeAnthropicSDKClient(fake_response)

    client = AnthropicClusteringClient(client=fake_sdk_client)
    client.cluster(["a"])
    assert fake_sdk_client.messages.create_calls[0]["model"] == "claude-sonnet-5"

    fake_sdk_client2 = _FakeAnthropicSDKClient(fake_response)
    client2 = AnthropicClusteringClient(client=fake_sdk_client2, model="claude-opus-5")
    client2.cluster(["a"])
    assert fake_sdk_client2.messages.create_calls[0]["model"] == "claude-opus-5"
