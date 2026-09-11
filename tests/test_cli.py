import json

import pytest
from click.testing import CliRunner

import weekly_ai_tutor.cli as cli_module
from weekly_ai_tutor.cli import _slugify, cli, main, render_summary, write_episode_output
from weekly_ai_tutor.pipeline import EpisodePlan
from weekly_ai_tutor.script_generation import TopicScript

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _script(topic="git rebase", minutes=10.0):
    return TopicScript(
        topic=topic,
        minutes=minutes,
        hook="On a recent day you asked Claude to resolve a rebase conflict.",
        concept="A rebase replays commits onto a new base...",
        example="Imagine two branches that both edited the same line...",
        next_time="Next time, run `git status` before rebasing.",
    )


def _plan(**overrides):
    base = dict(
        total_minutes=30.0,
        intro_outro_minutes=3.0,
        scripts=(),
        carried_over_topics=(),
        unallocated_minutes=27.0,
        transcripts_loaded=0,
        gap_candidates_found=0,
    )
    base.update(overrides)
    return EpisodePlan(**base)


def _write_transcript(path, session_id, messages):
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "title": "t",
                "started_at": "2026-08-17T10:00:00Z",
                "messages": messages,
            }
        ),
        encoding="utf-8",
    )


class _FakeGapDetectionClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def classify(self, transcript):
        self.calls.append(transcript.session_id)
        return self.response


class _FakeClusteringClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def cluster(self, topics):
        self.calls.append(list(topics))
        return self.response


class _FakeScriptGenerationClient:
    def __init__(self):
        self.calls = []

    def generate(self, topic, source_descriptions, minutes):
        self.calls.append((topic, list(source_descriptions), minutes))
        return {
            "hook": f"On a recent day you asked about {topic}.",
            "concept": f"{topic}, from first principles.",
            "example": f"A minimal example of {topic}.",
            "next_time": f"Watch for {topic} next time.",
        }


class _RecordingFactory:
    """Stands in for an Anthropic*Client class -- records constructor kwargs, returns a fixed fake."""

    def __init__(self, fake):
        self._fake = fake
        self.construct_calls = []

    def __call__(self, **kwargs):
        self.construct_calls.append(kwargs)
        return self._fake


def _install_fake_clients(monkeypatch, gap_response, cluster_response):
    gap_factory = _RecordingFactory(_FakeGapDetectionClient(gap_response))
    clustering_factory = _RecordingFactory(_FakeClusteringClient(cluster_response))
    script_fake = _FakeScriptGenerationClient()
    script_factory = _RecordingFactory(script_fake)

    monkeypatch.setattr(cli_module, "AnthropicGapDetectionClient", gap_factory)
    monkeypatch.setattr(cli_module, "AnthropicClusteringClient", clustering_factory)
    monkeypatch.setattr(cli_module, "AnthropicScriptGenerationClient", script_factory)
    return gap_factory, clustering_factory, script_factory, script_fake


def _one_flagged_transcript(tmp_path, topic="race conditions"):
    _write_transcript(
        tmp_path / "01.json",
        "s1",
        [
            {"role": "user", "content": f"fix this {topic} bug", "timestamp": "2026-08-17T10:00:00Z"},
            {"role": "assistant", "content": "Fixed it.", "timestamp": "2026-08-17T10:00:05Z"},
            {"role": "user", "content": "thanks", "timestamp": "2026-08-17T10:01:00Z"},
        ],
    )
    gap_response = {
        "candidates": [
            {
                "topic": topic,
                "description": f"User delegated a fix involving {topic} and accepted it without engaging.",
                "user_message_index": 0,
                "non_trivial_delegation": True,
                "no_engagement_signal": True,
                "immediate_accept": True,
                "reasoning": "matches all 3 gates",
            }
        ]
    }
    cluster_response = {"clusters": [{"canonical_topic": topic, "member_topics": [topic]}]}
    return gap_response, cluster_response


# ---------------------------------------------------------------------------
# render_summary
# ---------------------------------------------------------------------------


def test_render_summary_no_topics_selected():
    text = render_summary(_plan(transcripts_loaded=2, gap_candidates_found=0))
    assert "Loaded 2 transcript(s), found 0 gap candidate(s) total." in text
    assert "Selected 0 topic(s)" in text
    assert "(none -- no gap survived gating/clustering, or the budget was too small)" in text


def test_render_summary_lists_each_selected_topic_with_minutes():
    plan = _plan(scripts=(_script("git rebase", 10.0), _script("SQL joins", 8.0)))
    text = render_summary(plan)
    assert "- git rebase (10.0 min)" in text
    assert "- SQL joins (8.0 min)" in text


def test_render_summary_reports_carried_over_topics():
    plan = _plan(carried_over_topics=("regex lookahead", "oauth refresh"))
    text = render_summary(plan)
    assert "Carried over to next week: regex lookahead, oauth refresh" in text


def test_render_summary_omits_carried_over_line_when_empty():
    text = render_summary(_plan(carried_over_topics=()))
    assert "Carried over" not in text


def test_render_summary_reports_unallocated_budget_above_threshold():
    text = render_summary(_plan(unallocated_minutes=5.0))
    assert "Unallocated budget: 5.0 min" in text


def test_render_summary_omits_unallocated_line_at_zero():
    text = render_summary(_plan(unallocated_minutes=0.0))
    assert "Unallocated budget" not in text


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topic,expected",
    [
        ("git rebase", "git-rebase"),
        ("SQL Joins!", "sql-joins"),
        ("  OAuth / token refresh  ", "oauth-token-refresh"),
        ("regex (lookahead)", "regex-lookahead"),
    ],
)
def test_slugify_normalizes_topic(topic, expected):
    assert _slugify(topic) == expected


def test_slugify_empty_result_falls_back_to_topic():
    assert _slugify("!!!") == "topic"


# ---------------------------------------------------------------------------
# write_episode_output
# ---------------------------------------------------------------------------


def test_write_episode_output_writes_one_file_per_script(tmp_path):
    out_dir = tmp_path / "out"
    plan = _plan(scripts=(_script("git rebase", 10.0), _script("SQL joins", 8.0)))
    write_episode_output(plan, out_dir)

    assert (out_dir / "git-rebase.txt").exists()
    assert (out_dir / "sql-joins.txt").exists()
    content = (out_dir / "git-rebase.txt").read_text(encoding="utf-8")
    assert content == plan.scripts[0].full_text
    assert "[Hook]" in content


def test_write_episode_output_creates_missing_output_dir(tmp_path):
    out_dir = tmp_path / "nested" / "out"
    plan = _plan(scripts=(_script("git rebase", 10.0),))
    write_episode_output(plan, out_dir)
    assert out_dir.is_dir()
    assert (out_dir / "git-rebase.txt").exists()


def test_write_episode_output_empty_scripts_still_creates_dir_but_writes_nothing(tmp_path):
    out_dir = tmp_path / "out"
    write_episode_output(_plan(scripts=()), out_dir)
    assert out_dir.is_dir()
    assert list(out_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# `cli run` -- argument parsing, wiring, and output rendering (Anthropic
# clients replaced by fakes; see module docstring)
# ---------------------------------------------------------------------------


def test_cli_run_happy_path_prints_summary(tmp_path, monkeypatch):
    gap_response, cluster_response = _one_flagged_transcript(tmp_path)
    _install_fake_clients(monkeypatch, gap_response, cluster_response)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["run", "--transcripts-dir", str(tmp_path), "--minutes", "20"]
    )

    assert result.exit_code == 0, result.output
    assert "Loaded 1 transcript(s), found 1 gap candidate(s) total." in result.output
    assert "race conditions" in result.output


def test_cli_run_missing_transcripts_dir_fails_before_running_pipeline(tmp_path, monkeypatch):
    gap_response, cluster_response = _one_flagged_transcript(tmp_path)
    gap_factory, _, _, _ = _install_fake_clients(monkeypatch, gap_response, cluster_response)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--transcripts-dir", str(tmp_path / "does-not-exist"), "--minutes", "20"],
    )

    assert result.exit_code != 0
    # click's own Path(exists=True) validation should fail before any client
    # is even constructed.
    assert gap_factory.construct_calls == []


def test_cli_run_missing_required_options_fails(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--transcripts-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "minutes" in result.output.lower()


def test_cli_run_writes_scripts_to_output_dir(tmp_path, monkeypatch):
    gap_response, cluster_response = _one_flagged_transcript(tmp_path, topic="deadlocks")
    _install_fake_clients(monkeypatch, gap_response, cluster_response)
    out_dir = tmp_path / "episode-out"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--transcripts-dir",
            str(tmp_path),
            "--minutes",
            "20",
            "--output-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "deadlocks.txt").exists()
    assert "Wrote 1 script file(s)" in result.output


def test_cli_run_without_output_dir_writes_no_files(tmp_path, monkeypatch):
    gap_response, cluster_response = _one_flagged_transcript(tmp_path)
    _install_fake_clients(monkeypatch, gap_response, cluster_response)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["run", "--transcripts-dir", str(tmp_path), "--minutes", "20"]
    )

    assert result.exit_code == 0, result.output
    assert "Wrote" not in result.output


def test_cli_run_passes_model_override_to_every_client(tmp_path, monkeypatch):
    gap_response, cluster_response = _one_flagged_transcript(tmp_path)
    gap_factory, clustering_factory, script_factory, _ = _install_fake_clients(
        monkeypatch, gap_response, cluster_response
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--transcripts-dir",
            str(tmp_path),
            "--minutes",
            "20",
            "--model",
            "claude-opus-5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert gap_factory.construct_calls == [{"model": "claude-opus-5"}]
    assert clustering_factory.construct_calls == [{"model": "claude-opus-5"}]
    assert script_factory.construct_calls == [{"model": "claude-opus-5"}]


def test_cli_run_no_model_override_constructs_clients_with_defaults(tmp_path, monkeypatch):
    gap_response, cluster_response = _one_flagged_transcript(tmp_path)
    gap_factory, clustering_factory, script_factory, _ = _install_fake_clients(
        monkeypatch, gap_response, cluster_response
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["run", "--transcripts-dir", str(tmp_path), "--minutes", "20"]
    )

    assert result.exit_code == 0, result.output
    assert gap_factory.construct_calls == [{}]
    assert clustering_factory.construct_calls == [{}]
    assert script_factory.construct_calls == [{}]


# ---------------------------------------------------------------------------
# main() -- exit-code / exception-translation wrapper around cli.main
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_success(tmp_path, monkeypatch):
    gap_response, cluster_response = _one_flagged_transcript(tmp_path)
    _install_fake_clients(monkeypatch, gap_response, cluster_response)

    exit_code = main(["run", "--transcripts-dir", str(tmp_path), "--minutes", "20"])
    assert exit_code == 0


def test_main_returns_nonzero_on_click_usage_error():
    # Missing all required options -- click raises a UsageError (a
    # ClickException subclass) which main() must catch and translate into
    # an exit code, not let propagate as SystemExit.
    exit_code = main(["run"])
    assert exit_code != 0


def test_main_handles_bare_invocation_as_a_usage_error():
    # No subcommand -- click raises UsageError("Missing command.") (a
    # ClickException, exit_code 2), the first except branch in main().
    exit_code = main([])
    assert exit_code == 2


def test_main_handles_help_via_click_exit_branch():
    # --help prints help text and raises click.exceptions.Exit(0) under
    # standalone_mode=False -- the second except branch in main().
    exit_code = main(["--help"])
    assert exit_code == 0


def test_main_returns_one_on_unexpected_exception(tmp_path, monkeypatch):
    gap_response, cluster_response = _one_flagged_transcript(tmp_path)
    _install_fake_clients(monkeypatch, gap_response, cluster_response)

    def _boom(*args, **kwargs):
        raise RuntimeError("something went wrong deep in the pipeline")

    monkeypatch.setattr(cli_module, "run_pipeline", _boom)

    exit_code = main(["run", "--transcripts-dir", str(tmp_path), "--minutes", "20"])
    assert exit_code == 1
