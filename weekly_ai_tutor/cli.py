"""Command-line entry point: `weekly-ai-tutor run --minutes N`.

BUILD-SCHEDULE.md day 8: "CLI wiring -- `weekly-ai-tutor run --minutes N`
ties ingestion through script generation into one command." This module is
argument parsing and I/O only -- all the actual pipeline logic (which
clients get called in what order, how clusters get scored and fit to a time
budget) lives in `pipeline.py`, so it can be fully unit-tested without going
through a CLI invocation at all. This module's job is: parse arguments,
construct the real Anthropic-backed clients, call `pipeline.run_pipeline`,
and render the result to stdout / disk.

## Packaging note

There is no `setup.py`/`pyproject.toml` in this repo yet, so `weekly-ai-tutor`
is not installable as a standalone console script -- that packaging work is
not part of day 8's scope and is not claimed as done here. Today, this CLI is
invoked as `python3 -m weekly_ai_tutor.cli run --transcripts-dir DIR --minutes
N` (or `python3 -m weekly_ai_tutor run ...` via `weekly_ai_tutor/__main__.py`).
Adding a real `weekly-ai-tutor` console-script entry point is left for a later
day's packaging pass.

## Live-API caveat

`run_command` constructs `AnthropicGapDetectionClient` /
`AnthropicClusteringClient` / `AnthropicScriptGenerationClient` -- the same
classes flagged as NOT exercised against a live API in this build sandbox
(no `ANTHROPIC_API_KEY` available here; see each module's docstring). This
command has therefore not been run end-to-end against real transcripts in
this sandbox. What IS tested here (see tests/test_cli.py) is the argument
parsing, output rendering, and file-writing logic, with the three
Anthropic-backed clients replaced by fakes -- the same dependency-injection
seam `pipeline.run_pipeline` itself is tested through.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import click

from .clustering import AnthropicClusteringClient
from .gap_detection import AnthropicGapDetectionClient
from .pipeline import EpisodePlan, run_pipeline
from .script_generation import AnthropicScriptGenerationClient

__all__ = ["cli", "main", "render_summary", "write_episode_output"]


@click.group()
def cli() -> None:
    """weekly-ai-tutor: turn a week of AI usage into a personalized lesson podcast."""


@cli.command("run")
@click.option(
    "--transcripts-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory of transcript JSON files (see schema.py for the expected shape).",
)
@click.option(
    "--minutes",
    "total_minutes",
    required=True,
    type=float,
    help="Total time budget in minutes for this week's episode.",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to write one .txt script per selected topic. If omitted, only the stdout summary is produced.",
)
@click.option(
    "--model",
    default=None,
    help="Override the Claude model used for gap-detection/clustering/script-generation calls.",
)
def run_command(
    transcripts_dir: Path, total_minutes: float, output_dir: Path | None, model: str | None
) -> None:
    """Run the pipeline end to end: ingest, gap-detect, cluster, fit to budget, script."""
    client_kwargs = {"model": model} if model else {}
    gap_client = AnthropicGapDetectionClient(**client_kwargs)
    clustering_client = AnthropicClusteringClient(**client_kwargs)
    script_client = AnthropicScriptGenerationClient(**client_kwargs)

    plan = run_pipeline(
        transcripts_dir=transcripts_dir,
        total_minutes=total_minutes,
        gap_client=gap_client,
        clustering_client=clustering_client,
        script_client=script_client,
    )

    click.echo(render_summary(plan))

    if output_dir is not None:
        write_episode_output(plan, output_dir)
        click.echo(f"\nWrote {len(plan.scripts)} script file(s) to {output_dir}")


def render_summary(plan: EpisodePlan) -> str:
    """Human-readable summary of an EpisodePlan, for stdout.

    Deterministic and side-effect-free (no click/IO dependency) so it is
    directly unit-testable without a CLI invocation.
    """
    lines = [
        f"Loaded {plan.transcripts_loaded} transcript(s), "
        f"found {plan.gap_candidates_found} gap candidate(s) total.",
        f"Budget: {plan.total_minutes:.1f} min "
        f"({plan.intro_outro_minutes:.1f} min intro/outro reserve).",
        f"Selected {len(plan.scripts)} topic(s) for this week's episode:",
    ]
    if not plan.scripts:
        lines.append("  (none -- no gap survived gating/clustering, or the budget was too small)")
    for script in plan.scripts:
        lines.append(f"  - {script.topic} ({script.minutes:.1f} min)")
    if plan.carried_over_topics:
        lines.append(f"Carried over to next week: {', '.join(plan.carried_over_topics)}")
    if plan.unallocated_minutes > 1e-6:
        lines.append(f"Unallocated budget: {plan.unallocated_minutes:.1f} min")
    return "\n".join(lines)


def write_episode_output(plan: EpisodePlan, output_dir: Path) -> None:
    """Write one `.txt` file per generated script to `output_dir`.

    File names are a simple slug of the topic (lowercased, non-alphanumeric
    runs collapsed to '-'), so re-running against the same topics overwrites
    rather than accumulating duplicates across weeks. Creates `output_dir`
    (and any missing parents) if it doesn't already exist. Writes nothing
    (but still creates the directory) if `plan.scripts` is empty.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for script in plan.scripts:
        path = output_dir / f"{_slugify(script.topic)}.txt"
        path.write_text(script.full_text, encoding="utf-8")


def _slugify(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug or "topic"


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m weekly_ai_tutor` / `python -m weekly_ai_tutor.cli`.

    Returns a process exit code rather than calling `sys.exit` itself, so it
    can be called directly from tests without raising `SystemExit`.
    """
    try:
        cli.main(args=argv, standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 -- defensive top-level guard, see module docstring
        click.echo(f"weekly-ai-tutor: error: {exc}", err=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
