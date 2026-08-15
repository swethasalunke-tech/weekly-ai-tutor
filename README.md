# weekly-ai-tutor

Heavy AI users accumulate a quiet backlog of things they let the model handle without ever learning them. This tool looks at a week of real transcripts, finds the concepts that were outsourced rather than learned, and turns them into a short podcast lesson sized to whatever time you have.

**Status: early build, in progress.** This repo is being built incrementally, one real piece at a time — see `BUILD-SCHEDULE.md` for the day-by-day plan and `DESIGN.md` for the full architecture and gap-detection rubric. Nothing here is a stub described as finished work; each commit is what it says it is.

## What's implemented so far

- **Transcript schema and ingestion** (`weekly_ai_tutor/schema.py`, `weekly_ai_tutor/ingest.py`): a documented, tool-agnostic JSON transcript format, plus loaders that validate structure and report bad files without silently dropping them. See the docstring in `schema.py` for the exact expected shape.
- **Gap-detection gating** (`weekly_ai_tutor/gap_detection.py`): the 3 gating conditions from `DESIGN.md` (non-trivial delegation, no engagement signal, immediate accept), plus 5 additional signals (foundational, opted_out, boilerplate, already_understood, incident_context) needed for scoring — all evaluated via a single forced-tool-use Claude API call with a dependency-injected client. The 5 additional signals are optional in the response shape (default `False`), so this is a backward-compatible extension of day 2's contract. **Caveat:** no `ANTHROPIC_API_KEY` is available in the build sandbox, so `AnthropicGapDetectionClient`'s request/response plumbing is tested against a mocked SDK client, not a live call — see the module docstring.
- **Recurrence + foundational-weight scoring** (`weekly_ai_tutor/scoring.py`): takes a pool of `GapCandidate`s spanning one or more transcripts, filters to only those passing all 3 gates and not hard-excluded (opted_out/boilerplate/already_understood), groups by topic (exact-match, ahead of day 4's real clustering), and scores each group by `severity = recurrence_count x foundational_weight x incident_multiplier`. Incident-context candidates are down-weighted (0.5x when *every* occurrence was incident-driven), not excluded, per `DESIGN.md`'s "note it, don't rank it the same" language. Rework-cost weighting (`DESIGN.md` rubric item 6) is intentionally not implemented yet.

Everything else in the pipeline (topic clustering, curriculum fitting, script generation, TTS rendering, CLI) is not built yet — see `BUILD-SCHEDULE.md`.

## Setup

```bash
pip install -r requirements.txt
python3 -m pytest tests/
```

## Why Piper for text-to-speech

Piper is open-source and runs locally — no API key, no per-episode cost, no sending transcript-derived content to a third-party TTS service. The trade-off: voice models are downloaded from Hugging Face on first use, so a one-time internet-connected setup step is required before audio rendering works (see `weekly_ai_tutor/tts.py` once it lands, in the build schedule's TTS milestone).
