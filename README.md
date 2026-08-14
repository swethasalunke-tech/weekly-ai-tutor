# weekly-ai-tutor

Heavy AI users accumulate a quiet backlog of things they let the model handle without ever learning them. This tool looks at a week of real transcripts, finds the concepts that were outsourced rather than learned, and turns them into a short podcast lesson sized to whatever time you have.

**Status: early build, in progress.** This repo is being built incrementally, one real piece at a time — see `BUILD-SCHEDULE.md` for the day-by-day plan and `DESIGN.md` for the full architecture and gap-detection rubric. Nothing here is a stub described as finished work; each commit is what it says it is.

## What's implemented so far

- **Transcript schema and ingestion** (`weekly_ai_tutor/schema.py`, `weekly_ai_tutor/ingest.py`): a documented, tool-agnostic JSON transcript format, plus loaders that validate structure and report bad files without silently dropping them. See the docstring in `schema.py` for the exact expected shape.
- **Gap-detection gating, part 1** (`weekly_ai_tutor/gap_detection.py`): the 3 gating conditions from `DESIGN.md` (non-trivial delegation, no engagement signal, immediate accept), evaluated via a forced-tool-use Claude API call with dependency-injected client. Recurrence/foundational-weight scoring and the 4 non-gate exclusion rules aren't implemented yet — see `BUILD-SCHEDULE.md` day 3. **Caveat:** no `ANTHROPIC_API_KEY` is available in the build sandbox, so `AnthropicGapDetectionClient`'s request/response plumbing is tested against a mocked SDK client, not a live call — see the module docstring.

Everything else in the pipeline (recurrence weighting, clustering, curriculum fitting, script generation, TTS rendering, CLI) is not built yet — see `BUILD-SCHEDULE.md`.

## Setup

```bash
pip install -r requirements.txt
python3 -m pytest tests/
```

## Why Piper for text-to-speech

Piper is open-source and runs locally — no API key, no per-episode cost, no sending transcript-derived content to a third-party TTS service. The trade-off: voice models are downloaded from Hugging Face on first use, so a one-time internet-connected setup step is required before audio rendering works (see `weekly_ai_tutor/tts.py` once it lands, in the build schedule's TTS milestone).
