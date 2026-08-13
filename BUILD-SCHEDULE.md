# weekly-ai-tutor — build schedule

Same discipline as the rest of this profile's repos: no commit describes work that doesn't exist. Each day below is scoped to something that can actually be written and tested that day. If a day's real work turns out smaller or larger than planned, the schedule gets adjusted rather than the commit message stretched to match a plan.

TTS decision (settled 2026-08-13): Piper (open-source, local, no API key) over a paid API. `pip install piper-tts` works in this build sandbox; downloading a voice model from Hugging Face does not (network policy here blocks it), so the audio-rendering milestone will be written and unit-tested with the synthesis call mocked, and flagged for a real end-to-end audio test on a machine without that restriction — not claimed as verified until that happens.

| Day | Scope | Real deliverable |
|---|---|---|
| 1 | Repo scaffold + ingestion | `weekly_ai_tutor/schema.py` (Transcript/Message data model), `ingest.py` (loads the documented generic transcript JSON schema), tests, README stub |
| 2 | Gap-detection rubric, part 1 | `gap_detection.py` implementing the 3 gating conditions (non-trivial delegation, no engagement signal, immediate accept) as a structured Claude API call against a transcript; tests against fixture transcripts with known expected outcomes |
| 3 | Gap-detection rubric, part 2 | Recurrence + foundational-weight scoring and the 4 non-gate exclusion rules; tests |
| 4 | Clustering | Group gap candidates across a week's transcripts into topic clusters; tests |
| 5 | Curriculum fitting | Greedy time-budget allocator (intro/outro reserve, 8-min floor, 4-5 topic cap); tests |
| 6 | Script generation | Per-topic script from the 4-part template (hook/concept/example/next-time), real Claude API call, privacy-paraphrase step enforced before any content reaches a script; tests with mocked API |
| 7 | TTS module | Piper integration, mocked-synthesis unit tests; explicit README note that live audio output needs a manual one-time voice-model download outside this sandbox |
| 8 | CLI wiring | `weekly-ai-tutor run --minutes N` ties ingestion through script generation into one command; integration test against fixtures |
| 9 | End-to-end fixture run + README | Run the full pipeline against fixture transcripts, capture real output, write the README's usage/example section from that actual run (not invented output) |
| 10+ | Hardening | Edge cases found during the Day 9 run, error handling, docs gaps |

Rotation note: like the other repos, if a day's real work is smaller than expected, the surplus doesn't get invented — the commit just reflects what's actually done, and the next day continues from there.
