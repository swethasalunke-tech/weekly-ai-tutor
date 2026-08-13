# weekly-ai-tutor — design doc (draft)

## Problem

Heavy AI users accumulate a quiet backlog of things they let the model handle without ever learning them. Nobody notices this happening week to week — there's no natural moment where it surfaces. This tool closes that loop: look at a week of real AI usage, find the concepts that were outsourced rather than learned, and turn them into a short podcast lesson sized to whatever time the user has.

## Scope for v1

- **Data source:** local Claude Code / Cowork session transcripts (via `list_sessions` + `read_transcript`). No manual export step.
- **Output:** audio only — single-narrator explainer scripts rendered to mp3 via a real TTS API (OpenAI TTS or ElevenLabs). No video, no avatar, in v1.
- **Cadence:** run weekly, user sets a time budget per run (e.g. 20 / 60 / 120 minutes).

## Pipeline

1. **Ingest** — pull all sessions from the past 7 days, read full transcripts.
2. **Gap detection** — classify each transcript into candidate "learning gaps" (rubric below).
3. **Dedup + cluster** — group gap candidates by topic (e.g. "SQL joins," "git rebase," "regex lookahead," "OAuth flow").
4. **Prioritize + fit to time budget** — rank clusters, select as many as fit the stated budget.
5. **Script generation** — one explainer script per selected topic, grounded in the user's real (paraphrased) task, not generic textbook content.
6. **Render** — TTS per script, concatenate into one episode with a short intro/outro.
7. **Deliver** — mp3 + a written transcript/show-notes file per week.

## Gap-detection rubric

The core risk here is the same one the daily-repo-review process guards against elsewhere in this profile: inventing findings that aren't real. A "gap" must be a verifiable pattern in the transcript, not an inference about what the user probably doesn't know. Concretely, flag a moment as a gap candidate only when **all** of these hold:

1. **Non-trivial delegation.** The user asked Claude to do something with real conceptual weight — implement/fix/explain logic, not reformat, rename, or fetch. ("Fix this race condition" qualifies. "Convert this CSV to JSON" doesn't.)
2. **No engagement signal in response.** After Claude's output, the user shows no sign of trying to understand it: no "why did that work," no follow-up question, no manual edit to the generated content, no pushback or request for an alternative approach. If any of these are present, the user *was* engaging — not a gap, even if the topic is advanced.
3. **Immediate accept-and-move-on.** The output was used as-is (code run, text sent, command executed) with no diff review or hesitation visible in the transcript.

And weight/prioritize confirmed gaps by:

4. **Recurrence.** The same topic appearing 2+ times in the week with zero engagement either time is a much stronger signal than a one-off — it means the user is repeatedly hitting the same wall and routing around it instead of over it. Weight roughly by count.
5. **Foundational vs. niche.** Concepts that generalize (git operations, SQL fundamentals, regex, auth/session flows, core language features) outrank one-off library-specific trivia, even at equal recurrence — the lesson value compounds.
6. **Rework cost.** If a task required multiple back-and-forth corrections because the user under-specified it (a sign they didn't understand the domain well enough to ask precisely), that's itself a gap signal independent of the final "accept" — flag it even if the *last* turn looks like engagement.

Explicit **non-gates** — do not flag, even if the above conditions are technically met:

- The user stated intent not to learn this ("just do it," "don't explain, just fix it," time-pressure language). Respect it — log it as an opted-out delegation, not a lesson candidate.
- The task is genuinely boilerplate/mechanical (formatting, renaming, running a command the user has clearly used correctly many times before).
- The same topic was demonstrably understood earlier — e.g. the user explained it correctly themselves in an earlier session this week. Don't relearn what they already know.
- Production firefighting / incident context — appropriate to delegate fast under real deadline pressure; note it, don't rank it the same as leisure-time exploration.

Scoring sketch: `severity = complexity_weight × recurrence_count × foundational_weight`, computed only over candidates that pass all three gating conditions above.

## Curriculum fitting

Given a ranked list of gap clusters and a time budget T minutes:
- Reserve ~10% for intro/outro.
- Greedily select topics highest-severity-first until remaining time can't fit another topic at a minimum viable depth (~8 min floor per topic — below that, a topic gets merged with a related one or dropped this week and carried to next week's backlog).
- Cap at a max of 4–5 topics per episode regardless of budget — depth over breadth.

## Script template (per topic)

1. **Real hook** (30–60s): paraphrase the actual moment from this week — "on Tuesday you asked Claude to debug a race condition in your worker queue" — without exposing sensitive specifics (no real API keys, customer data, internal system names beyond what's needed for context).
2. **Concept, first principles** (3–5 min): explain the underlying idea independent of their specific code.
3. **Worked example** (3–5 min): a clean, minimal example illustrating the concept — can reuse the shape of their real problem without reusing sensitive content.
4. **Next time** (1 min): a concrete prompt/checklist for catching this themselves next time it comes up.

## Privacy constraint (hard rule)

The gap-detection and script-generation steps must never pass verbatim proprietary content (real code with internal identifiers, credentials, customer data, business logic specifics) into the final script. Extraction should paraphrase to the *concept* before it ever reaches the script-writing step. This mirrors the standing rule already in place for this GitHub profile: don't fabricate, and don't leak.

## Open questions for the build phase

- Exact TTS provider/voice, and whether cost per episode is acceptable at expected usage volume.
- Where "a week of AI usage" ends and begins for a user who works across multiple tools/machines (v1 scope is Claude Code/Cowork only, which undercounts anyone using Claude.ai chat or other assistants).
- Whether clusters should be shown to the user for approval before scripting (avoids wasted TTS cost on a gap the user disagrees with) — likely yes for v1, a quick approve/skip step per cluster before rendering.
- Storage/delivery: local files vs. a lightweight RSS feed so episodes show up in a podcast app automatically.
