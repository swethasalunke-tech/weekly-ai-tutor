"""Text-to-speech rendering: DESIGN.md's "Render" step, BUILD-SCHEDULE.md day 7.

TTS decision (BUILD-SCHEDULE.md, settled 2026-08-13): Piper -- open-source,
local, no API key -- instead of a paid API (OpenAI TTS / ElevenLabs, as
DESIGN.md's v1 scope originally sketched). `pip install piper-tts` works in
this build sandbox, but downloading an actual voice model (the `.onnx` +
`.onnx.json` pair) from Hugging Face does not: this sandbox's network policy
blocks it. That means this module's real Piper-backed synthesis path is
written and unit-tested against a fake/injected `PiperVoice`-shaped object
that exercises the real chunk-to-`AudioSegment` conversion logic, but has
**not** been end-to-end verified against a real voice model producing real
audio anyone has listened to. See `PiperSpeechSynthesisClient`'s docstring
and the top-level README for the manual one-time setup step this needs on a
machine without that network restriction, before live audio output can be
claimed as verified.

## Pattern

Same shape as gap_detection.py / clustering.py / script_generation.py: a
`SpeechSynthesisClient` Protocol so format-handling and orchestration logic
can be tested without a real model, and a real `PiperSpeechSynthesisClient`
implementation written directly against the installed `piper-tts` package's
documented API (`piper.PiperVoice.load` + `PiperVoice.synthesize`, verified
against piper-tts 1.8.0's actual installed source in this sandbox -- see
`PiperSpeechSynthesisClient` docstring for the exact methods/attributes this
was checked against).

## What this module does NOT do

- Decide which topics get scripted, budget minutes, or produce script text
  -- that's curriculum.py (day 5) and script_generation.py (day 6). This
  module takes an already-built `TopicScript` (or raw text) and turns it
  into audio, nothing upstream of that.
- Add a spoken intro/outro or write show notes -- DESIGN.md's "Render" step
  also mentions concatenating a full episode with an intro/outro; that full
  episode-assembly + delivery step (mp3 encoding, show-notes file) is CLI
  wiring (day 8) or later, once there's an actual pipeline entry point to
  assemble multiple `TopicAudio` segments plus intro/outro copy into one
  output. This module's `concatenate_audio_segments` provides the
  general-purpose splice-with-silence-gap primitive that step will need, but
  does not itself decide episode structure.
- Encode to mp3. Piper's native output is 16-bit PCM WAV; this module works
  in that format throughout (`AudioSegment` is raw PCM + format metadata,
  `write_wav_file` writes a standard WAV container). DESIGN.md's mp3
  deliverable is a format-conversion step for a later day once ffmpeg/lame
  availability in this sandbox has actually been checked -- not assumed
  here.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from weekly_ai_tutor.script_generation import TopicScript, enforce_privacy

__all__ = [
    "AudioSegment",
    "AudioFormatError",
    "SynthesisError",
    "VoiceModelNotFoundError",
    "SpeechSynthesisClient",
    "PiperSpeechSynthesisClient",
    "synthesize_topic_script",
    "concatenate_audio_segments",
    "write_wav_file",
    "read_wav_file",
    "DEFAULT_GAP_SECONDS",
]

# Silence gap inserted between spliced segments (e.g. between topics in an
# episode). Not mandated by DESIGN.md, which doesn't specify a value -- this
# is a reasonable default (long enough to read as a segment boundary, short
# enough not to waste the listener's time) with a caller-overridable
# parameter on concatenate_audio_segments.
DEFAULT_GAP_SECONDS = 1.0


class AudioFormatError(ValueError):
    """Raised when an AudioSegment is malformed, or segments being combined don't match."""


class SynthesisError(RuntimeError):
    """Raised when a SpeechSynthesisClient fails to produce usable audio."""


class VoiceModelNotFoundError(FileNotFoundError):
    """Raised when a Piper voice model's .onnx/.onnx.json files aren't present on disk.

    This is expected to happen in this build sandbox (see module docstring)
    since the model download from Hugging Face is network-blocked here --
    the message below is written for a human doing the one-time manual
    download on an unrestricted machine, not as an internal assertion.
    """


@dataclass(frozen=True)
class AudioSegment:
    """Raw PCM audio plus the format metadata needed to interpret it.

    `pcm_bytes` is interleaved signed-integer PCM (the format Piper and the
    `wave` module both use) at `sample_width` bytes per sample, `channels`
    interleaved channels, `sample_rate` samples/sec per channel.
    """

    pcm_bytes: bytes
    sample_rate: int
    sample_width: int
    channels: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise AudioFormatError(f"AudioSegment sample_rate must be > 0, got {self.sample_rate}")
        if self.sample_width <= 0:
            raise AudioFormatError(f"AudioSegment sample_width must be > 0, got {self.sample_width}")
        if self.channels <= 0:
            raise AudioFormatError(f"AudioSegment channels must be > 0, got {self.channels}")
        frame_size = self.sample_width * self.channels
        if len(self.pcm_bytes) % frame_size != 0:
            raise AudioFormatError(
                f"AudioSegment pcm_bytes length ({len(self.pcm_bytes)}) is not a whole "
                f"number of frames at sample_width={self.sample_width}, channels={self.channels} "
                f"(frame_size={frame_size})"
            )

    @property
    def duration_seconds(self) -> float:
        frame_size = self.sample_width * self.channels
        num_frames = len(self.pcm_bytes) // frame_size
        return num_frames / self.sample_rate

    def format_matches(self, other: "AudioSegment") -> bool:
        return (
            self.sample_rate == other.sample_rate
            and self.sample_width == other.sample_width
            and self.channels == other.channels
        )

    def silence(self, seconds: float) -> "AudioSegment":
        """Build a silent AudioSegment of the given duration, in this segment's format."""
        if seconds < 0:
            raise AudioFormatError(f"silence() seconds must be >= 0, got {seconds}")
        frame_size = self.sample_width * self.channels
        num_frames = round(seconds * self.sample_rate)
        return AudioSegment(
            pcm_bytes=b"\x00" * (num_frames * frame_size),
            sample_rate=self.sample_rate,
            sample_width=self.sample_width,
            channels=self.channels,
        )


class SpeechSynthesisClient(Protocol):
    """Anything that can turn spoken text into an AudioSegment.

    Implementations should raise SynthesisError if synthesis fails or
    produces no audio, rather than returning an empty/malformed segment.
    """

    def synthesize(self, text: str) -> AudioSegment: ...


class PiperSpeechSynthesisClient:
    """Real SpeechSynthesisClient backed by a loaded Piper voice model.

    Written directly against piper-tts 1.8.0's actual installed API in this
    build sandbox (checked via `inspect` against the installed package, not
    assumed from memory/docs):

    - `piper.PiperVoice.load(model_path, config_path=None, ...)` loads a
      voice from an `.onnx` model file plus its `.onnx.json` config.
    - `PiperVoice.synthesize(text, syn_config=None)` returns an
      `Iterable[piper.voice.AudioChunk]`, each with `.sample_rate`,
      `.sample_width`, `.sample_channels`, and an `.audio_int16_bytes`
      property giving that chunk's audio as 16-bit PCM bytes.

    NOT exercised against a real `.onnx` voice model in this sandbox --
    downloading one from Hugging Face is blocked by this sandbox's network
    policy (see module docstring / BUILD-SCHEDULE.md's TTS decision note).
    Two things ARE tested here without a real model:

    1. `__init__`'s file-existence check, which raises
       `VoiceModelNotFoundError` before ever calling into `piper` if the
       model/config files aren't present on disk -- this is real,
       runnable, sandbox-local behavior.
    2. `synthesize()`'s chunk-to-AudioSegment conversion logic, exercised
       via the `voice=` constructor parameter, which accepts any object
       shaped like a loaded `PiperVoice` (i.e. exposing `.synthesize(text,
       syn_config=...)` returning an iterable of chunk-like objects) --
       tests inject a fake voice object built from real `piper.AudioChunk`
       *shape* (same attribute names) without loading real model weights.

    Live audio output from a real voice model has not been heard/verified
    in this sandbox. That verification needs to happen on a machine without
    the Hugging Face network restriction -- see README.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        voice: Any = None,
        synthesis_config: Any = None,
    ) -> None:
        if voice is not None:
            self._voice = voice
        else:
            if model_path is None:
                raise ValueError("PiperSpeechSynthesisClient requires model_path (or voice= for testing)")
            resolved_model = Path(model_path)
            resolved_config = Path(config_path) if config_path is not None else Path(f"{resolved_model}.json")
            missing = [p for p in (resolved_model, resolved_config) if not p.is_file()]
            if missing:
                raise VoiceModelNotFoundError(
                    "Piper voice model file(s) not found: "
                    + ", ".join(str(p) for p in missing)
                    + ". This build sandbox cannot download voice models from Hugging Face "
                    "(network policy blocks it) -- download a voice manually (e.g. from "
                    "https://huggingface.co/rhasspy/piper-voices) on a machine without that "
                    "restriction, place the .onnx and .onnx.json files together, and pass "
                    "that path here. See README for the one-time setup step."
                )
            import piper  # deferred import, see module docstring

            self._voice = piper.PiperVoice.load(str(resolved_model), config_path=str(resolved_config))
        self._synthesis_config = synthesis_config

    def synthesize(self, text: str) -> AudioSegment:
        if not text.strip():
            raise SynthesisError("cannot synthesize empty text")

        chunks = list(self._voice.synthesize(text, syn_config=self._synthesis_config))
        if not chunks:
            raise SynthesisError(
                f"Piper synthesize() returned no audio chunks for text of length {len(text)}"
            )

        first = chunks[0]
        pcm = b"".join(chunk.audio_int16_bytes for chunk in chunks)
        try:
            return AudioSegment(
                pcm_bytes=pcm,
                sample_rate=first.sample_rate,
                sample_width=first.sample_width,
                channels=first.sample_channels,
            )
        except AudioFormatError as exc:
            raise SynthesisError(f"Piper produced malformed audio: {exc}") from exc


def synthesize_topic_script(script: TopicScript, client: SpeechSynthesisClient) -> AudioSegment:
    """Render a TopicScript's full text to audio via `client`.

    Defense-in-depth privacy re-check: `TopicScript` is normally produced by
    `script_generation.generate_script`/`parse_topic_script`, which already
    run `enforce_privacy` on every section before returning a `TopicScript`.
    But `TopicScript` is a public dataclass nothing stops a caller from
    constructing directly (e.g. in a future CLI-wiring bug), so this
    function re-runs `enforce_privacy` on the assembled `full_text` before
    it ever reaches a synthesis client -- the same "no fabrication, no
    leak" defense-in-depth discipline script_generation.py applies at its
    own stage, applied again at this stage since audio is the last place a
    leak could surface before delivery.
    """
    enforce_privacy({"full_text": script.full_text})
    return client.synthesize(script.full_text)


def concatenate_audio_segments(
    segments: list[AudioSegment], gap_seconds: float = DEFAULT_GAP_SECONDS
) -> AudioSegment:
    """Splice multiple AudioSegments into one, inserting a silence gap between each.

    All segments must share the same sample_rate/sample_width/channels --
    raises AudioFormatError naming the mismatched segment's index otherwise,
    rather than silently producing corrupted/garbled audio by concatenating
    incompatible PCM streams. Raises AudioFormatError if `segments` is
    empty (nothing to concatenate) or `gap_seconds` is negative.
    """
    if not segments:
        raise AudioFormatError("concatenate_audio_segments requires at least one segment")
    if gap_seconds < 0:
        raise AudioFormatError(f"concatenate_audio_segments gap_seconds must be >= 0, got {gap_seconds}")

    first = segments[0]
    for i, seg in enumerate(segments[1:], start=1):
        if not first.format_matches(seg):
            raise AudioFormatError(
                f"segment {i} format (sample_rate={seg.sample_rate}, "
                f"sample_width={seg.sample_width}, channels={seg.channels}) does not match "
                f"segment 0's format (sample_rate={first.sample_rate}, "
                f"sample_width={first.sample_width}, channels={first.channels})"
            )

    if len(segments) == 1 or gap_seconds == 0:
        combined = b"".join(seg.pcm_bytes for seg in segments)
    else:
        gap = first.silence(gap_seconds)
        parts = []
        for i, seg in enumerate(segments):
            parts.append(seg.pcm_bytes)
            if i != len(segments) - 1:
                parts.append(gap.pcm_bytes)
        combined = b"".join(parts)

    return AudioSegment(
        pcm_bytes=combined,
        sample_rate=first.sample_rate,
        sample_width=first.sample_width,
        channels=first.channels,
    )


def write_wav_file(segment: AudioSegment, path: str | Path) -> None:
    """Write an AudioSegment to `path` as a standard WAV file."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setframerate(segment.sample_rate)
        wav_file.setsampwidth(segment.sample_width)
        wav_file.setnchannels(segment.channels)
        wav_file.writeframes(segment.pcm_bytes)


def read_wav_file(path: str | Path) -> AudioSegment:
    """Read a standard WAV file back into an AudioSegment (used by tests/round-trip checks)."""
    with wave.open(str(path), "rb") as wav_file:
        return AudioSegment(
            pcm_bytes=wav_file.readframes(wav_file.getnframes()),
            sample_rate=wav_file.getframerate(),
            sample_width=wav_file.getsampwidth(),
            channels=wav_file.getnchannels(),
        )


def _wav_bytes(segment: AudioSegment) -> bytes:
    """Serialize an AudioSegment to in-memory WAV bytes (helper for tests)."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setframerate(segment.sample_rate)
        wav_file.setsampwidth(segment.sample_width)
        wav_file.setnchannels(segment.channels)
        wav_file.writeframes(segment.pcm_bytes)
    return buf.getvalue()
