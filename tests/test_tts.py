import wave
from io import BytesIO

import pytest

from weekly_ai_tutor.script_generation import PrivacyViolationError, TopicScript
from weekly_ai_tutor.tts import (
    AudioFormatError,
    AudioSegment,
    PiperSpeechSynthesisClient,
    SynthesisError,
    VoiceModelNotFoundError,
    concatenate_audio_segments,
    read_wav_file,
    synthesize_topic_script,
    write_wav_file,
)

# ---------------------------------------------------------------------------
# AudioSegment
# ---------------------------------------------------------------------------


def _pcm(num_frames, sample_width=2, channels=1, fill=b"\x01"):
    return fill * (num_frames * sample_width * channels)


def test_audio_segment_valid_construction():
    seg = AudioSegment(pcm_bytes=_pcm(100), sample_rate=22050, sample_width=2, channels=1)
    assert seg.duration_seconds == pytest.approx(100 / 22050)


@pytest.mark.parametrize("field,value", [("sample_rate", 0), ("sample_width", 0), ("channels", 0)])
def test_audio_segment_rejects_non_positive_fields(field, value):
    kwargs = dict(pcm_bytes=b"\x00\x00", sample_rate=22050, sample_width=2, channels=1)
    kwargs[field] = value
    with pytest.raises(AudioFormatError, match=f"{field} must be > 0"):
        AudioSegment(**kwargs)


def test_audio_segment_rejects_partial_frame():
    # 3 bytes at sample_width=2, channels=1 is not a whole number of frames.
    with pytest.raises(AudioFormatError, match="not a whole number of frames"):
        AudioSegment(pcm_bytes=b"\x00\x00\x00", sample_rate=22050, sample_width=2, channels=1)


def test_audio_segment_duration_accounts_for_channels_and_width():
    # 4-byte frames (sample_width=2, channels=2), 10 frames -> 40 bytes.
    seg = AudioSegment(pcm_bytes=_pcm(10, sample_width=2, channels=2), sample_rate=10, sample_width=2, channels=2)
    assert seg.duration_seconds == pytest.approx(1.0)


def test_format_matches():
    a = AudioSegment(pcm_bytes=_pcm(10), sample_rate=22050, sample_width=2, channels=1)
    b = AudioSegment(pcm_bytes=_pcm(5), sample_rate=22050, sample_width=2, channels=1)
    c = AudioSegment(pcm_bytes=_pcm(5), sample_rate=16000, sample_width=2, channels=1)
    assert a.format_matches(b)
    assert not a.format_matches(c)


def test_silence_produces_correct_frame_count_and_zeroed_bytes():
    seg = AudioSegment(pcm_bytes=_pcm(1), sample_rate=100, sample_width=2, channels=1)
    quiet = seg.silence(0.5)
    assert quiet.duration_seconds == pytest.approx(0.5)
    assert quiet.pcm_bytes == b"\x00" * (50 * 2)


def test_silence_rejects_negative_seconds():
    seg = AudioSegment(pcm_bytes=_pcm(1), sample_rate=100, sample_width=2, channels=1)
    with pytest.raises(AudioFormatError, match="seconds must be >= 0"):
        seg.silence(-1)


# ---------------------------------------------------------------------------
# concatenate_audio_segments
# ---------------------------------------------------------------------------


def test_concatenate_single_segment_returns_equivalent_audio():
    seg = AudioSegment(pcm_bytes=_pcm(10, fill=b"\x02"), sample_rate=100, sample_width=2, channels=1)
    result = concatenate_audio_segments([seg])
    assert result.pcm_bytes == seg.pcm_bytes


def test_concatenate_inserts_silence_gap_of_requested_duration():
    a = AudioSegment(pcm_bytes=_pcm(10, fill=b"\x01"), sample_rate=100, sample_width=2, channels=1)
    b = AudioSegment(pcm_bytes=_pcm(10, fill=b"\x01"), sample_rate=100, sample_width=2, channels=1)
    result = concatenate_audio_segments([a, b], gap_seconds=0.5)
    # 10 frames + 50 silence frames + 10 frames = 70 frames of 2 bytes each.
    assert len(result.pcm_bytes) == 70 * 2
    assert result.duration_seconds == pytest.approx(0.7)
    # Gap region (frames 10-59) should be all zero.
    gap_region = result.pcm_bytes[10 * 2 : 60 * 2]
    assert gap_region == b"\x00" * len(gap_region)


def test_concatenate_zero_gap_produces_no_silence():
    a = AudioSegment(pcm_bytes=_pcm(5, fill=b"\x01"), sample_rate=100, sample_width=2, channels=1)
    b = AudioSegment(pcm_bytes=_pcm(5, fill=b"\x01"), sample_rate=100, sample_width=2, channels=1)
    result = concatenate_audio_segments([a, b], gap_seconds=0)
    assert result.pcm_bytes == a.pcm_bytes + b.pcm_bytes


def test_concatenate_rejects_empty_list():
    with pytest.raises(AudioFormatError, match="at least one segment"):
        concatenate_audio_segments([])


def test_concatenate_rejects_negative_gap():
    seg = AudioSegment(pcm_bytes=_pcm(1), sample_rate=100, sample_width=2, channels=1)
    with pytest.raises(AudioFormatError, match="gap_seconds must be >= 0"):
        concatenate_audio_segments([seg, seg], gap_seconds=-1)


def test_concatenate_rejects_mismatched_sample_rate():
    a = AudioSegment(pcm_bytes=_pcm(5), sample_rate=100, sample_width=2, channels=1)
    b = AudioSegment(pcm_bytes=_pcm(5), sample_rate=200, sample_width=2, channels=1)
    with pytest.raises(AudioFormatError, match="segment 1 format"):
        concatenate_audio_segments([a, b])


def test_concatenate_rejects_mismatched_channels():
    a = AudioSegment(pcm_bytes=_pcm(5, channels=1), sample_rate=100, sample_width=2, channels=1)
    b = AudioSegment(pcm_bytes=_pcm(5, channels=2), sample_rate=100, sample_width=2, channels=2)
    with pytest.raises(AudioFormatError, match="segment 1 format"):
        concatenate_audio_segments([a, b])


def test_concatenate_three_segments_preserves_order():
    a = AudioSegment(pcm_bytes=b"\x01\x00" * 5, sample_rate=100, sample_width=2, channels=1)
    b = AudioSegment(pcm_bytes=b"\x02\x00" * 5, sample_rate=100, sample_width=2, channels=1)
    c = AudioSegment(pcm_bytes=b"\x03\x00" * 5, sample_rate=100, sample_width=2, channels=1)
    result = concatenate_audio_segments([a, b, c], gap_seconds=0)
    assert result.pcm_bytes == a.pcm_bytes + b.pcm_bytes + c.pcm_bytes


# ---------------------------------------------------------------------------
# write_wav_file / read_wav_file round-trip
# ---------------------------------------------------------------------------


def test_wav_round_trip(tmp_path):
    seg = AudioSegment(pcm_bytes=_pcm(1000, fill=b"\x07\x08"), sample_rate=22050, sample_width=2, channels=1)
    path = tmp_path / "out.wav"
    write_wav_file(seg, path)
    result = read_wav_file(path)
    assert result.pcm_bytes == seg.pcm_bytes
    assert result.sample_rate == seg.sample_rate
    assert result.sample_width == seg.sample_width
    assert result.channels == seg.channels


def test_wav_round_trip_stereo(tmp_path):
    seg = AudioSegment(pcm_bytes=_pcm(50, sample_width=2, channels=2, fill=b"\x01\x02"), sample_rate=16000, sample_width=2, channels=2)
    path = tmp_path / "stereo.wav"
    write_wav_file(seg, path)
    result = read_wav_file(path)
    assert result.pcm_bytes == seg.pcm_bytes
    assert result.channels == 2


def test_written_wav_file_is_valid_wave_container(tmp_path):
    seg = AudioSegment(pcm_bytes=_pcm(10), sample_rate=22050, sample_width=2, channels=1)
    path = tmp_path / "check.wav"
    write_wav_file(seg, path)
    # Independently verify with the stdlib wave reader directly (not through
    # our own read_wav_file), so this test doesn't just check symmetry
    # between our own write/read functions.
    with wave.open(str(path), "rb") as f:
        assert f.getframerate() == 22050
        assert f.getsampwidth() == 2
        assert f.getnchannels() == 1
        assert f.getnframes() == 10


# ---------------------------------------------------------------------------
# PiperSpeechSynthesisClient -- file-existence validation (no real model needed)
# ---------------------------------------------------------------------------


def test_piper_client_raises_when_model_file_missing(tmp_path):
    missing_model = tmp_path / "nonexistent-voice.onnx"
    with pytest.raises(VoiceModelNotFoundError, match="nonexistent-voice.onnx"):
        PiperSpeechSynthesisClient(model_path=missing_model)


def test_piper_client_raises_when_config_file_missing(tmp_path):
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"not a real onnx file")
    # voice.onnx.json is not created, so the config file is missing.
    with pytest.raises(VoiceModelNotFoundError, match="voice.onnx.json"):
        PiperSpeechSynthesisClient(model_path=model_path)


def test_piper_client_error_message_mentions_manual_download_step(tmp_path):
    missing_model = tmp_path / "nonexistent-voice.onnx"
    with pytest.raises(VoiceModelNotFoundError, match="huggingface.co"):
        PiperSpeechSynthesisClient(model_path=missing_model)


def test_piper_client_requires_model_path_or_voice():
    with pytest.raises(ValueError, match="requires model_path"):
        PiperSpeechSynthesisClient()


# ---------------------------------------------------------------------------
# PiperSpeechSynthesisClient.synthesize -- real conversion logic, fake voice
# ---------------------------------------------------------------------------


class _FakeAudioChunk:
    """Shaped like a real piper.AudioChunk -- same attribute names used by our code."""

    def __init__(self, pcm_bytes, sample_rate=22050, sample_width=2, sample_channels=1):
        self._pcm_bytes = pcm_bytes
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.sample_channels = sample_channels

    @property
    def audio_int16_bytes(self):
        return self._pcm_bytes


class _FakePiperVoice:
    """Test double for a loaded piper.PiperVoice -- returns canned chunks."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def synthesize(self, text, syn_config=None):
        self.calls.append((text, syn_config))
        return self.chunks


def test_piper_client_synthesize_joins_chunks_into_one_segment():
    chunks = [
        _FakeAudioChunk(b"\x01\x00" * 5),
        _FakeAudioChunk(b"\x02\x00" * 5),
    ]
    fake_voice = _FakePiperVoice(chunks)
    client = PiperSpeechSynthesisClient(voice=fake_voice)

    result = client.synthesize("hello world")

    assert isinstance(result, AudioSegment)
    assert result.pcm_bytes == (b"\x01\x00" * 5) + (b"\x02\x00" * 5)
    assert result.sample_rate == 22050
    assert result.sample_width == 2
    assert result.channels == 1
    assert fake_voice.calls == [("hello world", None)]


def test_piper_client_synthesize_passes_synthesis_config_through():
    fake_voice = _FakePiperVoice([_FakeAudioChunk(b"\x01\x00")])
    syn_config = object()
    client = PiperSpeechSynthesisClient(voice=fake_voice, synthesis_config=syn_config)

    client.synthesize("hi")

    assert fake_voice.calls == [("hi", syn_config)]


def test_piper_client_synthesize_rejects_empty_text():
    fake_voice = _FakePiperVoice([_FakeAudioChunk(b"\x01\x00")])
    client = PiperSpeechSynthesisClient(voice=fake_voice)
    with pytest.raises(SynthesisError, match="empty text"):
        client.synthesize("   ")
    assert fake_voice.calls == []


def test_piper_client_synthesize_raises_when_no_chunks_returned():
    fake_voice = _FakePiperVoice([])
    client = PiperSpeechSynthesisClient(voice=fake_voice)
    with pytest.raises(SynthesisError, match="no audio chunks"):
        client.synthesize("hello")


def test_piper_client_synthesize_wraps_malformed_chunk_audio():
    # sample_width=2, channels=1 but only 3 bytes -- not a whole frame.
    fake_voice = _FakePiperVoice([_FakeAudioChunk(b"\x01\x00\x02")])
    client = PiperSpeechSynthesisClient(voice=fake_voice)
    with pytest.raises(SynthesisError, match="malformed audio"):
        client.synthesize("hello")


# ---------------------------------------------------------------------------
# synthesize_topic_script -- orchestration + defense-in-depth privacy re-check
# ---------------------------------------------------------------------------


class FakeSynthesisClient:
    def __init__(self, segment):
        self.segment = segment
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return self.segment


def _valid_script(**overrides):
    base = dict(
        topic="git rebase",
        minutes=10.0,
        hook="On Tuesday you asked Claude to resolve a rebase conflict.",
        concept="A rebase replays commits onto a new base...",
        example="Imagine two branches that both edited the same line...",
        next_time="Next time, run `git status` before rebasing.",
    )
    base.update(overrides)
    return TopicScript(**base)


def test_synthesize_topic_script_happy_path():
    script = _valid_script()
    seg = AudioSegment(pcm_bytes=_pcm(10), sample_rate=22050, sample_width=2, channels=1)
    fake = FakeSynthesisClient(seg)

    result = synthesize_topic_script(script, fake)

    assert result is seg
    assert fake.calls == [script.full_text]


def test_synthesize_topic_script_rejects_leaked_content_before_calling_client():
    script = _valid_script(example="use this key: sk-abcdefghijklmnopqrstuvwxyz123456")
    fake = FakeSynthesisClient(AudioSegment(pcm_bytes=_pcm(1), sample_rate=100, sample_width=2, channels=1))

    with pytest.raises(PrivacyViolationError):
        synthesize_topic_script(script, fake)

    # Privacy is checked before the synthesis client is ever called.
    assert fake.calls == []
