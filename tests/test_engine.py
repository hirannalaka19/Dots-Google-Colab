import numpy as np
import pytest
import soundfile as sf
import torch

import engine
from conftest import SAMPLE_RATE, FakeRuntime, tone
from engine import Settings, SynthesisError, Voice


def test_synthesize_single_chunk_passes_settings_through(runtime, tmp_path):
    settings = Settings(language="EN", num_steps=16, guidance_scale=1.5, speaker_scale=2.0, normalize_text=True)
    take = engine.synthesize(runtime, "Hello there.", Voice(), settings, seed=7, work_dir=tmp_path)

    assert take.chunk_count == 1 and take.seed == 7 and take.sample_rate == SAMPLE_RATE
    call = runtime.calls[0]
    assert call["text"] == "Hello there."
    assert call["prompt_audio_path"] is None and call["prompt_text"] is None
    assert (call["language"], call["num_steps"], call["guidance_scale"]) == ("EN", 16, 1.5)
    assert (call["speaker_scale"], call["normalize_text"]) == (2.0, True)


def test_language_not_set_is_sent_as_none(runtime, tmp_path):
    engine.synthesize(runtime, "Hi.", Voice(), Settings(), seed=1, work_dir=tmp_path)
    assert runtime.calls[0]["language"] is None


def test_clone_voice_sends_reference_and_transcript(runtime, tmp_path):
    voice = Voice(audio_path="ref.wav", transcript="exact words")
    engine.synthesize(runtime, "Hi.", voice, Settings(), seed=1, work_dir=tmp_path)
    assert runtime.calls[0]["prompt_audio_path"] == "ref.wav"
    assert runtime.calls[0]["prompt_text"] == "exact words"


def test_timbre_only_voice_sends_no_transcript(runtime, tmp_path):
    engine.synthesize(runtime, "Hi.", Voice(audio_path="ref.wav"), Settings(), seed=1, work_dir=tmp_path)
    assert runtime.calls[0]["prompt_text"] is None


def test_long_random_voice_text_locks_the_first_chunk_voice(runtime, tmp_path):
    text = "This is sentence number one. " * 12
    take = engine.synthesize(runtime, text, Voice(), Settings(chunk_chars=80, chunk_pause=0.5),
                             seed=3, work_dir=tmp_path)

    assert take.chunk_count == len(runtime.calls) > 1
    first, later = runtime.calls[0], runtime.calls[1:]
    assert first["prompt_audio_path"] is None
    lock_paths = {call["prompt_audio_path"] for call in later}
    assert len(lock_paths) == 1 and None not in lock_paths
    assert all(call["prompt_text"] == first["text"] for call in later)
    pieces = sum(len(tone(0.05 * len(call["text"]))) for call in runtime.calls)
    assert len(take.audio) == pieces + int(0.5 * SAMPLE_RATE) * (take.chunk_count - 1)
    assert take.first_chunk_text == first["text"]


def test_auto_split_off_sends_text_whole(runtime, tmp_path):
    text = "Sentence. " * 60
    engine.synthesize(runtime, text, Voice(), Settings(auto_split=False), seed=1, work_dir=tmp_path)
    assert len(runtime.calls) == 1


def test_empty_text_is_rejected(runtime, tmp_path):
    with pytest.raises(SynthesisError, match="no text"):
        engine.synthesize(runtime, "  ", Voice(), Settings(), seed=1, work_dir=tmp_path)


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (np.array([np.nan, 0.1], dtype=np.float32), "NaN"),
        (np.zeros(4800, dtype=np.float32), "silence"),
        (np.zeros(0, dtype=np.float32), "silence"),
    ],
)
def test_broken_audio_is_reported(tmp_path, audio, message):
    runtime = FakeRuntime(audio_fn=lambda text: audio)
    with pytest.raises(SynthesisError, match=message):
        engine.synthesize(runtime, "Hi.", Voice(), Settings(), seed=1, work_dir=tmp_path)


def test_failures_say_which_chunk(tmp_path):
    calls = []

    def audio_fn(text):
        calls.append(text)
        return tone(0.5) if len(calls) == 1 else np.zeros(10, dtype=np.float32)

    runtime = FakeRuntime(audio_fn=audio_fn)
    with pytest.raises(SynthesisError, match=r"chunk 2 of"):
        engine.synthesize(runtime, "Sentence one here. " * 20, Voice(), Settings(chunk_chars=80),
                          seed=1, work_dir=tmp_path)


def test_runtime_value_error_becomes_a_friendly_error(tmp_path):
    class Rejecting(FakeRuntime):
        def generate(self, **kwargs):
            raise ValueError("Unsupported language='XX'.")

    with pytest.raises(SynthesisError, match="Unsupported language"):
        engine.synthesize(Rejecting(), "Hi.", Voice(), Settings(), seed=1, work_dir=tmp_path)


def test_out_of_memory_becomes_a_friendly_error(tmp_path):
    class OutOfMemory(FakeRuntime):
        def generate(self, **kwargs):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(SynthesisError, match="out of memory"):
        engine.synthesize(OutOfMemory(), "Hi.", Voice(), Settings(), seed=1, work_dir=tmp_path)


def test_same_seed_reseeds_identically(tmp_path):
    runtime = FakeRuntime(audio_fn=lambda text: (np.random.rand(4800).astype(np.float32) - 0.5))
    first = engine.synthesize(runtime, "Hi.", Voice(), Settings(), seed=99, work_dir=tmp_path)
    second = engine.synthesize(runtime, "Hi.", Voice(), Settings(), seed=99, work_dir=tmp_path)
    np.testing.assert_array_equal(first.audio, second.audio)


def _write(path, seconds):
    sf.write(str(path), tone(seconds, 16000), 16000)
    return str(path)


def test_shorten_reference_keeps_short_clips(tmp_path):
    path = _write(tmp_path / "short.wav", 5)
    assert engine.shorten_reference(path, tmp_path / "work") == path


def test_shorten_reference_cuts_long_clips(tmp_path):
    path = _write(tmp_path / "long.wav", 40)
    shortened = engine.shorten_reference(path, tmp_path / "work")
    assert shortened != path
    low, high = engine.REFERENCE_CUT_WINDOW
    assert low <= engine.audio_seconds(shortened) <= high


def test_shorten_reference_rejects_tiny_clips(tmp_path):
    path = _write(tmp_path / "tiny.wav", 0.3)
    with pytest.raises(SynthesisError, match="too short|only"):
        engine.shorten_reference(path, tmp_path / "work")


def _take(seconds):
    audio = tone(seconds)
    return engine.Take(SAMPLE_RATE, audio, 1, 1, 0.0, audio, "exact words")


def test_voice_from_take_keeps_transcript_for_short_takes(tmp_path):
    voice = engine.voice_from_take(_take(5), tmp_path)
    assert voice.transcript == "exact words"
    assert engine.audio_seconds(voice.audio_path) == pytest.approx(5, abs=0.01)


def test_voice_from_take_drops_transcript_when_it_must_cut(tmp_path):
    voice = engine.voice_from_take(_take(25), tmp_path)
    assert voice.transcript == ""
    assert engine.audio_seconds(voice.audio_path) <= engine.MAX_REFERENCE_SECONDS


def test_take_duration():
    assert _take(2).duration == pytest.approx(2)


def test_takes_compare_by_identity_without_touching_numpy():
    first, second = _take(1), _take(1)
    assert first == first and first != second
    assert len({first, second}) == 2
