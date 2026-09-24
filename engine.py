"""Turns text into one finished take with a loaded dots.tts runtime.

`synthesize` works with anything that has dots.tts's `generate(...)` method
and a `sample_rate`, which keeps it testable without the 5 GB model (see
tests/test_engine.py).
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch

import tts_utils

# The dots.tts authors recommend ~10 s of reference audio; longer does not
# clone better, and every second of reference uses up the 80 s per-call budget.
MAX_REFERENCE_SECONDS = 15.0
REFERENCE_CUT_WINDOW = (10.0, 15.0)
MIN_REFERENCE_SECONDS = 1.0
# Generated speech is never this quiet; a take below it is a broken take.
SILENCE_PEAK = 1e-4

NAN_MESSAGE = (
    "The model produced invalid audio (NaN), which float16 can do on rare inputs. "
    "In the notebook's Run cell set precision to 'bfloat16' and run it again."
)
SILENT_MESSAGE = (
    "The model produced silence. Try another seed, a cleaner reference clip, "
    "or precision 'bfloat16' in the notebook's Run cell."
)
OOM_MESSAGE = (
    "The GPU ran out of memory. Lower 'Max characters per chunk', use a shorter "
    "reference clip, or pick a GPU with more memory."
)


class SynthesisError(Exception):
    """Generation failed; the message is written for the person using the app."""


@dataclass(frozen=True)
class Voice:
    """Who is speaking. No audio means the model picks a random voice."""

    audio_path: str | None = None
    # Exact words spoken in `audio_path`. Empty -> clone the timbre only.
    transcript: str = ""


@dataclass(frozen=True)
class Settings:
    language: str = tts_utils.LANGUAGE_NOT_SET
    num_steps: int = 10
    guidance_scale: float = 1.2
    speaker_scale: float = 1.5
    normalize_text: bool = False
    auto_split: bool = True
    chunk_chars: int = 250
    chunk_pause: float = 0.3


# eq=False: the generated __eq__/__hash__ would compare numpy arrays and raise.
@dataclass(frozen=True, eq=False)
class Take:
    sample_rate: int
    audio: np.ndarray  # mono float32 in [-1, 1]
    seed: int
    chunk_count: int
    elapsed: float
    # The first chunk on its own: a short, exactly-transcribed clip that later
    # takes can use as their reference to keep the same random voice.
    first_chunk_audio: np.ndarray
    first_chunk_text: str

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0


def seed_everything(seed: int) -> None:
    """Seed every RNG generation draws from, like the dots.tts CLI does."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def audio_seconds(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate if info.samplerate else 0.0


def shorten_reference(path: str, work_dir: Path) -> str:
    """Return `path`, or a copy cut at a quiet point if it is too long."""
    import librosa  # heavy import; only needed when a reference is used

    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    duration = len(audio) / sample_rate
    if duration < MIN_REFERENCE_SECONDS:
        raise SynthesisError(
            f"The reference clip is only {duration:.1f} s long. Use 5–15 s of clear speech."
        )
    if duration <= MAX_REFERENCE_SECONDS:
        return path

    cut = tts_utils.quietest_point(audio, sample_rate, *REFERENCE_CUT_WINDOW)
    work_dir.mkdir(parents=True, exist_ok=True)
    trimmed = work_dir / f"reference_{uuid.uuid4().hex[:8]}.wav"
    sf.write(str(trimmed), audio[:cut], sample_rate)
    return str(trimmed)


def voice_from_take(take: Take, work_dir: Path) -> Voice:
    """Reuse a take's voice for later text, via its (short) first chunk."""
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"voice_lock_{uuid.uuid4().hex[:8]}.wav"
    audio = take.first_chunk_audio
    transcript = take.first_chunk_text
    if len(audio) / take.sample_rate > MAX_REFERENCE_SECONDS:
        # Too long to use whole; a cut clip no longer matches its transcript,
        # so keep only the timbre.
        cut = tts_utils.quietest_point(audio, take.sample_rate, *REFERENCE_CUT_WINDOW)
        audio, transcript = audio[:cut], ""
    sf.write(str(path), audio, take.sample_rate)
    return Voice(audio_path=str(path), transcript=transcript)


def _generate_chunk(runtime: Any, text: str, voice: Voice, settings: Settings) -> np.ndarray:
    try:
        result = runtime.generate(
            text=text,
            prompt_audio_path=voice.audio_path,
            prompt_text=(voice.transcript or None) if voice.audio_path else None,
            language=None if settings.language == tts_utils.LANGUAGE_NOT_SET else settings.language,
            num_steps=int(settings.num_steps),
            guidance_scale=float(settings.guidance_scale),
            speaker_scale=float(settings.speaker_scale),
            normalize_text=bool(settings.normalize_text),
        )
        audio = result["audio"].detach().float().cpu().reshape(-1).numpy()
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise SynthesisError(OOM_MESSAGE) from exc
    except ValueError as exc:  # dots.tts reports bad input as ValueError
        raise SynthesisError(str(exc)) from exc

    if audio.size == 0:
        raise SynthesisError(SILENT_MESSAGE)
    if not np.isfinite(audio).all():
        raise SynthesisError(NAN_MESSAGE)
    if float(np.max(np.abs(audio))) < SILENCE_PEAK:
        raise SynthesisError(SILENT_MESSAGE)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def synthesize(
    runtime: Any,
    text: str,
    voice: Voice,
    settings: Settings,
    *,
    seed: int,
    work_dir: Path,
    on_chunk: Callable[[int, int], None] | None = None,
) -> Take:
    """Generate `text` as one take, splitting long text into chunks.

    With no reference voice, the first chunk's voice is reused for the rest so
    a long take does not change speaker halfway through.
    """
    text = (text or "").strip()
    if not text:
        raise SynthesisError("There is no text to read.")

    if settings.auto_split:
        chunks = tts_utils.split_into_chunks(
            text, settings.chunk_chars, tts_utils.whisper_code(settings.language)
        )
    else:
        chunks = [text]

    seed_everything(seed)
    started = time.time()
    pieces: list[np.ndarray] = []
    chunk_voice = voice
    for index, chunk in enumerate(chunks):
        if on_chunk:
            on_chunk(index, len(chunks))
        try:
            pieces.append(_generate_chunk(runtime, chunk, chunk_voice, settings))
        except SynthesisError as exc:
            where = f" (chunk {index + 1} of {len(chunks)})" if len(chunks) > 1 else ""
            raise SynthesisError(f"{exc}{where}") from exc

        if index == 0 and voice.audio_path is None and len(chunks) > 1:
            first = Take(runtime.sample_rate, pieces[0], seed, 1, 0.0, pieces[0], chunk)
            chunk_voice = voice_from_take(first, work_dir)

    return Take(
        sample_rate=runtime.sample_rate,
        audio=tts_utils.join_with_pauses(pieces, runtime.sample_rate, settings.chunk_pause),
        seed=seed,
        chunk_count=len(chunks),
        elapsed=time.time() - started,
        first_chunk_audio=pieces[0],
        first_chunk_text=chunks[0],
    )
