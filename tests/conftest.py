import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 48_000


def tone(seconds: float, sample_rate: int = SAMPLE_RATE, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeRuntime:
    """Stands in for DotsTtsRuntime: records calls, returns a tone per character."""

    sample_rate = SAMPLE_RATE

    def __init__(self, audio_fn=None):
        self.calls: list[dict] = []
        self.audio_fn = audio_fn or (lambda text: tone(0.05 * len(text)))

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        audio = self.audio_fn(kwargs["text"])
        return {"audio": torch.from_numpy(np.asarray(audio)).unsqueeze(0), "sample_rate": SAMPLE_RATE}


@pytest.fixture
def runtime():
    return FakeRuntime()
