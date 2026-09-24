"""Install dots.tts and the app's extras on top of Colab's preinstalled stack.

Run by the notebook's Install cell. Colab already ships torch, torchaudio,
transformers and gradio versions that dots.tts works with, so we keep them
instead of downloading gigabytes of replacements. Three things need care:

* dots.tts declares Python <3.13, but Colab moved to Python 3.13 in Sept 2026.
  The package is pure Python and runs fine there, so we pass
  --ignore-requires-python.
* dots.tts refuses to import unless torch and torchaudio share a minor
  version. If they don't, install the matching torchaudio (torchaudio stopped
  at 2.11, so a newer torch is replaced by the 2.11 pair).
* numpy is pinned in colab.txt so Colab's numba (and librosa with it) keeps working.

Writes `.install_ok` on success so the notebook knows whether to clear the log.
"""

from __future__ import annotations

import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIREMENTS = HERE / "colab.txt"
MARKER = HERE / ".install_ok"
LAST_TORCHAUDIO = (2, 11)  # torchaudio's final release line
PYTORCH_INDEX = "https://download.pytorch.org/whl/{tag}"
KNOWN_CUDA_TAGS = ("cu126", "cu128", "cu130")
IMPORT_CHECK = "import dots_tts.runtime, gradio, faster_whisper, sentencex, pysrt"


def run(*args: str) -> int:
    print("$", " ".join(args), flush=True)
    try:
        return subprocess.call(list(args))
    except FileNotFoundError:  # e.g. no nvidia-smi on a CPU runtime
        print(f"⚠️ {args[0]} not found")
        return 127


def pip_install(*args: str) -> int:
    return run(sys.executable, "-m", "pip", "install", "-q", *args)


def installed(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def minor(version: str) -> tuple[int, int]:
    major, minor_part = version.split("+")[0].split(".")[:2]
    return int(major), int(minor_part)


def cuda_tag(torch_version: str) -> str:
    """'2.11.0+cu130' -> 'cu130'; anything unexpected falls back to cu130."""
    local = torch_version.partition("+")[2]
    return local if local in KNOWN_CUDA_TAGS else "cu130"


def ensure_matching_torchaudio() -> bool:
    torch_version = installed("torch")
    if torch_version is None:
        print("❌ PyTorch is missing — this notebook expects a standard Colab runtime.")
        return False
    audio_version = installed("torchaudio")
    if audio_version and minor(audio_version) == minor(torch_version):
        print(f"✅ torch {torch_version} / torchaudio {audio_version}")
        return True

    index = PYTORCH_INDEX.format(tag=cuda_tag(torch_version))
    print(f"🔧 torch {torch_version} needs a matching torchaudio (found {audio_version}).")
    if minor(torch_version) <= LAST_TORCHAUDIO:
        base = torch_version.split("+")[0]
        if pip_install(f"torchaudio=={base}", "--index-url", index) == 0:
            return True
    print("🔧 Installing the torch 2.11 + torchaudio 2.11 pair (this takes a few minutes) ...")
    return pip_install("torch==2.11.0", "torchaudio==2.11.0", "--index-url", index) == 0


def main() -> int:
    MARKER.unlink(missing_ok=True)
    print(f"🐍 Python {sys.version.split()[0]}", flush=True)
    run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")

    if not ensure_matching_torchaudio():
        return 1

    flags = ["--ignore-requires-python"] if sys.version_info >= (3, 13) else []
    if pip_install(*flags, "-r", str(REQUIREMENTS)) != 0:
        print("❌ pip install failed — see the messages above.")
        return 1

    if run(sys.executable, "-c", IMPORT_CHECK) != 0:
        print("❌ Packages installed but dots.tts failed to import — see the error above.")
        return 1

    MARKER.write_text("ok\n", encoding="utf-8")
    print("✅ Everything installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
