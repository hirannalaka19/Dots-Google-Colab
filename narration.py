"""Narration names, batch scripts, and the project folder they are saved in.

Every take is saved as `<narration name>.wav` (e.g. `1.1.wav`) inside
`<output root>/<project name>_<YYYY-MM-DD>/`. On Colab the output root is a
folder in Google Drive, so takes survive the runtime shutting down.
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_PROJECT = "My Project"
MAX_NAME_LENGTH = 80

# "1.1 text", "1.1: text", "[1.1] text", "3) text", "12 | text", "2.4 - text",
# or an id alone on its line with the text on the lines below.
_LABEL_LINE = re.compile(
    r"^\s*(?P<open>[\[(])?(?P<label>\d+(?:\.\d+)*)(?P<close>[\])])?"
    r"(?P<sep>\s*[:|.\-–—]\s*|\s+|\s*$)(?P<text>.*)$"
)
_EXPLICIT_SEPARATOR = re.compile(r"[:|.\-–—]")
# Characters Windows, macOS, and Google Drive all refuse in file names.
_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ScriptError(ValueError):
    """A batch script that cannot be turned into narrations; message is user-facing."""


@dataclass(frozen=True)
class Narration:
    name: str
    text: str


def _parse_label_line(line: str) -> tuple[str, str] | None:
    """Return (label, text) if `line` starts a new narration, else None.

    A bare number followed by a space only counts when it looks like a
    narration id ("1.1 Once upon…"); otherwise "2024 was a good year" would
    be mistaken for narration 2024.
    """
    match = _LABEL_LINE.match(line)
    if not match:
        return None
    label, text = match.group("label"), match.group("text").strip()
    marked = (
        "." in label
        or not text  # an id alone on its line, text on the lines below
        or match.group("open")
        or match.group("close")
        or _EXPLICIT_SEPARATOR.search(match.group("sep"))
    )
    if not marked:
        return None
    return label, text


def parse_script(script: str) -> list[Narration]:
    """Turn a batch script into narrations.

    Lines that start with a narration id begin a new narration; lines without
    one continue the previous narration. A script with no ids at all gets one
    narration per non-empty line, numbered 1, 2, 3, ...
    """
    lines = [line.rstrip() for line in (script or "").splitlines()]
    if not any(line.strip() for line in lines):
        raise ScriptError("The script is empty.")

    labelled = [_parse_label_line(line) for line in lines]
    if not any(labelled):
        texts = [line.strip() for line in lines if line.strip()]
        return [Narration(str(index), text) for index, text in enumerate(texts, 1)]

    names: list[str] = []
    texts: list[list[str]] = []
    for line_number, (line, parsed) in enumerate(zip(lines, labelled), 1):
        if parsed:
            names.append(parsed[0])
            texts.append([parsed[1]])
        elif line.strip():
            if not names:
                raise ScriptError(
                    f"Line {line_number} has text before the first narration id. "
                    "Start it with an id such as 1.1:"
                )
            texts[-1].append(line.strip())

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ScriptError(
            "These narration ids appear more than once, so their files would "
            f"overwrite each other: {', '.join(duplicates)}"
        )
    narrations = [
        Narration(name, " ".join(part for part in parts if part))
        for name, parts in zip(names, texts)
    ]
    empty = [narration.name for narration in narrations if not narration.text]
    if empty:
        raise ScriptError(f"These narrations have no text: {', '.join(empty)}")
    return narrations


def clean_name(name: str) -> str:
    """Make a user-typed name safe to use as a file or folder name."""
    cleaned = _UNSAFE_CHARS.sub("_", str(name or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned[:MAX_NAME_LENGTH].strip(" .")


def next_name(name: str) -> str:
    """The narration name to suggest after `name`: 1.1 -> 1.2, 1.9 -> 1.10.

    Names without a trailing number return "" so the next take is never
    silently saved over this one.
    """
    match = re.search(r"(\d+)(?!.*\d)", name or "")
    if not match:
        return ""
    digits = match.group(1)
    bumped = str(int(digits) + 1).zfill(len(digits))
    return f"{name[:match.start()]}{bumped}{name[match.end():]}"


def project_folder(output_root: str | Path, project: str, date_text: str | None) -> Path:
    """`<output_root>/<project>_<YYYY-MM-DD>`, falling back to today's date."""
    date_text = (date_text or "").strip()
    if not _DATE.match(date_text):
        date_text = dt.date.today().isoformat()
    project_name = clean_name(project) or DEFAULT_PROJECT
    return Path(output_root) / f"{project_name}_{date_text}"


def save_take(folder: Path, name: str, audio: np.ndarray, sample_rate: int) -> tuple[Path, bool]:
    """Write `<folder>/<name>.wav` (16-bit PCM). Returns (path, replaced_existing)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.wav"
    replaced = path.exists()
    pcm = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    sf.write(str(path), pcm, sample_rate, subtype="PCM_16")
    return path, replaced


def copy_subtitles(folder: Path, name: str, srt_paths: dict[str, str | None]) -> list[Path]:
    """Copy generated SRTs next to the take as `<name><suffix>.srt`."""
    saved: list[Path] = []
    for suffix, source in srt_paths.items():
        if source and Path(source).is_file():
            target = folder / f"{name}{suffix}.srt"
            shutil.copyfile(source, target)
            saved.append(target)
    return saved
