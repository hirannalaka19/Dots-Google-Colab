"""Narration names, batch scripts, and the project folder they are saved in.

Every take is saved as `<narration name>.wav` (e.g. `1.1.wav`) inside
`<output root>/<project name>_<YYYY-MM-DD>/`. On Colab the output root is a
folder in Google Drive, so takes survive the runtime shutting down.
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

DEFAULT_PROJECT = "My Project"
MAX_NAME_LENGTH = 80

# "1.1 text", "1.1: text", "[1.1] text", "(2.2) text", "2.4 — text", "★ 2.4 — text",
# "**1.1** text", "3) text", "12 | text", or an id alone on its line.
_LABEL_LINE = re.compile(
    r"^\s*(?P<marker>[^\w\s\[(]*)\s*"  # optional decoration such as ★, •, - or **
    r"(?P<open>[\[(])?(?P<label>\d+(?:\.\d+)*)(?P<close>[\])])?[*_`]*"
    r"(?P<sep>\s*[:|.\-–—]\s*|\s+|\s*$)(?P<text>.*)$"
)
_EXPLICIT_SEPARATOR = re.compile(r"[:|.\-–—]")
# Full-width digits and separators, mapped one-to-one so positions don't move.
_FULL_WIDTH = str.maketrans("０１２３４５６７８９．：｜－（）［］", "0123456789.:|-()[]")
_WIDE_BRACKETS = str.maketrans("［］", "[]")
_NUMBER_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
# "CHAPTER 1", "Part II", "PART TWO — THE WATER": structure, not speech. The
# title after the number may not end a sentence, so prose such as
# "Chapter one. The storm had been building." is still read.
_HEADING = re.compile(
    rf"^(chapter|part|section|act|episode|book|scene)\s+(\d+|[ivxlcdm]+|{_NUMBER_WORDS})"
    r"(\s*[:.\-–—]\s*[^.!?]*)?$",
    re.IGNORECASE,
)
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s")
# Bullets, stars and emphasis around a line, stripped before looking for a heading.
_DECORATION = re.compile(r"^[^\w\[(]+|[^\w.!?)\]]+$")
# [Production notes] are for the reader, not the listener. Innermost first, so
# nested notes go whole; a bracket that is never closed loses only the bracket.
_NOTE = re.compile(r"\[[^\[\]]*\]")
_MARKUP = re.compile(r"[\[\]*`]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.!?;:])")
_HAS_WORD = re.compile(r"\w")
# Characters Windows, macOS, and Google Drive all refuse in file names.
_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ScriptError(ValueError):
    """A batch script that cannot be turned into narrations; message is user-facing."""


@dataclass(frozen=True)
class Narration:
    name: str
    text: str


class ParsedScript(NamedTuple):
    narrations: list[Narration]
    # Lines skipped entirely (headings, text before the first id), for the status box.
    ignored: list[str]


def spoken_text(text: str) -> str:
    """What gets read aloud from one line: no [notes], markup or ★-style symbols."""
    text = text.translate(_WIDE_BRACKETS)
    previous = None
    while previous != text:
        previous, text = text, _NOTE.sub(" ", text)
    text = _MARKUP.sub(" ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "So")
    text = re.sub(r"\s+", " ", text).strip()
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)


def _is_heading(line: str) -> bool:
    return bool(_MARKDOWN_HEADING.match(line) or _HEADING.match(_DECORATION.sub("", line)))


def _parse_label_line(line: str) -> tuple[str, str] | None:
    """Return (label, rest of line) if `line` could start a new narration, else None.

    Prose that merely starts with a number is left alone: "6:19 the clock",
    "2024 was the year", "1.5 million people", "3: three strikes". A dotted id
    with only a space after it ("1.1 Once upon…") needs a capital after it.
    """
    match = _LABEL_LINE.match(line.translate(_FULL_WIDTH))
    if not match:
        return None
    label, separator = match.group("label"), match.group("sep")
    text = line[match.start("text"):].strip()
    if not text:
        return label, text  # an id alone on its line, text on the lines below
    if separator and not separator[-1].isspace() and text[0].isdigit():
        return None  # a time such as 6:19, or the rest of a longer number
    explicit = match.group("open") or match.group("close") or _EXPLICIT_SEPARATOR.search(separator)
    starts_lower = text[0].islower()
    if "." in label:
        return (label, text) if explicit or not starts_lower else None
    return (label, text) if explicit and not starts_lower else None


def _check(narrations: list[Narration]) -> list[Narration]:
    if not narrations:
        raise ScriptError("The script has nothing to read — only headings or notes.")
    names = [narration.name for narration in narrations]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ScriptError(
            "These narration ids appear more than once, so their files would "
            f"overwrite each other: {', '.join(duplicates)}"
        )
    empty = [narration.name for narration in narrations if not narration.text]
    if empty:
        raise ScriptError(f"These narrations have no text: {', '.join(empty)}")
    return narrations


def parse_script(script: str) -> ParsedScript:
    """Turn a batch script into narrations.

    Lines that start with a narration id begin a new narration; lines without
    one continue the previous narration. Headings ("CHAPTER 1") and anything
    before the first id are skipped and reported. When a script uses dotted
    ids (1.1, 1.2) a whole number at the start of a line (a list item, "3:")
    is content, not a new narration. A script with no ids at all gets one
    narration per line, numbered 1, 2, 3, ...
    """
    lines = [line.strip() for line in (script or "").splitlines() if _HAS_WORD.search(line)]
    if not lines:
        raise ScriptError("The script is empty.")
    headings = [_is_heading(line) for line in lines]
    labels = [None if heading else _parse_label_line(line) for line, heading in zip(lines, headings)]
    if any(label and "." in label[0] for label in labels):
        labels = [label if label and "." in label[0] else None for label in labels]

    ignored: list[str] = []
    names: list[str] = []
    parts: list[list[str]] = []
    no_ids = not any(labels)
    for line, label, heading in zip(lines, labels, headings):
        if heading:
            ignored.append(line)
        elif no_ids:
            if spoken_text(line):
                names.append(str(len(names) + 1))
                parts.append([line])
        elif label:
            names.append(label[0])
            parts.append([label[1]])
        elif names:
            parts[-1].append(line)
        else:
            ignored.append(line)

    # Clean each line on its own, so a bracket left open can't swallow the next line.
    narrations = [
        Narration(name, " ".join(filter(None, (spoken_text(text) for text in texts))))
        for name, texts in zip(names, parts)
    ]
    return ParsedScript(_check(narrations), ignored)


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
