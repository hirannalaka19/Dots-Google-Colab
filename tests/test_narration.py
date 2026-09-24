import datetime as dt

import numpy as np
import pytest
import soundfile as sf

import narration
from narration import Narration, ScriptError


def test_parse_script_reads_labelled_lines():
    script = "1.1: First line.\n1.2 | Second line.\n[2.1] Third line.\n3) Fourth.\n4 - Fifth."
    assert narration.parse_script(script) == [
        Narration("1.1", "First line."),
        Narration("1.2", "Second line."),
        Narration("2.1", "Third line."),
        Narration("3", "Fourth."),
        Narration("4", "Fifth."),
    ]


def test_parse_script_dotted_id_needs_no_separator():
    assert narration.parse_script("1.1 Once upon a time") == [Narration("1.1", "Once upon a time")]


def test_parse_script_joins_continuation_lines_and_id_only_lines():
    script = "1.1:\nThe storm grew.\nNobody slept.\n\n1.2: Dawn came."
    assert narration.parse_script(script) == [
        Narration("1.1", "The storm grew. Nobody slept."),
        Narration("1.2", "Dawn came."),
    ]


def test_parse_script_bare_ids_alone_on_their_lines():
    script = "1\nHello world, this is take one.\n2\nSecond take here."
    assert narration.parse_script(script) == [
        Narration("1", "Hello world, this is take one."),
        Narration("2", "Second take here."),
    ]


def test_parse_script_does_not_mistake_a_year_for_an_id():
    script = "1.1: It began quietly.\n2024 was the year everything changed."
    assert narration.parse_script(script) == [
        Narration("1.1", "It began quietly. 2024 was the year everything changed.")
    ]


def test_parse_script_without_ids_numbers_each_line():
    assert narration.parse_script("Hello.\n\nWorld.") == [
        Narration("1", "Hello."),
        Narration("2", "World."),
    ]


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ("", "empty"),
        ("   \n  ", "empty"),
        ("Intro text\n1.1: First", "before the first narration id"),
        ("1.1: A\n1.1: B", "more than once"),
        ("1.1:\n1.2: B", "no text: 1.1"),
    ],
)
def test_parse_script_rejects_bad_scripts(script, message):
    with pytest.raises(ScriptError, match=message):
        narration.parse_script(script)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("1.1", "1.1"),
        ("  Chapter 1: Intro  ", "Chapter 1_ Intro"),
        ("a/b\\c", "a_b_c"),
        ("..hidden..", "hidden"),
        ("", ""),
        (None, ""),
    ],
)
def test_clean_name(name, expected):
    assert narration.clean_name(name) == expected


def test_clean_name_is_length_limited():
    assert len(narration.clean_name("x" * 500)) == narration.MAX_NAME_LENGTH


@pytest.mark.parametrize(
    ("name", "expected"),
    [("1.1", "1.2"), ("1.9", "1.10"), ("3", "4"), ("scene_09", "scene_10"), ("intro", ""), ("", "")],
)
def test_next_name(name, expected):
    assert narration.next_name(name) == expected


def test_project_folder_uses_project_and_date(tmp_path):
    folder = narration.project_folder(tmp_path, "My: Story", "2026-09-25")
    assert folder == tmp_path / "My_ Story_2026-09-25"


def test_project_folder_falls_back_to_defaults(tmp_path):
    folder = narration.project_folder(tmp_path, "  ", "not a date")
    assert folder.name == f"{narration.DEFAULT_PROJECT}_{dt.date.today().isoformat()}"


def test_save_take_writes_pcm16_and_reports_replacement(tmp_path):
    audio = np.array([0.0, 0.5, 2.0, -2.0], dtype=np.float32)
    path, replaced = narration.save_take(tmp_path / "proj", "1.1", audio, 48000)
    assert path.name == "1.1.wav" and not replaced
    info = sf.info(str(path))
    assert info.samplerate == 48000 and info.subtype == "PCM_16"
    data, _ = sf.read(str(path))
    assert data.max() <= 1.0 and data.min() >= -1.0

    _, replaced_again = narration.save_take(tmp_path / "proj", "1.1", audio, 48000)
    assert replaced_again


def test_copy_subtitles_names_files_after_the_take(tmp_path):
    source = tmp_path / "whatever.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
    saved = narration.copy_subtitles(tmp_path, "1.1", {"": str(source), "_words": None, "_x": "missing.srt"})
    assert [p.name for p in saved] == ["1.1.srt"]
