import datetime as dt

import numpy as np
import pytest
import soundfile as sf

import narration
from narration import Narration, ScriptError


def narrations(script: str) -> list[Narration]:
    return narration.parse_script(script).narrations


def test_parse_script_reads_labelled_lines():
    script = "1.1: First line.\n1.2 | Second line.\n[2.1] Third line.\n(2.2) Fourth.\n2.3 — Fifth."
    assert narrations(script) == [
        Narration("1.1", "First line."),
        Narration("1.2", "Second line."),
        Narration("2.1", "Third line."),
        Narration("2.2", "Fourth."),
        Narration("2.3", "Fifth."),
    ]


def test_parse_script_reads_whole_number_ids():
    script = "1: First.\n2) Second.\n3 - Third.\n4 — Fourth.\n12 | Fifth."
    assert [n.name for n in narrations(script)] == ["1", "2", "3", "4", "12"]


@pytest.mark.parametrize(
    "prose",
    [
        "3: three strikes and you are out.",
        "6:19 the clock struck.",
        "1. First item.",
        "2) Second item.",
        "12 | Nine.",
    ],
)
def test_dotted_scripts_treat_whole_numbers_as_content(prose):
    script = f"1.1: Steps.\n{prose}\n1.2: Done."
    assert narrations(script) == [Narration("1.1", f"Steps. {prose}"), Narration("1.2", "Done.")]


def test_whole_number_scripts_still_guard_times_and_lowercase():
    script = "1: The bell rang.\n6:19 the clock struck.\n3: three strikes.\n2: Done."
    assert narrations(script) == [
        Narration("1", "The bell rang. 6:19 the clock struck. 3: three strikes."),
        Narration("2", "Done."),
    ]


def test_parse_script_handles_markdown():
    script = "## CHAPTER 1\n**1.1** First line.\n**CHAPTER 2**\n- 1.2: Second *line*.\n# Notes for the editor"
    parsed = narration.parse_script(script)
    assert parsed.narrations == [Narration("1.1", "First line."), Narration("1.2", "Second line.")]
    assert parsed.ignored == ["## CHAPTER 1", "**CHAPTER 2**", "# Notes for the editor"]


def test_parse_script_handles_full_width_ids():
    assert narrations("１.１：First line.\n１.２：Second.") == [
        Narration("1.1", "First line."),
        Narration("1.2", "Second."),
    ]


def test_parse_script_handles_crlf_tabs_and_nbsp():
    assert narrations("1.1:\tHello.\r\n1.2:\xa0World.\r\n") == [
        Narration("1.1", "Hello."),
        Narration("1.2", "World."),
    ]


def test_unclosed_bracket_never_deletes_the_lines_after_it():
    text = narrations("1.1: Hello [note continues\nreal important sentence here] world.")[0].text
    assert "real important sentence here" in text and text.endswith("world.")
    assert "[" not in text and "]" not in text


def test_nested_brackets_are_removed_whole():
    assert narrations("1.1: Hello [outer [inner] still outer] world.") == [
        Narration("1.1", "Hello world.")
    ]


def test_parse_script_dotted_id_needs_no_separator():
    assert narrations("1.1 Once upon a time") == [Narration("1.1", "Once upon a time")]


def test_parse_script_joins_continuation_lines_and_id_only_lines():
    script = "1.1:\nThe storm grew.\nNobody slept.\n\n1.2: Dawn came."
    assert narrations(script) == [
        Narration("1.1", "The storm grew. Nobody slept."),
        Narration("1.2", "Dawn came."),
    ]


def test_parse_script_bare_ids_alone_on_their_lines():
    script = "1\nHello world, this is take one.\n2\nSecond take here."
    assert narrations(script) == [
        Narration("1", "Hello world, this is take one."),
        Narration("2", "Second take here."),
    ]


@pytest.mark.parametrize(
    "prose",
    [
        "2024 was the year everything changed.",
        "1.5 million people fled the city.",
        "“1.5 million people fled,” he said.",
        "Part of the problem was the heat.",
        "Chapter one. The storm had been building all afternoon.",
    ],
)
def test_parse_script_keeps_prose_that_starts_like_an_id_or_heading(prose):
    assert narrations(f"1.1: It began quietly.\n{prose}") == [
        Narration("1.1", f"It began quietly. {prose}")
    ]


def test_parse_script_without_ids_numbers_each_line():
    assert narrations("Hello.\n\nWorld.") == [
        Narration("1", "Hello."),
        Narration("2", "World."),
    ]


# The shape of a real production script: chapter headings, ★/⭐ markers,
# em-dash separators, [production notes] after the id, and stray quote lines.
PRODUCTION_SCRIPT = """"
CHAPTER 1

1.1 — [RETENTION-FIRST OPENING] [COLD OPEN: STATED INSIDE 30 SECONDS] [CHAR: 694]
At five minutes to one in the morning, a torpedo tore open a troopship.
A rabbi stopped him.

1.2 — [SLUG + CONTEXT] [CHAR: 664]
The Labrador Sea.


CHAPTER 2

★ 2.4 — [ASK 1 of 3: PLACEMENT QUESTION, ≈6:19–7:01] [CHAR: 557]
Then the locker was empty.

2.5 — ⭐ [PAYLOAD: THE JACKETS] [CHAR: 590]
The four of them unbuckled their own life jackets.
\""""


def test_parse_script_reads_a_production_script():
    parsed = narration.parse_script(PRODUCTION_SCRIPT)
    assert parsed.narrations == [
        Narration("1.1", "At five minutes to one in the morning, a torpedo tore open a troopship. A rabbi stopped him."),
        Narration("1.2", "The Labrador Sea."),
        Narration("2.4", "Then the locker was empty."),
        Narration("2.5", "The four of them unbuckled their own life jackets."),
    ]
    assert parsed.ignored == ["CHAPTER 1", "CHAPTER 2"]


@pytest.mark.parametrize(
    "heading",
    ["CHAPTER 1", "Chapter 12", "Part II", "PART TWO — THE WATER", "Act 3: Aftermath", "Episode 4"],
)
def test_parse_script_skips_headings(heading):
    parsed = narration.parse_script(f"1.1: First.\n{heading}\n1.2: Second.")
    assert parsed.narrations == [Narration("1.1", "First."), Narration("1.2", "Second.")]
    assert parsed.ignored == [heading]


def test_parse_script_reports_text_before_the_first_id():
    parsed = narration.parse_script("THE FOUR CHAPLAINS\n1.1: First.")
    assert parsed.narrations == [Narration("1.1", "First.")]
    assert parsed.ignored == ["THE FOUR CHAPLAINS"]


def test_parse_script_drops_bracketed_notes_and_symbols_from_speech():
    assert narrations("1.1: Hello [pause] world [beat].\n1.2: ⭐ Wow ★!") == [
        Narration("1.1", "Hello world."),
        Narration("1.2", "Wow!"),
    ]


def test_parse_script_without_ids_skips_headings_and_notes():
    parsed = narration.parse_script("CHAPTER 1\nHello [soft].\n\nWorld.")
    assert parsed.narrations == [Narration("1", "Hello."), Narration("2", "World.")]
    assert parsed.ignored == ["CHAPTER 1"]


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ("", "empty"),
        ("   \n  ", "empty"),
        ('"\n---\n"', "empty"),
        ("CHAPTER 1\nCHAPTER 2", "nothing to read"),
        ("1.1: A\n1.1: B", "more than once"),
        ("1.1:\n1.2: B", "no text: 1.1"),
        ("1.1 — [ONLY A NOTE] [CHAR: 12]\n1.2: B", "no text: 1.1"),
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
