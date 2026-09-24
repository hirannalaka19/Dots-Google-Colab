import numpy as np
import pytest

import tts_utils


def test_language_choices_start_with_special_options_and_cover_every_language():
    choices = tts_utils.language_choices()
    assert [value for _, value in choices[:2]] == ["none", "auto_detect"]
    assert len(choices) == len(tts_utils.LANGUAGES) + 2
    assert ("English", "EN") in choices


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("EN", "en"),
        ("NB", "no"),
        ("FIL", "tl"),
        ("口音:粤语", "yue"),
        ("口音:四川话", "zh"),
        ("AST", None),
        ("none", None),
        ("auto_detect", None),
        ("", None),
        (None, None),
    ],
)
def test_whisper_code_maps_dots_tags(tag, expected):
    assert tts_utils.whisper_code(tag) == expected


def test_spoken_weight_counts_cjk_characters_triple():
    assert tts_utils.spoken_weight("abc") == 3
    assert tts_utils.spoken_weight("你好") == 6
    assert tts_utils.spoken_weight("a你") == 4


def test_split_into_chunks_keeps_sentences_whole():
    text = "One sentence here. Another one follows. " * 20
    chunks = tts_utils.split_into_chunks(text, 100)
    assert len(chunks) > 1
    assert all(chunk.endswith(".") for chunk in chunks)
    assert all(tts_utils.spoken_weight(chunk) <= 150 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_split_into_chunks_short_text_is_one_chunk():
    assert tts_utils.split_into_chunks("Hello there.", 250) == ["Hello there."]


def test_split_into_chunks_empty_text():
    assert tts_utils.split_into_chunks("   ", 250) == []


def test_split_into_chunks_splits_chinese_on_full_width_stops():
    text = "今天天气很好。我们去公园散步吧！" * 10
    chunks = tts_utils.split_into_chunks(text, 60)
    assert len(chunks) > 1
    assert all(tts_utils.spoken_weight(chunk) <= 90 for chunk in chunks)
    assert "".join(chunks) == text  # no spaces added inside Chinese text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("真的吗？！世界很大。", ["真的吗？！", "世界很大。"]),
        ("他说：“好。”然后走了。", ["他说：“好。”", "然后走了。"]),
        ("停！！走吧。", ["停！！", "走吧。"]),
        ("Really?! Yes.", ["Really?!", "Yes."]),
    ],
)
def test_split_sentences_keeps_punctuation_runs_together(text, expected):
    assert tts_utils.split_sentences(text) == expected


def test_split_sentences_never_returns_punctuation_only_pieces():
    for text in ["好。！", "真的吗？！", "Wait . . . what?", "“好。”"]:
        assert all(any(ch.isalnum() for ch in s) for s in tts_utils.split_sentences(text))


def test_split_into_chunks_breaks_a_giant_sentence_at_commas_then_words():
    sentence = ", ".join(f"clause number {i} goes on" for i in range(40)) + "."
    chunks = tts_utils.split_into_chunks(sentence, 80)
    assert len(chunks) > 1
    assert all(tts_utils.spoken_weight(chunk) <= 120 for chunk in chunks)
    assert " ".join(chunks).split() == sentence.split()


def test_split_into_chunks_breaks_unspaced_text_by_characters():
    text = "字" * 200
    chunks = tts_utils.split_into_chunks(text, 60)
    assert "".join(chunks) == text
    assert all(tts_utils.spoken_weight(chunk) <= 60 for chunk in chunks)


def test_split_sentences_falls_back_without_sentencex(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fail_sentencex(name, *args, **kwargs):
        if name == "sentencex":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_sentencex)
    assert tts_utils.split_sentences("First one. Second one! Third?") == [
        "First one.",
        "Second one!",
        "Third?",
    ]


def test_join_with_pauses_inserts_silence_between_pieces():
    pieces = [np.ones(10, dtype=np.float32), np.ones(5, dtype=np.float32)]
    joined = tts_utils.join_with_pauses(pieces, sample_rate=100, pause_s=0.1)
    assert len(joined) == 25
    assert np.all(joined[10:20] == 0)


def test_join_with_pauses_empty():
    assert tts_utils.join_with_pauses([], 100, 0.3).size == 0


def test_quietest_point_finds_the_gap():
    sample_rate = 1000
    audio = np.full(3000, 0.5, dtype=np.float32)
    audio[2200:2300] = 0.0
    cut = tts_utils.quietest_point(audio, sample_rate, 1.5, 2.9)
    assert 2200 <= cut <= 2300


def test_quietest_point_window_past_the_end():
    audio = np.ones(100, dtype=np.float32)
    assert tts_utils.quietest_point(audio, 1000, 5.0, 6.0) == 100


@pytest.mark.parametrize("seed", [-1, "x", None])
def test_resolve_seed_picks_one_when_not_given(seed):
    value = tts_utils.resolve_seed(seed)
    assert 0 <= value < 2**31


def test_resolve_seed_keeps_a_given_seed():
    assert tts_utils.resolve_seed(42) == 42
    assert tts_utils.resolve_seed(42.0) == 42
