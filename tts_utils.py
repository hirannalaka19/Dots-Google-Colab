"""Text and audio helpers for the dots.tts Colab app.

Nothing in here touches the model or the GPU, so every function can be tested
on its own (see tests/test_tts_utils.py).
"""

from __future__ import annotations

import random
import re

import numpy as np

# ---------------------------------------------------------------------------
# Languages
#
# English display name -> the tag dots.tts understands. The list and codes
# mirror the official demo (apps/gradio/languages.py in studio-dots-ai/dots.tts);
# Chinese dialects use the model's own accent tags.
# ---------------------------------------------------------------------------
LANGUAGES: dict[str, str] = {
    "Chinese (Mandarin)": "ZH",
    "Cantonese": "口音:粤语",
    "Beijing Mandarin": "口音:北京官话",
    "Northeastern Mandarin": "口音:东北话",
    "Sichuanese": "口音:四川话",
    "Hokkien (Min Nan)": "口音:闽南话",
    "Wu (Shanghainese)": "口音:吴语",
    "English": "EN",
    "Spanish": "ES",
    "Hindi": "HI",
    "Arabic": "AR",
    "Bengali": "BN",
    "Portuguese": "PT",
    "Russian": "RU",
    "Japanese": "JA",
    "French": "FR",
    "German": "DE",
    "Korean": "KO",
    "Italian": "IT",
    "Turkish": "TR",
    "Vietnamese": "VI",
    "Indonesian": "ID",
    "Urdu": "UR",
    "Persian": "FA",
    "Tamil": "TA",
    "Telugu": "TE",
    "Filipino": "FIL",
    "Malay": "MS",
    "Punjabi": "PA",
    "Marathi": "MR",
    "Gujarati": "GU",
    "Malayalam": "ML",
    "Kannada": "KN",
    "Polish": "PL",
    "Ukrainian": "UK",
    "Dutch": "NL",
    "Thai": "TH",
    "Romanian": "RO",
    "Swahili": "SW",
    "Hebrew": "HE",
    "Czech": "CS",
    "Greek": "EL",
    "Hungarian": "HU",
    "Swedish": "SV",
    "Danish": "DA",
    "Finnish": "FI",
    "Norwegian (Bokmål)": "NB",
    "Slovak": "SK",
    "Slovenian": "SL",
    "Serbian": "SR",
    "Bosnian": "BS",
    "Croatian": "HR",
    "Bulgarian": "BG",
    "Macedonian": "MK",
    "Lithuanian": "LT",
    "Latvian": "LV",
    "Estonian": "ET",
    "Icelandic": "IS",
    "Irish": "GA",
    "Welsh": "CY",
    "Catalan": "CA",
    "Galician": "GL",
    "Occitan": "OC",
    "Asturian": "AST",
    "Nepali": "NE",
    "Sindhi": "SD",
    "Odia": "OR",
    "Assamese": "AS",
    "Pashto": "PS",
    "Burmese": "MY",
    "Khmer": "KM",
    "Lao": "LO",
    "Kazakh": "KK",
    "Uzbek": "UZ",
    "Kyrgyz": "KY",
    "Tajik": "TG",
    "Azerbaijani": "AZ",
    "Georgian": "KA",
    "Armenian": "HY",
    "Belarusian": "BE",
    "Luxembourgish": "LB",
    "Maltese": "MT",
    "Maori": "MI",
    "Afrikaans": "AF",
    "Zulu": "ZU",
    "Xhosa": "XH",
    "Yoruba": "YO",
    "Hausa": "HA",
    "Igbo": "IG",
    "Amharic": "AM",
    "Oromo": "OM",
    "Northern Sotho": "NSO",
    "Chichewa (Nyanja)": "NY",
    "Shona": "SN",
    "Somali": "SO",
    "Luganda": "LG",
    "Lingala": "LN",
    "Luo": "LUO",
    "Kamba": "KAM",
    "Umbundu": "UMB",
    "Fula": "FF",
    "Wolof": "WO",
    "Central Kurdish (Sorani)": "CKB",
    "Cebuano": "CEB",
    "Kabuverdianu": "KEA",
    "Mongolian": "MN",
    "Javanese": "JV",
}

# dots.tts reads "none" as "no language tag" and "auto_detect" as "guess it".
LANGUAGE_NOT_SET = "none"
LANGUAGE_AUTO_DETECT = "auto_detect"

# Whisper names a few languages differently from dots.tts.
_WHISPER_OVERRIDES = {"NB": "no", "FIL": "tl", "口音:粤语": "yue"}


def language_choices() -> list[tuple[str, str]]:
    """Dropdown entries as (label, value) pairs, special options first."""
    named = sorted(LANGUAGES.items(), key=lambda item: item[0].lower())
    return [
        ("Not set (the model reads the text)", LANGUAGE_NOT_SET),
        ("Auto-detect from the text", LANGUAGE_AUTO_DETECT),
        *named,
    ]


def whisper_code(language_tag: str | None) -> str | None:
    """Map a dots.tts language tag to a Whisper code; None means auto-detect."""
    if not language_tag or language_tag in (LANGUAGE_NOT_SET, LANGUAGE_AUTO_DETECT):
        return None
    if language_tag in _WHISPER_OVERRIDES:
        return _WHISPER_OVERRIDES[language_tag]
    if language_tag.startswith("口音:"):
        return "zh"
    code = language_tag.lower()
    return code if len(code) == 2 else None


# ---------------------------------------------------------------------------
# Splitting long text
#
# dots.tts caps one call at 500 audio patches (~80 s, reference included), and
# very long single calls drift. So long text is split on sentence boundaries,
# generated piece by piece, and joined with a short pause.
# ---------------------------------------------------------------------------

# Characters that take roughly three times longer to say than a Latin letter.
_WIDE_CHARS = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]"
)
# Latin stops need a following space; full-width stops don't, but a run of them
# ("？！") or a stop plus closing quote ("。”") stays with its sentence.
_SENTENCE_END = re.compile(
    r"(?<=[.!?…])\s+"
    r"|(?<=[。！？；])(?![。！？；」』”’）)])"
    r"|(?<=[。！？；][」』”’）)])"
)
_CLAUSE_END = re.compile(r"(?<=[,;:，、；：])\s*")
_HAS_WORD = re.compile(r"\w")
# A chunk may overshoot the budget by this much before we break mid-sentence.
_OVERSHOOT = 1.5


def spoken_weight(text: str) -> int:
    """Length of `text` in "Latin character" units, so CJK chunks stay short."""
    return len(text) + 2 * len(_WIDE_CHARS.findall(text))


def split_sentences(text: str, lang_code: str | None = None) -> list[str]:
    """Split into sentences with sentencex, falling back to punctuation rules."""
    try:
        from sentencex import segment

        parts = list(segment(lang_code or "en", text))
    except Exception:
        parts = _SENTENCE_END.split(text)

    sentences: list[str] = []
    for part in parts:
        # sentencex keeps Chinese/Japanese sentences glued together unless it
        # was told the language, so always split on full-width stops too.
        for piece in (p.strip() for p in _SENTENCE_END.split(part or "")):
            if not piece:
                continue
            if sentences and not _HAS_WORD.search(piece):
                sentences[-1] += piece  # stray punctuation belongs to the sentence before
            else:
                sentences.append(piece)
    return sentences


def _join(left: str, right: str) -> str:
    """Join two pieces of text; no space between CJK text, which uses none."""
    if not left:
        return right
    wide_boundary = _WIDE_CHARS.match(right[:1]) or not _HAS_WORD.search(right[:1])
    if wide_boundary and (_WIDE_CHARS.match(left[-1:]) or left[-1:] in "。！？；，、：」』”’）"):
        return left + right
    return f"{left} {right}"


def _group(parts: list[str], limit: int) -> list[str]:
    """Greedily join parts while staying within `limit` weight."""
    groups: list[str] = []
    current = ""
    for part in parts:
        candidate = _join(current, part)
        if current and spoken_weight(candidate) > limit:
            groups.append(current)
            current = part
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _split_by_characters(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for char in text:
        if current and spoken_weight(current + char) > limit:
            pieces.append(current)
            current = char
        else:
            current += char
    if current:
        pieces.append(current)
    return pieces


def _break_long_sentence(sentence: str, limit: int) -> list[str]:
    """Break one over-long sentence at commas, then words, then characters."""
    pieces: list[str] = []
    for clause in _group([c for c in _CLAUSE_END.split(sentence) if c], limit):
        if spoken_weight(clause) <= limit * _OVERSHOOT:
            pieces.append(clause)
            continue
        words = clause.split()
        if len(words) > 1:
            for group in _group(words, limit):
                if spoken_weight(group) <= limit * _OVERSHOOT:
                    pieces.append(group)
                else:  # a single enormous "word", e.g. a URL
                    pieces.extend(_split_by_characters(group, limit))
        else:  # scripts written without spaces
            pieces.extend(_split_by_characters(clause, limit))
    return pieces


def split_into_chunks(text: str, limit: int, lang_code: str | None = None) -> list[str]:
    """Group whole sentences into chunks of at most `limit` spoken weight."""
    text = (text or "").strip()
    if not text:
        return []
    limit = max(20, int(limit))

    chunks: list[str] = []
    for chunk in _group(split_sentences(text, lang_code), limit):
        if spoken_weight(chunk) <= limit * _OVERSHOOT:
            chunks.append(chunk)
        else:
            chunks.extend(_break_long_sentence(chunk, limit))
    return chunks


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def join_with_pauses(pieces: list[np.ndarray], sample_rate: int, pause_s: float) -> np.ndarray:
    """Concatenate mono float clips with `pause_s` seconds of silence between."""
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(sample_rate * max(0.0, float(pause_s))), dtype=np.float32)
    joined: list[np.ndarray] = []
    for index, piece in enumerate(pieces):
        if index:
            joined.append(gap)
        joined.append(np.asarray(piece, dtype=np.float32).reshape(-1))
    return np.concatenate(joined)


def quietest_point(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> int:
    """Sample index of the quietest 20 ms frame between start_s and end_s.

    Used to shorten a long reference clip without cutting a word in half.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    start = int(start_s * sample_rate)
    end = min(int(end_s * sample_rate), len(audio))
    frame = max(1, int(0.02 * sample_rate))
    if end - start < frame:
        return min(end, len(audio))

    best_index, best_energy = start, float("inf")
    for index in range(start, end - frame + 1, frame):
        energy = float(np.mean(audio[index:index + frame] ** 2))
        if energy < best_energy:
            best_index, best_energy = index, energy
    return best_index + frame // 2


def resolve_seed(seed) -> int:
    """A seed below zero (or garbage) means "pick one"; return the one used."""
    try:
        value = int(seed)
    except (TypeError, ValueError):
        value = -1
    return random.randint(0, 2**31 - 1) if value < 0 else value
