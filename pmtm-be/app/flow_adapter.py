from typing import Any
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
_ai_app_dir = _project_root / "pmtm-ai"
if _ai_app_dir.exists() and str(_ai_app_dir) not in sys.path:
    sys.path.insert(0, str(_ai_app_dir))

try:
    # pyrefly: ignore [missing-import]
    from app.lyric_prompts import (
        ALLOWED_LYRIC_CHARS_REGEX,
        clean_unsupported_characters,
        find_unsupported_characters,
    )
except ImportError:
    ALLOWED_LYRIC_CHARS_REGEX = r"[가-힣a-zA-Z0-9\s.,!?~'\"()\[\]{}:;·…-]"

    def find_unsupported_characters(text: str) -> str:
        return re.sub(ALLOWED_LYRIC_CHARS_REGEX, "", text)

    def clean_unsupported_characters(text: str) -> str:
        return "".join(re.findall(ALLOWED_LYRIC_CHARS_REGEX, text))

# pyrefly: ignore [missing-import]
from g2pk2 import G2p
# pyrefly: ignore [missing-import]
import pronouncing
# pyrefly: ignore [missing-import]
from kiwipiepy import Kiwi

g2p = G2p()
_kiwi_instance = None


def get_kiwi() -> Kiwi:
    global _kiwi_instance
    if _kiwi_instance is None:
        _kiwi_instance = Kiwi()
    return _kiwi_instance


TARGET_BARS = 8
MAX_SYLLABLES_PER_BAR = 28
F0_TIMESTEP_SEC = 0.01
VOICED_ENERGY_DB = -26.0
SILENCE_ENERGY_DB = -80.0
VOICED_BREATHINESS_DB = -60.0
SILENCE_BREATHINESS_DB = -80.0

CONTENT_POS_PREFIXES = ("NN", "VV", "VA", "MAG", "XR", "SL", "SN")
FUNCTION_POS_PREFIXES = ("J", "E", "X", "SP", "SS", "SF", "SE", "SO", "SW")

CHOSEONG = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)
JUNGSEONG = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
    "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
)
JONGSEONG = (
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ",
    "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ",
    "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)

# Korean phoneme mapping used by the POTG DiffSinger voicebank.
ONSET_PHONEMES = {
    "ㄱ": "g", "ㄲ": "kk", "ㄴ": "n", "ㄷ": "d", "ㄸ": "tt", "ㄹ": "rx",
    "ㅁ": "m", "ㅂ": "b", "ㅃ": "pp", "ㅅ": "sc", "ㅆ": "s", "ㅇ": "",
    "ㅈ": "jh", "ㅉ": "jj", "ㅊ": "ch", "ㅋ": "k", "ㅌ": "t", "ㅍ": "p", "ㅎ": "hh",
}
VOWEL_PHONEMES = {
    "ㅏ": "a", "ㅐ": "e", "ㅑ": "ia", "ㅒ": "ie", "ㅓ": "eo", "ㅔ": "e",
    "ㅕ": "ieo", "ㅖ": "ie", "ㅗ": "o", "ㅘ": "oa", "ㅙ": "oe", "ㅚ": "oe",
    "ㅛ": "io", "ㅜ": "u", "ㅝ": "uo", "ㅞ": "oe", "ㅟ": "ui", "ㅠ": "iu",
    "ㅡ": "eu", "ㅢ": "i", "ㅣ": "i",
}
CODA_PHONEMES = {
    "": "", "ㄱ": "kcl", "ㄲ": "kcl", "ㄳ": "kcl", "ㄴ": "n", "ㄵ": "n",
    "ㄶ": "n", "ㄷ": "tcl", "ㄹ": "l", "ㄺ": "kcl", "ㄻ": "m", "ㄼ": "pcl",
    "ㄽ": "l", "ㄾ": "l", "ㄿ": "pcl", "ㅀ": "l", "ㅁ": "m", "ㅂ": "pcl",
    "ㅄ": "pcl", "ㅅ": "tcl", "ㅆ": "tcl", "ㅇ": "ng", "ㅈ": "tcl",
    "ㅊ": "tcl", "ㅋ": "kcl", "ㅌ": "tcl", "ㅍ": "pcl", "ㅎ": "tcl",
}

PALATAL_VOWELS = {"ㅑ", "ㅒ", "ㅕ", "ㅖ", "ㅛ", "ㅠ", "ㅣ"}
SILENT_ONSET_COMPOUNDS = {
    "ㅘ": ("w", "a"),
    "ㅙ": ("w", "e"),
    "ㅚ": ("w", "e"),
    "ㅝ": ("w", "o"),
    "ㅞ": ("w", "e"),
    "ㅟ": ("w", "i"),
    "ㅢ": ("", "ui"),
}

ARPABET_VOWELS = {
    "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey",
    "ih", "iy", "ow", "oy", "uh", "uw"
}
ARPABET_TO_POTG = {
    "aa": "a", "ae": "e", "ah": "eo", "ao": "o", "aw": "oa",
    "ax": "eo", "ay": "a", "b": "b", "ch": "ch", "d": "d",
    "dh": "d", "dx": "rx", "eh": "e", "er": "eo", "ey": "e",
    "f": "p", "g": "g", "hh": "hh", "ih": "i", "iy": "i",
    "jh": "jh", "k": "k", "l": "l", "m": "m", "n": "n",
    "ng": "ng", "ow": "o", "oy": "oe", "p": "p", "r": "rx",
    "s": "sc", "sh": "sh", "t": "t", "th": "sc", "uh": "u",
    "uw": "u", "v": "b", "w": "w", "y": "i", "z": "sc", "zh": "jh"
}
LETTER_TO_POTG = {
    "a": "a", "b": "b", "c": "k", "d": "d", "e": "e", "f": "p", "g": "g",
    "h": "hh", "i": "i", "j": "jh", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "o", "p": "p", "q": "k", "r": "rx", "s": "sc", "t": "t", "u": "u",
    "v": "b", "w": "w", "x": "k", "y": "i", "z": "sc"
}


def _is_hangul(char: str) -> bool:
    return len(char) == 1 and "가" <= char <= "힣"


def _hangul_chars(text: str) -> list[tuple[int, str]]:
    return [(index, char) for index, char in enumerate(text) if _is_hangul(char)]


def _align_sequences(source: list[str], spoken: list[str]) -> list[tuple[str, int | None, int | None]]:
    rows, columns = len(source) + 1, len(spoken) + 1
    costs = [[0] * columns for _ in range(rows)]
    moves = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0], moves[row][0] = row, "delete"
    for column in range(1, columns):
        costs[0][column], moves[0][column] = column, "insert"

    for row in range(1, rows):
        for column in range(1, columns):
            substitution = costs[row - 1][column - 1] + (
                0 if source[row - 1] == spoken[column - 1] else 1
            )
            deletion = costs[row - 1][column] + 1
            insertion = costs[row][column - 1] + 1
            candidates = (
                (substitution, "match" if source[row - 1] == spoken[column - 1] else "substitute"),
                (deletion, "delete"),
                (insertion, "insert"),
            )
            costs[row][column], moves[row][column] = min(candidates, key=lambda item: item[0])

    result = []
    row, column = len(source), len(spoken)
    while row or column:
        move = moves[row][column]
        if move in ("match", "substitute"):
            result.append((move, row - 1, column - 1))
            row -= 1
            column -= 1
        elif move == "delete":
            result.append((move, row - 1, None))
            row -= 1
        else:
            result.append(("insert", None, column - 1))
            column -= 1
    return list(reversed(result))


@dataclass(frozen=True)
class SyllableLinguisticInfo:
    text: str
    stress: float
    is_content: bool
    is_word_start: bool
    midi_note: int
    is_liaison_accent_target: bool = False


def _analyze_korean_line_linguistics(text: str, g2p_text: str) -> list[SyllableLinguisticInfo]:
    kiwi = get_kiwi()
    tokens = kiwi.tokenize(text)
    morphemes = []
    for index, token in enumerate(tokens):
        morphemes.append({
            "id": index,
            "form": token.form,
            "tag": token.tag,
            "start": token.start,
            "end": token.start + token.len,
            "isContent": token.tag.startswith(CONTENT_POS_PREFIXES),
        })

    eojeols = []
    for index, match in enumerate(re.finditer(r"\S+", text)):
        eojeols.append({
            "id": index,
            "surface": match.group(0),
            "start": match.start(),
            "end": match.end(),
        })

    source_hangul = _hangul_chars(text)
    sources = []
    for source_idx, (char_idx, char) in enumerate(source_hangul):
        morpheme = next((m for m in morphemes if m["start"] <= char_idx < m["end"]), None)
        eojeol = next((e for e in eojeols if e["start"] <= char_idx < e["end"]), None)
        tag = morpheme["tag"] if morpheme else None
        if tag and tag.startswith(CONTENT_POS_PREFIXES):
            stress = 1.0
            is_content = True
        elif tag and tag.startswith(FUNCTION_POS_PREFIXES):
            stress = 0.65
            is_content = False
        else:
            stress = 0.85
            is_content = False

        is_word_start = bool(eojeol and char_idx == eojeol["start"])
        if is_word_start:
            stress += 0.1

        sources.append({
            "sourceIndex": source_idx,
            "charIndex": char_idx,
            "text": char,
            "stress": stress,
            "isContent": is_content,
            "isWordStart": is_word_start,
            "isFunction": bool(tag and tag.startswith(FUNCTION_POS_PREFIXES)),
            "morphemeEnd": morpheme["end"] if morpheme else char_idx + 1,
            "eojeolId": eojeol["id"] if eojeol else None,
        })

    spoken_hangul = [c for _, c in _hangul_chars(g2p_text)]
    if not sources or not spoken_hangul:
        return [SyllableLinguisticInfo(c, 1.0, True, True, 60) for c in spoken_hangul]

    operations = _align_sequences([s["text"] for s in sources], spoken_hangul)
    source_to_spoken = {
        source_index: spoken_index
        for _, source_index, spoken_index in operations
        if source_index is not None and spoken_index is not None
    }
    liaison_target_indices = set()
    for source_index, source in enumerate(sources[:-1]):
        following = sources[source_index + 1]
        source_spoken_index = source_to_spoken.get(source_index)
        target_spoken_index = source_to_spoken.get(source_index + 1)
        has_coda = JONGSEONG[(ord(source["text"]) - ord("가")) % 28] != ""
        spoken_source_has_coda = (
            source_spoken_index is not None
            and JONGSEONG[(ord(spoken_hangul[source_spoken_index]) - ord("가")) % 28] != ""
        )
        is_liaison = (
            source["isContent"]
            and source["isWordStart"]
            and following["isFunction"]
            and source["morphemeEnd"] == following["charIndex"]
            and source["eojeolId"] == following["eojeolId"]
            and has_coda
            and source_spoken_index is not None
            and target_spoken_index is not None
            and not spoken_source_has_coda
            and spoken_hangul[source_spoken_index] != source["text"]
            and spoken_hangul[target_spoken_index] != following["text"]
        )
        if is_liaison:
            liaison_target_indices.add(target_spoken_index)
    result: list[SyllableLinguisticInfo] = []

    for _, s_idx, p_idx in operations:
        if p_idx is None:
            continue
        source = sources[s_idx] if s_idx is not None else sources[0]
        char = spoken_hangul[p_idx]
        stress = source["stress"]
        is_content = source["isContent"]
        is_word_start = source["isWordStart"]

        # MIDI note pitch assignment (Baseline C4 for normal/function words)
        # Normal & Function words: C4 (60), Content word start: D4 (62), High Stress: D#4 (63)
        if stress >= 1.2:
            midi_note = 63
        elif is_content and is_word_start:
            midi_note = 62
        else:
            midi_note = 60

        result.append(SyllableLinguisticInfo(
            text=char,
            stress=stress,
            is_content=is_content,
            is_word_start=is_word_start,
            midi_note=midi_note,
            is_liaison_accent_target=p_idx in liaison_target_indices,
        ))

    return result


def _english_word_to_syllables(word: str) -> list[list[str]]:
    clean = re.sub(r"[^a-zA-Z']", "", word).lower()
    if not clean:
        return []
    lookup = re.sub(r"'", "", clean)
    phones_list = pronouncing.phones_for_word(clean) or pronouncing.phones_for_word(lookup)
    if phones_list:
        raw_phones = [re.sub(r"\d+", "", p).lower() for p in phones_list[0].split()]
        # POTG has no single phoneme for the English /eɪ/ diphthong.  Keep its
        # closing glide as a separate /i/ syllable so "someday" is sung
        # "썸-데-이", rather than ending at "썸-데".
        raw_phones = [
            expanded
            for phone in raw_phones
            for expanded in (("eh", "iy") if phone == "ey" else (phone,))
        ]
        potg_phones = [ARPABET_TO_POTG.get(p, p) for p in raw_phones]
        v_indices = [i for i, p in enumerate(raw_phones) if p in ARPABET_VOWELS]
        if len(v_indices) <= 1:
            return [potg_phones]
        syllables = []
        start = 0
        for k in range(len(v_indices) - 1):
            v_curr = v_indices[k]
            v_next = v_indices[k + 1]
            consonants_between = v_next - v_curr - 1
            split_at = v_curr + (1 if consonants_between <= 1 else 2)
            syllables.append(potg_phones[start:split_at])
            start = split_at
        syllables.append(potg_phones[start:])
        return syllables

    letters = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not letters:
        return []
    return [[LETTER_TO_POTG.get(char, "a")] for char in letters]


@dataclass(frozen=True)
class WordChunk:
    syllables: list[list[str]]
    punctuation: str | None = None
    linguistics: list[SyllableLinguisticInfo] | None = None


KOREAN_NUM_UNITS = ["", "십", "백", "천"]
KOREAN_NUM_BIG_UNITS = ["", "만", "억", "조", "경"]
KOREAN_NUM_DIGITS = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]


def int_to_korean(num: int) -> str:
    if num == 0:
        return "영"
    str_num = str(num)
    groups = []
    while str_num:
        groups.append(str_num[-4:])
        str_num = str_num[:-4]

    group_parts = []
    for g_idx, group in enumerate(groups):
        part = ""
        g_len = len(group)
        for d_idx, digit_char in enumerate(group):
            d = int(digit_char)
            pos = g_len - 1 - d_idx
            if d != 0:
                if d == 1 and pos > 0:
                    part += KOREAN_NUM_UNITS[pos]
                else:
                    part += KOREAN_NUM_DIGITS[d] + KOREAN_NUM_UNITS[pos]
        if part:
            if part == "일" and g_idx > 0:
                part = KOREAN_NUM_BIG_UNITS[g_idx]
            else:
                part += KOREAN_NUM_BIG_UNITS[g_idx]
        group_parts.append(part)

    return "".join(reversed(group_parts))


def convert_numbers_to_hangul(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return int_to_korean(int(match.group(0)))
    return re.sub(r"\d+", repl, text)


def _extract_word_syllables_and_text(text: str) -> tuple[list[WordChunk], str]:
    text = convert_numbers_to_hangul(text)
    unsupported = find_unsupported_characters(text)
    if unsupported:
        raise ValueError(f"현재 SVS 테스트는 한글과 영문 가사만 지원합니다. 지원하지 않는 문자: {unsupported[:20]}")
    raw_words = text.strip().split()
    word_chunks: list[WordChunk] = []
    g2p_parts: list[str] = []

    for raw_word in raw_words:
        tokens = re.findall(r"[가-힣]+|[a-zA-Z']+", raw_word)
        for token in tokens:
            if re.match(r"^[가-힣]+$", token):
                g2p_parts.append(g2p(token))
            else:
                g2p_parts.append(token)

    g2p_line = " ".join(g2p_parts)
    line_linguistics = _analyze_korean_line_linguistics(text, g2p_line)
    ling_cursor = 0

    for raw_word in raw_words:
        punct_match = re.search(r"([?,!.])+$", raw_word)
        punct = punct_match.group(1) if punct_match else None

        tokens = re.findall(r"[가-힣]+|[a-zA-Z']+", raw_word)
        current_word_syllables: list[list[str]] = []
        current_word_ling: list[SyllableLinguisticInfo] = []

        for token in tokens:
            if re.match(r"^[가-힣]+$", token):
                g2p_token = g2p(token)
                syllables = re.findall(r"[가-힣]", g2p_token)
                for syl in syllables:
                    current_word_syllables.append(_syllable_to_phonemes(syl))
                    if ling_cursor < len(line_linguistics):
                        current_word_ling.append(line_linguistics[ling_cursor])
                        ling_cursor += 1
                    else:
                        current_word_ling.append(SyllableLinguisticInfo(syl, 1.0, True, False, 60))
            else:
                e_syls = _english_word_to_syllables(token)
                current_word_syllables.extend(e_syls)
                for e_idx in range(len(e_syls)):
                    current_word_ling.append(SyllableLinguisticInfo(token, 1.0, True, e_idx == 0, 60))

        if current_word_syllables:
            word_chunks.append(WordChunk(
                syllables=current_word_syllables,
                punctuation=punct,
                linguistics=current_word_ling,
            ))

    return word_chunks, g2p_line


@dataclass(frozen=True)
class BeatMap:
    bpm: int
    beatsPerBar: int
    barCount: int
    firstBarStartSec: float
    barDurationSec: float
    barStartTimes: list[float]
    beatTimes: list[float]
    slotDurations: list[float] | None = None
    snareTimes: list[float] | None = None
    gridTimes: list[float] | None = None


@dataclass(frozen=True)
class FlowPhoneme:
    symbol: str
    durationSec: float


@dataclass(frozen=True)
class FlowBar:
    barIndex: int
    text: str
    startSec: float
    endSec: float
    template: str
    syllableCount: int
    phonemes: list[FlowPhoneme]
    phNum: list[int] | None = None
    noteSeq: list[str] | None = None
    noteDur: list[float] | None = None


@dataclass(frozen=True)
class FlowPlan:
    version: int
    voicebank: str
    beatMap: BeatMap
    bars: list[FlowBar]
    genre: str = "boom_bap"


def parse_eight_bar_lyrics(lyrics: str) -> list[str]:
    lines = [
        line.strip()
        for line in lyrics.splitlines()
        if line.strip() and not line.strip().lower().startswith("[verse")
    ]
    if len(lines) not in (8, 16):
        raise ValueError(f"가이드는 정확히 8줄 또는 16줄의 가사가 필요합니다. 현재 {len(lines)}줄입니다.")
    return lines


def build_beat_map(
    bpm: int,
    first_bar_start_sec: float,
    bar_count: int = 8,
    beat_analysis: dict[str, Any] | None = None,
) -> BeatMap:
    if bpm < 40 or bpm > 220:
        raise ValueError("BPM은 40부터 220 사이여야 합니다.")
    if not math.isfinite(first_bar_start_sec) or first_bar_start_sec < 0:
        raise ValueError("첫 마디 시작 시각은 0 이상의 유한한 값이어야 합니다.")

    actual_bpm = bpm / 2.0 if bpm >= 115.0 else float(bpm)
    beat_duration = 60.0 / actual_bpm
    beats_per_unit = 4
    bar_duration = beat_duration * beats_per_unit

    slot_durations: list[float] | None = None
    snare_times: list[float] | None = None
    grid_times: list[float] | None = None

    if beat_analysis:
        analysis_bpm = beat_analysis.get("bpm", {}).get("fixed_integer")
        half_time_multiplier = 2 if analysis_bpm and 115 <= analysis_bpm <= 220 else 1
        if analysis_bpm and 40 <= analysis_bpm <= 220:
            actual_bpm = float(analysis_bpm / half_time_multiplier)
            beat_duration = 60.0 / actual_bpm
            bar_duration = beat_duration * beats_per_unit

        absolute_grid = beat_analysis.get("absolute_grid", [])
        if absolute_grid:
            grid_times = [float(slot["time_sec"]) for slot in absolute_grid][::half_time_multiplier]
            if len(grid_times) >= 2:
                slot_durations = [grid_times[i + 1] - grid_times[i] for i in range(len(grid_times) - 1)]
                slot_durations.append(slot_durations[-1] if slot_durations else beat_duration / 4.0)

            extracted_bar_starts = [
                slot["time_sec"] for slot in absolute_grid
                if slot.get("beat_in_bar") == 1 and slot.get("subdivision") == 0
            ][::half_time_multiplier]
            extracted_beat_times = [
                slot["time_sec"] for slot in absolute_grid
                if slot.get("subdivision") == 0
            ][::half_time_multiplier]

            grid_shift = (
                first_bar_start_sec - float(extracted_bar_starts[0])
                if extracted_bar_starts
                else 0.0
            )
            extracted_bar_starts = [float(value) + grid_shift for value in extracted_bar_starts]
            extracted_beat_times = [float(value) + grid_shift for value in extracted_beat_times]

            if len(extracted_bar_starts) >= bar_count:
                bar_starts = extracted_bar_starts[:bar_count]
            else:
                bar_starts = [first_bar_start_sec + index * bar_duration for index in range(bar_count)]

            if len(extracted_beat_times) >= bar_count * beats_per_unit:
                beat_times = extracted_beat_times[:bar_count * beats_per_unit]
            else:
                beat_times = [first_bar_start_sec + index * beat_duration for index in range(bar_count * beats_per_unit)]
        else:
            bar_starts = [first_bar_start_sec + index * bar_duration for index in range(bar_count)]
            beat_times = [first_bar_start_sec + index * beat_duration for index in range(bar_count * beats_per_unit)]

        snare_events = beat_analysis.get("snare_detection", {}).get("events", [])
        if snare_events:
            snare_times = [float(evt["original_time"]) for evt in snare_events if "original_time" in evt]
    else:
        bar_starts = [first_bar_start_sec + index * bar_duration for index in range(bar_count)]
        beat_times = [first_bar_start_sec + index * beat_duration for index in range(bar_count * beats_per_unit)]

    return BeatMap(
        bpm=int(round(actual_bpm)),
        beatsPerBar=beats_per_unit,
        barCount=bar_count,
        firstBarStartSec=round(first_bar_start_sec, 6),
        barDurationSec=round(bar_duration, 6),
        barStartTimes=_rounded(bar_starts),
        beatTimes=_rounded(beat_times),
        slotDurations=_rounded(slot_durations) if slot_durations else None,
        snareTimes=_rounded(snare_times) if snare_times else None,
        gridTimes=_rounded(grid_times) if grid_times else None,
    )


def build_flow_plan(
    lyrics: str,
    bpm: int,
    first_bar_start_sec: float,
    voicebank: str,
    *,
    base_f0_hz: float,
    genre: str = "boom_bap",
    beat_analysis: dict[str, Any] | None = None,
) -> FlowPlan:
    lines = parse_eight_bar_lyrics(lyrics)
    if len(lines) == 16:
        bar_lines = [f"{lines[i]} {lines[i+1]}" for i in range(0, 16, 2)]
    else:
        bar_lines = lines

    beat_map = build_beat_map(bpm, first_bar_start_sec, bar_count=len(bar_lines), beat_analysis=beat_analysis)
    bars: list[FlowBar] = []
    for index, line_text in enumerate(bar_lines):
        word_chunks, g2p_line = _extract_word_syllables_and_text(line_text)
        total_syllable_count = sum(len(w.syllables) for w in word_chunks)
        if total_syllable_count == 0:
            raise ValueError(f"{index + 1}마디에 합성 가능한 음절이 없습니다.")
        if total_syllable_count > MAX_SYLLABLES_PER_BAR:
            raise ValueError(
                f"{index + 1}마디가 너무 조밀합니다. 한 마디는 {MAX_SYLLABLES_PER_BAR}음절 이하로 수정해주세요."
            )

        phonemes, ph_num, durations, template, note_seq, note_dur = _allocate_word_hierarchical_durations(
            word_chunks,
            beat_map.barDurationSec,
            index,
            genre=genre,
            bpm=beat_map.bpm,
        )
        start = beat_map.barStartTimes[index]
        bars.append(
            FlowBar(
                barIndex=index + 1,
                text=g2p_line,
                startSec=start,
                endSec=round(start + beat_map.barDurationSec, 6),
                template=template,
                syllableCount=total_syllable_count,
                phonemes=[
                    FlowPhoneme(symbol=symbol, durationSec=duration)
                    for symbol, duration in zip(phonemes, durations)
                ],
                phNum=ph_num,
                noteSeq=note_seq,
                noteDur=note_dur,
            )
        )

    plan = FlowPlan(version=1, voicebank=voicebank, beatMap=beat_map, bars=bars, genre=genre)
    _validate_plan_duration(plan)
    return plan


def write_flow_plan(plan: FlowPlan, path: Path) -> None:
    path.write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2), encoding="utf-8")


def write_diffsinger_ds(plan: FlowPlan, path: Path, *, base_f0_hz: float) -> None:
    sections: list[dict[str, str]] = []
    for bar in plan.bars:
        symbols = [phoneme.symbol for phoneme in bar.phonemes]
        durations = [phoneme.durationSec for phoneme in bar.phonemes]
        if bar.phNum is not None:
            ph_num = bar.phNum
        else:
            word_chunks, _ = _extract_word_syllables_and_text(bar.text)
            flat_syllables = [syl for w in word_chunks for syl in w.syllables]
            ph_num = [1] + [len(syl_p) for syl_p in flat_syllables] + [1]

        if sum(ph_num) != len(symbols):
            raise RuntimeError(f"{bar.barIndex}마디 음절과 음소 매핑이 맞지 않습니다.")

        if bar.noteSeq is not None and bar.noteDur is not None:
            note_seq_list = bar.noteSeq
            note_dur_list = bar.noteDur
            note_slur_list = [0] * len(note_seq_list)
        else:
            note_seq_list = []
            note_dur_list = []
            note_slur_list = []
            cursor = 0
            for count in ph_num:
                group_symbols = symbols[cursor : cursor + count]
                group_durs = durations[cursor : cursor + count]
                group_dur_sum = sum(group_durs)

                if group_symbols == ["SP"]:
                    note_seq_list.append("rest")
                    note_dur_list.append(group_dur_sum)
                    note_slur_list.append(0)
                else:
                    note_seq_list.append("C4")
                    note_dur_list.append(group_dur_sum)
                    note_slur_list.append(0)

                cursor += count

        f0_values = _build_f0_curve(symbols, durations, base_f0_hz, bar.barIndex, note_seq_list, ph_num)
        zero_curve = " ".join("0" for _ in f0_values)
        energy_curve = " ".join(
            f"{VOICED_ENERGY_DB:.1f}" if value > 0 else f"{SILENCE_ENERGY_DB:.1f}"
            for value in f0_values
        )
        breathiness_curve = " ".join(
            f"{VOICED_BREATHINESS_DB:.1f}" if value > 0 else f"{SILENCE_BREATHINESS_DB:.1f}"
            for value in f0_values
        )
        sections.append(
            {
                "offset": f"{bar.startSec:.6f}",
                "text": bar.text,
                "ph_seq": " ".join(symbols),
                "ph_dur": " ".join(f"{value:.6f}" for value in durations),
                "ph_num": " ".join(str(value) for value in ph_num),
                "note_seq": " ".join(note_seq_list),
                "note_dur": " ".join(f"{value:.6f}" for value in note_dur_list),
                "note_slur": " ".join(str(value) for value in note_slur_list),
                "f0_seq": " ".join(f"{value:.3f}" for value in f0_values),
                "f0_timestep": f"{F0_TIMESTEP_SEC:.3f}",
                "breathiness": breathiness_curve,
                "breathiness_timestep": f"{F0_TIMESTEP_SEC:.3f}",
                "voicing": zero_curve,
                "voicing_timestep": f"{F0_TIMESTEP_SEC:.3f}",
                "tension": zero_curve,
                "tension_timestep": f"{F0_TIMESTEP_SEC:.3f}",
                "energy": energy_curve,
                "energy_timestep": f"{F0_TIMESTEP_SEC:.3f}",
            }
        )
    path.write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")
def _syllable_to_phonemes(syllable: str) -> list[str]:
    code = ord(syllable) - 0xAC00
    onset = CHOSEONG[code // 588]
    vowel = JUNGSEONG[(code % 588) // 28]
    coda = JONGSEONG[code % 28]

    onset_phone = ONSET_PHONEMES[onset]
    vowel_phone = VOWEL_PHONEMES[vowel]
    coda_phone = CODA_PHONEMES[coda]
    if onset == "ㅇ" and vowel in SILENT_ONSET_COMPOUNDS:
        onset_phone, vowel_phone = SILENT_ONSET_COMPOUNDS[vowel]
    elif onset == "ㅅ" and vowel in PALATAL_VOWELS:
        onset_phone = "sh"
    elif onset == "ㅆ" and vowel in PALATAL_VOWELS:
        onset_phone = "sy"

    return [phone for phone in (onset_phone, vowel_phone, coda_phone) if phone]


DEFAULT_MIN_DUR_SEC = 0.11
ABSOLUTE_MIN_DUR_SEC = 0.08
MIN_VOWEL_DUR_SEC = 0.085
MIN_PLOSIVE_DUR_SEC = 0.030
MIN_FRICATIVE_DUR_SEC = 0.040
MIN_NASAL_DUR_SEC = 0.035
MAX_SYLLABLE_DUR_SEC = 0.60

PLOSIVE_SYMBOLS = {"g", "kk", "d", "tt", "b", "pp", "k", "t", "p", "kcl", "tcl", "pcl", "cl", "K", "P", "T"}
FRICATIVE_SYMBOLS = {"sc", "s", "sh", "sy", "hh", "jh", "ch", "jj"}
NASAL_LIQUID_SYMBOLS = {"n", "m", "ng", "l", "rx", "N", "M"}
VOWEL_SYMBOLS = {
    "a", "e", "eo", "eu", "i", "o", "u", "ia", "ie", "ieo", "io", "iu", "oa", "oe", "uo", "ui",
    "a1", "a2", "a3", "a4", "e1", "e2", "e3", "e4", "eo1", "eo2", "eo3", "eo4", "eu1", "eu2", "eu3", "eu4",
    "i1", "i2", "i3", "i4", "o1", "o2", "o3", "o4", "u1", "u2", "u3", "u4",
    "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw",
}
CODA_SYMBOLS = {"ng", "l", "n", "m", "kcl", "tcl", "pcl", "cl", "K", "N", "M", "P"}


def _phoneme_weight(symbol: str) -> float:
    if symbol in VOWEL_SYMBOLS:
        return 0.68
    if symbol in CODA_SYMBOLS:
        return 0.14
    return 0.18


def _phoneme_weight_v2(symbol: str, bpm: float = 120.0, stress: float = 1.0) -> float:
    """Option A: Dynamic phoneme category weighting with BPM 115 threshold scaling & stress modulation."""
    is_fast_tempo = bpm >= 115.0
    if symbol in VOWEL_SYMBOLS:
        base = 0.62 if is_fast_tempo else 0.68
        if stress >= 1.15:
            base *= 1.15
        return base
    elif symbol in FRICATIVE_SYMBOLS:
        return 0.22 if is_fast_tempo else 0.18
    elif symbol in PLOSIVE_SYMBOLS:
        return 0.16 if is_fast_tempo else 0.14
    elif symbol in NASAL_LIQUID_SYMBOLS:
        return 0.15
    return 0.18


def _allocate_syllable_grid_slots(
    weights: list[float],
    available_slots: int,
    priority_indices: list[int] | None = None,
    extra_eligible_indices: list[int] | None = None,
) -> list[int]:
    """Allocate one slot per syllable before prioritizing content-word attacks."""
    if not weights:
        return []
    if len(weights) > available_slots:
        return [1] * len(weights)

    durations = [1] * len(weights)
    remaining_slots = available_slots - len(weights)
    for index in priority_indices or []:
        if remaining_slots == 0:
            break
        if 0 <= index < len(durations) and durations[index] == 1:
            durations[index] += 1
            remaining_slots -= 1

    if remaining_slots == 0:
        return durations

    eligible_indices = set(extra_eligible_indices or range(len(weights)))
    allocation_weights = [weight if index in eligible_indices else 0.0 for index, weight in enumerate(weights)]
    total = sum(allocation_weights)
    if total <= 0:
        allocation_weights = weights
        total = sum(allocation_weights)
    raw = [remaining_slots * weight / total for weight in allocation_weights]
    extra_slots = [math.floor(value) for value in raw]
    durations = [duration + extra for duration, extra in zip(durations, extra_slots)]
    while sum(durations) < available_slots:
        target = max(
            range(len(weights)),
            key=lambda i: (raw[i] - extra_slots[i], allocation_weights[i]),
        )
        durations[target] += 1
        extra_slots[target] += 1
    return durations


def _get_word_syllable_weights(syllable_count: int) -> list[float]:
    return [1.0] * max(1, syllable_count)


def _midi_note_to_name(midi_note: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return names[midi_note % 12] + str(midi_note // 12 - 1)


def _midi_note_to_hz(midi_note: int, base_f0_hz: float) -> float:
    # Relative Pitch Shift from C4 (MIDI 60)
    semitones = midi_note - 60
    return base_f0_hz * (2.0 ** (semitones / 12.0))


def _allocate_word_hierarchical_durations(
    word_chunks: list[WordChunk],
    bar_duration_sec: float,
    bar_index: int,
    genre: str = "boom_bap",
    bpm: float = 120.0,
) -> tuple[list[str], list[int], list[float], str, list[str], list[float]]:
    total_syllable_count = sum(len(w.syllables) for w in word_chunks)
    word_count = len(word_chunks)

    # Use the full 16th-note grid. Rests are cadence choices, not fixed
    # reservations based solely on syllable count.
    slots_per_bar = 16
    slot_duration = bar_duration_sec / float(slots_per_bar)

    # Collect flat list of syllables and linguistic stress
    syl_list: list[dict[str, Any]] = []
    current_syl_index = 0
    for w_idx, chunk in enumerate(word_chunks):
        syl_weights = _get_word_syllable_weights(len(chunk.syllables))
        for s_idx, syl_phones in enumerate(chunk.syllables):
            ling = chunk.linguistics[s_idx] if chunk.linguistics and s_idx < len(chunk.linguistics) else None
            ling_stress = ling.stress if ling else 1.0
            midi_note = ling.midi_note if ling else 60
            is_liaison_target = bool(ling and ling.is_liaison_accent_target)
            is_liaison_source = bool(
                chunk.linguistics
                and s_idx + 1 < len(chunk.linguistics)
                and chunk.linguistics[s_idx + 1].is_liaison_accent_target
            )

            combined_weight = syl_weights[s_idx] * ling_stress

            syl_list.append({
                "syl_idx": current_syl_index,
                "phones": syl_phones,
                "weight": combined_weight,
                "stress": ling_stress,
                "midi_note": midi_note,
                "is_liaison_source": is_liaison_source,
                "is_priority_attack": bool(
                    is_liaison_target
                    or (ling and ling.is_content and ling.is_word_start and not is_liaison_source)
                ),
                "punct": chunk.punctuation if s_idx == len(chunk.syllables) - 1 else None,
            })
            current_syl_index += 1

    # Allocate every syllable one slot first. A single cadence tail slot is
    # used only after every content-word attack can receive one extra slot.
    syl_weights_list = [float(item["weight"]) for item in syl_list]
    priority_indices = [
        index for index, item in enumerate(syl_list) if item["is_priority_attack"]
    ]
    remaining_slots = max(0, slots_per_bar - total_syllable_count)
    tail_slots = 1 if remaining_slots > len(priority_indices) else 0
    available_slots = max(1, slots_per_bar - tail_slots)
    allocated_slots = _allocate_syllable_grid_slots(
        syl_weights_list,
        available_slots,
        priority_indices,
        [index for index, item in enumerate(syl_list) if not item["is_liaison_source"]],
    )

    for index, item in enumerate(syl_list[:-1]):
        if item["is_liaison_source"] and allocated_slots[index + 1] >= 2:
            item["midi_note"] = 60
            syl_list[index + 1]["midi_note"] = 62

    lead_duration = 0.04
    punctuation_duration = sum(
        0.04 if item["punct"] == "," else 0.06
        for item in syl_list
        if item["punct"] in ("?", "!", ".", ",")
    )
    tail_duration = slot_duration * tail_slots + 0.04
    raw_syl_duration_sum = sum(slots * slot_duration for slots in allocated_slots)
    active_duration = max(0.1, bar_duration_sec - lead_duration - tail_duration - punctuation_duration)
    scale_factor = active_duration / raw_syl_duration_sum if raw_syl_duration_sum > active_duration else 1.0

    target_durations = []
    for item, slots in zip(syl_list, allocated_slots):
        rel_pos = item["syl_idx"] / max(1, total_syllable_count)
        is_snare_region = (0.20 <= rel_pos <= 0.35) or (0.70 <= rel_pos <= 0.85) if genre == "boom_bap" else (0.45 <= rel_pos <= 0.60)
        micro_offset = 0.0
        if genre == "boom_bap" and (item["stress"] >= 1.15 or is_snare_region):
            micro_offset = 0.012
        elif genre == "trap" and (item["stress"] >= 1.15 or is_snare_region):
            micro_offset = -0.010
        target_durations.append(slots * slot_duration * scale_factor + micro_offset)

    duration_total = sum(target_durations)
    if duration_total > 0:
        target_durations = [duration * active_duration / duration_total for duration in target_durations]

    items: list[dict[str, Any]] = []
    cap_overflow = 0.0

    for item, slots, target_dur in zip(syl_list, allocated_slots, target_durations):
        max_cap = MAX_SYLLABLE_DUR_SEC
        if target_dur > max_cap:
            cap_overflow += (target_dur - max_cap)
            target_dur = max_cap

        items.append({
            "type": "SYLLABLE",
            "phones": item["phones"],
            "dur": target_dur,
            "weight": item["weight"],
            "stress": item["stress"],
            "midi_note": 63 if item["is_priority_attack"] and slots >= 3 else item["midi_note"],
        })
        if item["punct"] in ("?", "!", ".", ","):
            sp_dur = 0.04 if item["punct"] == "," else 0.06
            items.append({
                "type": "SP",
                "phones": None,
                "dur": sp_dur,
                "weight": 0.0,
                "midi_note": 0,
            })

    tail_duration += cap_overflow

    # Prepare outputs
    phonemes: list[str] = ["SP"]
    ph_num: list[int] = [1]
    durations: list[float] = [lead_duration]
    note_seq: list[str] = ["rest"]
    note_dur: list[float] = [lead_duration]

    for item in items:
        if item["type"] == "SP":
            dur = item["dur"]
            phonemes.append("SP")
            ph_num.append(1)
            durations.append(dur)
            note_seq.append("rest")
            note_dur.append(dur)
        else:
            syl_dur = item["dur"]
            syl_phones = item["phones"]
            ling_stress = item["stress"]
            assert isinstance(syl_phones, list)

            phonemes.extend(syl_phones)
            ph_num.append(len(syl_phones))

            # Enhanced Option A phoneme weighting with category & BPM scaling (115 BPM threshold)
            phone_weights = [_phoneme_weight_v2(sym, bpm=bpm, stress=ling_stress) for sym in syl_phones]
            pw_sum = sum(phone_weights)
            syl_phone_durs = [syl_dur * pw / pw_sum for pw in phone_weights]

            # Enforce minimum duration floors (Vowel: 85ms, Fricative: 40ms, Plosive: 30ms, Nasal/Liquid: 35ms)
            for i, sym in enumerate(syl_phones):
                floor = 0.0
                if sym in VOWEL_SYMBOLS:
                    floor = MIN_VOWEL_DUR_SEC
                elif sym in FRICATIVE_SYMBOLS:
                    floor = MIN_FRICATIVE_DUR_SEC
                elif sym in PLOSIVE_SYMBOLS:
                    floor = MIN_PLOSIVE_DUR_SEC
                elif sym in NASAL_LIQUID_SYMBOLS:
                    floor = MIN_NASAL_DUR_SEC

                if floor > 0.0 and syl_phone_durs[i] < floor:
                    target_floor = min(floor, syl_dur * 0.75)
                    if target_floor > syl_phone_durs[i]:
                        diff = target_floor - syl_phone_durs[i]
                        other_indices = [k for k in range(len(syl_phones)) if k != i]
                        other_sum = sum(syl_phone_durs[k] for k in other_indices)
                        if other_sum > 0:
                            syl_phone_durs[i] = target_floor
                            for k in other_indices:
                                syl_phone_durs[k] -= diff * (syl_phone_durs[k] / other_sum)

            durations.extend(syl_phone_durs)

            midi_note = int(item["midi_note"])
            note_name = _midi_note_to_name(midi_note) if midi_note > 0 else "C4"
            note_seq.append(note_name)
            note_dur.append(syl_dur)

    phonemes.append("SP")
    ph_num.append(1)
    durations.append(tail_duration)
    note_seq.append("rest")
    note_dur.append(tail_duration)

    rounded_durations = [round(v, 6) for v in durations]
    rounded_durations[-1] = round(rounded_durations[-1] + (bar_duration_sec - sum(rounded_durations)), 6)
    rounded_note_durs = [round(v, 6) for v in note_dur]
    rounded_note_durs[-1] = round(rounded_note_durs[-1] + (bar_duration_sec - sum(rounded_note_durs)), 6)

    template_name = f"adaptive_hierarchical_{genre}_{total_syllable_count}syl_{word_count}words"
    return phonemes, ph_num, rounded_durations, template_name, note_seq, rounded_note_durs


def _note_name_to_midi(note_name: str) -> int:
    if note_name == "rest" or not note_name:
        return 0
    match = re.match(r"^([A-G]#?)(-?\d+)$", note_name)
    if not match:
        return 60
    name, octave = match.group(1), int(match.group(2))
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return (octave + 1) * 12 + names.index(name)


def _build_f0_curve(
    symbols: list[str],
    durations: list[float],
    base_f0_hz: float,
    bar_index: int,
    note_seq: list[str] | None = None,
    ph_num: list[int] | None = None,
) -> list[float]:
    values: list[float] = []

    # Map target F0 Hz for each phoneme symbol
    target_hz_list: list[float] = []
    if note_seq and ph_num and len(note_seq) == len(ph_num):
        cursor = 0
        for note_name, count in zip(note_seq, ph_num):
            midi_note = _note_name_to_midi(note_name)
            hz = _midi_note_to_hz(midi_note, base_f0_hz) if midi_note > 0 else 0.0
            for _ in range(count):
                target_hz_list.append(hz)
    else:
        target_hz_list = [base_f0_hz if sym != "SP" else 0.0 for sym in symbols]

    # 1. Frame-by-frame initial F0 generation
    for symbol, duration, target_hz in zip(symbols, durations, target_hz_list):
        frame_count = max(1, round(duration / F0_TIMESTEP_SEC))
        for _ in range(frame_count):
            if symbol == "SP" or target_hz <= 0.0:
                values.append(0.0)
            else:
                values.append(target_hz)

    # 2. Apply portamento/glide smoothing to voiced segments (prevents sharp pitch clicks at boundaries)
    n_frames = len(values)
    i = 0
    while i < n_frames:
        if values[i] == 0.0:
            i += 1
            continue

        start_idx = i
        while i < n_frames and values[i] > 0.0:
            i += 1
        end_idx = i

        segment_len = end_idx - start_idx
        if segment_len <= 1:
            continue

        for j in range(segment_len):
            idx = start_idx + j

            # Portamento/Glide smoothing at segment boundaries
            glide_frames = min(4, segment_len // 2)
            if j < glide_frames:
                ramp = 0.96 + 0.04 * (j / glide_frames)
                values[idx] *= ramp
            elif j >= segment_len - glide_frames:
                k = segment_len - 1 - j
                ramp = 0.96 + 0.04 * (k / glide_frames)
                values[idx] *= ramp

    return values


def _validate_plan_duration(plan: FlowPlan) -> None:
    for bar in plan.bars:
        total = sum(phoneme.durationSec for phoneme in bar.phonemes)
        if abs(total - plan.beatMap.barDurationSec) > 0.0001:
            raise RuntimeError(f"{bar.barIndex}마디 duration 합계가 마디 길이와 맞지 않습니다.")


def _rounded(values: list[float]) -> list[float]:
    return [round(value, 6) for value in values]
