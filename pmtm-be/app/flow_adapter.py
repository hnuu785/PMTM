import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# pyrefly: ignore [missing-import]
from g2pk2 import G2p
# pyrefly: ignore [missing-import]
import pronouncing

g2p = G2p()




TARGET_BARS = 8
MAX_SYLLABLES_PER_BAR = 24
F0_TIMESTEP_SEC = 0.01
VOICED_ENERGY_DB = -26.0
SILENCE_ENERGY_DB = -80.0
VOICED_BREATHINESS_DB = -60.0
SILENCE_BREATHINESS_DB = -80.0

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
    # POTG's dictionary lists `wx i`, but `wx` is absent from its acoustic
    # phoneme inventory. `ui` is the closest supported single vowel.
    "ㅢ": ("", "ui"),
}

ARPABET_VOWELS = {
    "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey",
    "ih", "iy", "ow", "oy", "uh", "uw"
}
ARPABET_TO_POTG = {
    "aa": "aa", "ae": "ae", "ah": "ah", "ao": "ao", "aw": "aw",
    "ax": "ax", "ay": "ay", "b": "b", "ch": "ch", "d": "d",
    "dh": "dh", "dx": "dx", "eh": "eh", "er": "er", "ey": "ey",
    "f": "f", "g": "g", "hh": "hh", "ih": "ih", "iy": "iy",
    "jh": "jh", "k": "k", "l": "l", "m": "m", "n": "n",
    "ng": "ng", "ow": "ow", "oy": "oy", "p": "p", "r": "r",
    "s": "s", "sh": "sh", "t": "t", "th": "th", "uh": "uh",
    "uw": "uw", "v": "v", "w": "w", "y": "y", "z": "z", "zh": "zh"
}
LETTER_TO_POTG = {
    "a": "a", "b": "b", "c": "k", "d": "d", "e": "e", "f": "f", "g": "g",
    "h": "hh", "i": "i", "j": "jh", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "o", "p": "p", "q": "k", "r": "r", "s": "s", "t": "t", "u": "u",
    "v": "v", "w": "w", "x": "k", "y": "i", "z": "z"
}


def _english_word_to_syllables(word: str) -> list[list[str]]:
    clean = re.sub(r"[^a-zA-Z']", "", word).lower()
    if not clean:
        return []
    lookup = re.sub(r"'", "", clean)
    phones_list = pronouncing.phones_for_word(clean) or pronouncing.phones_for_word(lookup)
    if phones_list:
        raw_phones = [re.sub(r"\d+", "", p).lower() for p in phones_list[0].split()]
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


def _extract_word_syllables_and_text(text: str) -> tuple[list[WordChunk], str]:
    unsupported = re.sub(r"[가-힣a-zA-Z\s.,!?~'\"()\[\]{}:;·…-]", "", text)
    if unsupported:
        raise ValueError(f"현재 SVS 테스트는 한글과 영문 가사만 지원합니다. 지원하지 않는 문자: {unsupported[:20]}")
    raw_words = text.strip().split()
    word_chunks: list[WordChunk] = []
    g2p_parts: list[str] = []

    for raw_word in raw_words:
        punct_match = re.search(r"([?,!.])+$", raw_word)
        punct = punct_match.group(1) if punct_match else None

        tokens = re.findall(r"[가-힣]+|[a-zA-Z']+", raw_word)
        current_word_syllables: list[list[str]] = []
        for token in tokens:
            if re.match(r"^[가-힣]+$", token):
                g2p_token = g2p(token)
                g2p_parts.append(g2p_token)
                syllables = re.findall(r"[가-힣]", g2p_token)
                for syl in syllables:
                    current_word_syllables.append(_syllable_to_phonemes(syl))
            else:
                g2p_parts.append(token)
                e_syls = _english_word_to_syllables(token)
                current_word_syllables.extend(e_syls)
        if current_word_syllables:
            word_chunks.append(WordChunk(syllables=current_word_syllables, punctuation=punct))

    return word_chunks, " ".join(g2p_parts)


@dataclass(frozen=True)
class BeatMap:
    bpm: int
    beatsPerBar: int
    barCount: int
    firstBarStartSec: float
    barDurationSec: float
    barStartTimes: list[float]
    beatTimes: list[float]


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


@dataclass(frozen=True)
class FlowPlan:
    version: int
    voicebank: str
    beatMap: BeatMap
    bars: list[FlowBar]


def parse_eight_bar_lyrics(lyrics: str) -> list[str]:
    lines = [
        line.strip()
        for line in lyrics.splitlines()
        if line.strip() and not line.strip().lower().startswith("[verse")
    ]
    if len(lines) not in (8, 16):
        raise ValueError(f"가이드는 정확히 8줄 또는 16줄의 가사가 필요합니다. 현재 {len(lines)}줄입니다.")
    return lines


def build_beat_map(bpm: int, first_bar_start_sec: float, bar_count: int = 8) -> BeatMap:
    if bpm < 40 or bpm > 220:
        raise ValueError("BPM은 40부터 220 사이여야 합니다.")
    if not math.isfinite(first_bar_start_sec) or first_bar_start_sec < 0:
        raise ValueError("첫 마디 시작 시각은 0 이상의 유한한 값이어야 합니다.")

    # 115 BPM 이상의 빠른 템포는 물리적 1마디 길이를 확보하기 위해 하프타임으로 보정합니다.
    # 예: 140 BPM -> 70 BPM (1마디가 1.71초에서 3.42초로 늘어남)
    actual_bpm = bpm / 2.0 if bpm >= 115.0 else float(bpm)

    beat_duration = 60.0 / actual_bpm
    beats_per_unit = 4 if bar_count == 8 else 2
    bar_duration = beat_duration * beats_per_unit
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
    )



def build_flow_plan(
    lyrics: str,
    bpm: int,
    first_bar_start_sec: float,
    voicebank: str,
    *,
    base_f0_hz: float,
) -> FlowPlan:
    lines = parse_eight_bar_lyrics(lyrics)
    beat_map = build_beat_map(bpm, first_bar_start_sec, bar_count=len(lines))
    bars: list[FlowBar] = []
    for index, line in enumerate(lines):
        word_chunks, g2p_line = _extract_word_syllables_and_text(line)
        total_syllable_count = sum(len(w.syllables) for w in word_chunks)
        if total_syllable_count == 0:
            raise ValueError(f"{index + 1}마디에 합성 가능한 음절이 없습니다.")
        if total_syllable_count > MAX_SYLLABLES_PER_BAR:
            raise ValueError(
                f"{index + 1}마디가 너무 조밀합니다. 한 마디는 {MAX_SYLLABLES_PER_BAR}음절 이하로 수정해주세요."
            )

        phonemes, ph_num, durations, template = _allocate_word_hierarchical_durations(
            word_chunks,
            beat_map.barDurationSec,
            index,
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
            )
        )

    plan = FlowPlan(version=1, voicebank=voicebank, beatMap=beat_map, bars=bars)
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

        note_seq_list: list[str] = []
        note_dur_list: list[float] = []
        note_slur_list: list[int] = []

        cursor = 0
        is_after_rest = True

        for count in ph_num:
            group_symbols = symbols[cursor : cursor + count]
            group_durs = durations[cursor : cursor + count]
            group_dur_sum = sum(group_durs)

            if group_symbols == ["SP"]:
                note_seq_list.append("rest")
                note_dur_list.append(group_dur_sum)
                note_slur_list.append(0)
                is_after_rest = True
            else:
                note_seq_list.append("C4")
                note_dur_list.append(group_dur_sum)
                if is_after_rest:
                    note_slur_list.append(0)
                    is_after_rest = False
                else:
                    note_slur_list.append(1)

            cursor += count

        f0_values = _build_f0_curve(symbols, durations, base_f0_hz, bar.barIndex)
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


def _extract_hangul_syllables(text: str) -> list[str]:
    unsupported = re.sub(r"[가-힣a-zA-Z\s.,!?~'\"()\[\]{}:;·…-]", "", text)
    if unsupported:
        raise ValueError(f"현재 SVS 테스트는 한글과 영문 가사만 지원합니다. 지원하지 않는 문자: {unsupported[:20]}")
    return re.findall(r"[가-힣]", text)



def _line_to_phonemes(syllables: list[str]) -> tuple[list[str], list[int]]:
    phones = ["SP"]
    counts = [1]
    for syllable in syllables:
        syllable_phones = _syllable_to_phonemes(syllable)
        phones.extend(syllable_phones)
        counts.append(len(syllable_phones))
    phones.append("SP")
    counts.append(1)
    return phones, counts


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
MAX_SYLLABLE_DUR_SEC = 0.60


def _get_word_syllable_weights(syllable_count: int) -> list[float]:
    if syllable_count == 1:
        return [1.5]
    elif syllable_count == 2:
        return [2.0, 1.0]
    elif syllable_count == 3:
        return [2.0, 1.0, 1.0]
    elif syllable_count == 4:
        return [1.5, 1.0, 1.5, 1.0]
    else:
        return [2.0] + [1.0] * (syllable_count - 1)


def _allocate_word_hierarchical_durations(
    word_chunks: list[WordChunk],
    bar_duration_sec: float,
    bar_index: int,
) -> tuple[list[str], list[int], list[float], str]:
    total_syllable_count = sum(len(w.syllables) for w in word_chunks)

    if total_syllable_count <= 8:
        lead_ratio, tail_ratio = 0.10, 0.12
    elif total_syllable_count <= 16:
        lead_ratio, tail_ratio = 0.06, 0.08
    else:
        lead_ratio, tail_ratio = 0.04, 0.05

    lead_duration = bar_duration_sec * lead_ratio
    tail_duration = bar_duration_sec * tail_ratio
    active_duration = bar_duration_sec - lead_duration - tail_duration

    if total_syllable_count > 0:
        avail_per_syl = active_duration / total_syllable_count
        target_min_dur = max(ABSOLUTE_MIN_DUR_SEC, min(DEFAULT_MIN_DUR_SEC, avail_per_syl))
    else:
        target_min_dur = DEFAULT_MIN_DUR_SEC

    required_syl_time = total_syllable_count * target_min_dur
    residual_time = active_duration - required_syl_time

    word_count = len(word_chunks)
    items: list[dict[str, object]] = []

    for w_idx, chunk in enumerate(word_chunks):
        syl_weights = _get_word_syllable_weights(len(chunk.syllables))
        for s_idx, syl_phones in enumerate(chunk.syllables):
            items.append({
                "type": "SYLLABLE",
                "phones": syl_phones,
                "base_dur": target_min_dur,
                "weight": syl_weights[s_idx],
            })

        if chunk.punctuation in ("?", "!", "."):
            sp_weight = 1.2
        elif chunk.punctuation == ",":
            sp_weight = 0.8
        else:
            sp_weight = 0.0

        if sp_weight > 0:
            if residual_time <= 0:
                sp_base = 0.06 if chunk.punctuation in ("?", "!", ".") else (0.04 if chunk.punctuation == "," else 0.0)
                if sp_base > 0:
                    items.append({
                        "type": "SP",
                        "phones": None,
                        "base_dur": sp_base,
                        "weight": 0.0,
                    })
            else:
                items.append({
                    "type": "SP",
                    "phones": None,
                    "base_dur": 0.0,
                    "weight": sp_weight,
                })

    total_weight = sum(float(item["weight"]) for item in items)
    base_sum = sum(float(item["base_dur"]) for item in items)
    effective_residual = max(0.0, active_duration - base_sum)

    cap_overflow = 0.0
    raw_item_durations: list[float] = []

    for item in items:
        w = float(item["weight"])
        b = float(item["base_dur"])
        add_dur = (effective_residual * (w / total_weight)) if total_weight > 0 else 0.0
        dur = b + add_dur

        if item["type"] == "SYLLABLE" and dur > MAX_SYLLABLE_DUR_SEC:
            cap_overflow += (dur - MAX_SYLLABLE_DUR_SEC)
            dur = MAX_SYLLABLE_DUR_SEC
        raw_item_durations.append(dur)

    tail_duration += cap_overflow

    phonemes: list[str] = ["SP"]
    ph_num: list[int] = [1]
    durations: list[float] = [lead_duration]

    for item, dur in zip(items, raw_item_durations):
        if item["type"] == "SP":
            phonemes.append("SP")
            ph_num.append(1)
            durations.append(dur)
        else:
            syl_phones = item["phones"]
            assert isinstance(syl_phones, list)
            phonemes.extend(syl_phones)
            ph_num.append(len(syl_phones))
            phone_weights = [_phoneme_weight(sym) for sym in syl_phones]
            pw_sum = sum(phone_weights)
            durations.extend(dur * pw / pw_sum for pw in phone_weights)

    phonemes.append("SP")
    ph_num.append(1)
    durations.append(tail_duration)

    rounded = [round(v, 6) for v in durations]
    rounded[-1] = round(rounded[-1] + (bar_duration_sec - sum(rounded)), 6)

    template_name = f"adaptive_hierarchical_{total_syllable_count}syl_{word_count}words"
    return phonemes, ph_num, rounded, template_name





def _phoneme_weight(symbol: str) -> float:
    if symbol in {"a", "e", "eo", "eu", "i", "o", "u", "a1", "a2", "a3", "a4", "e1", "e2", "e3", "e4", "eo1", "eo2", "eo3", "eo4", "eu1", "eu2", "eu3", "eu4", "i1", "i2", "i3", "i4", "o1", "o2", "o3", "o4", "u1", "u2", "u3", "u4"}:
        return 0.58
    if symbol in {"K", "N", "M", "P", "cl"}:
        return 0.18
    return 0.24


def _build_f0_curve(
    symbols: list[str],
    durations: list[float],
    base_f0_hz: float,
    bar_index: int,
) -> list[float]:
    values: list[float] = []
    
    # 1. Frame-by-frame initial F0 generation (flat pitch for voiced segments)
    for symbol, duration in zip(symbols, durations):
        frame_count = max(1, round(duration / F0_TIMESTEP_SEC))
        for _ in range(frame_count):
            if symbol == "SP":
                values.append(0.0)
            else:
                values.append(base_f0_hz)

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
            
            # Portamento/Glide smoothing at segment boundaries (prevents sharp pitch clicks)
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
