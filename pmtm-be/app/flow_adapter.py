import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path


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
    if len(lines) != TARGET_BARS:
        raise ValueError(f"가이드는 정확히 {TARGET_BARS}줄의 가사가 필요합니다. 현재 {len(lines)}줄입니다.")
    return lines


def build_beat_map(bpm: int, first_bar_start_sec: float) -> BeatMap:
    if bpm < 40 or bpm > 220:
        raise ValueError("BPM은 40부터 220 사이여야 합니다.")
    if not math.isfinite(first_bar_start_sec) or first_bar_start_sec < 0:
        raise ValueError("첫 마디 시작 시각은 0 이상의 유한한 값이어야 합니다.")

    beat_duration = 60.0 / bpm
    bar_duration = beat_duration * 4
    bar_starts = [first_bar_start_sec + index * bar_duration for index in range(TARGET_BARS)]
    beat_times = [first_bar_start_sec + index * beat_duration for index in range(TARGET_BARS * 4)]
    return BeatMap(
        bpm=bpm,
        beatsPerBar=4,
        barCount=TARGET_BARS,
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
    beat_map = build_beat_map(bpm, first_bar_start_sec)
    bars: list[FlowBar] = []
    for index, line in enumerate(lines):
        syllables = _extract_hangul_syllables(line)
        if not syllables:
            raise ValueError(f"{index + 1}마디에 합성 가능한 한글 음절이 없습니다.")
        if len(syllables) > MAX_SYLLABLES_PER_BAR:
            raise ValueError(
                f"{index + 1}마디가 너무 조밀합니다. 한 마디는 한글 {MAX_SYLLABLES_PER_BAR}음절 이하로 수정해주세요."
            )

        phonemes, ph_num = _line_to_phonemes(syllables)
        durations, template = _allocate_phoneme_durations(
            phonemes,
            ph_num,
            beat_map.barDurationSec,
            index,
            len(syllables),
        )
        start = beat_map.barStartTimes[index]
        bars.append(
            FlowBar(
                barIndex=index + 1,
                text=line,
                startSec=start,
                endSec=round(start + beat_map.barDurationSec, 6),
                template=template,
                syllableCount=len(syllables),
                phonemes=[
                    FlowPhoneme(symbol=symbol, durationSec=duration)
                    for symbol, duration in zip(phonemes, durations)
                ],
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
        _, ph_num = _line_to_phonemes(_extract_hangul_syllables(bar.text))
        if sum(ph_num) != len(symbols):
            raise RuntimeError(f"{bar.barIndex}마디 음절과 음소 매핑이 맞지 않습니다.")
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
                "note_seq": "C4",
                "note_dur": f"{plan.beatMap.barDurationSec:.6f}",
                "note_slur": "0",
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
    unsupported = re.sub(r"[가-힣\s.,!?~'\"()\[\]{}:;·…-]", "", text)
    if unsupported:
        raise ValueError(f"현재 SVS 테스트는 한글 가사만 지원합니다. 지원하지 않는 문자: {unsupported[:20]}")
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


def _allocate_phoneme_durations(
    phonemes: list[str],
    ph_num: list[int],
    bar_duration_sec: float,
    bar_index: int,
    syllable_count: int,
) -> tuple[list[float], str]:
    if syllable_count <= 8:
        template = "laid_back_eighths"
        lead_ratio, tail_ratio = (0.10, 0.16) if bar_index % 2 == 0 else (0.16, 0.10)
    elif syllable_count <= 16:
        template = "syncopated_sixteenths"
        lead_ratio, tail_ratio = (0.06, 0.10) if bar_index % 2 == 0 else (0.10, 0.06)
    else:
        template = "dense_thirty_seconds"
        lead_ratio, tail_ratio = 0.04, 0.05

    lead_duration = bar_duration_sec * lead_ratio
    tail_duration = bar_duration_sec * tail_ratio
    active_duration = bar_duration_sec - lead_duration - tail_duration
    syllable_duration = active_duration / syllable_count
    durations = [lead_duration]
    cursor = 1
    for phone_count in ph_num[1:-1]:
        symbols = phonemes[cursor : cursor + phone_count]
        weights = [_phoneme_weight(symbol) for symbol in symbols]
        weight_sum = sum(weights)
        durations.extend(syllable_duration * weight / weight_sum for weight in weights)
        cursor += phone_count
    durations.append(tail_duration)

    rounded = [round(value, 6) for value in durations]
    rounded[-1] = round(rounded[-1] + (bar_duration_sec - sum(rounded)), 6)
    return rounded, template


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
    for symbol, duration in zip(symbols, durations):
        frame_count = max(1, round(duration / F0_TIMESTEP_SEC))
        if symbol == "SP":
            values.extend([0.0] * frame_count)
            continue
        accent = 1.0 + (0.025 if bar_index % 2 else -0.012)
        values.extend([base_f0_hz * accent] * frame_count)
    return values


def _validate_plan_duration(plan: FlowPlan) -> None:
    for bar in plan.bars:
        total = sum(phoneme.durationSec for phoneme in bar.phonemes)
        if abs(total - plan.beatMap.barDurationSec) > 0.0001:
            raise RuntimeError(f"{bar.barIndex}마디 duration 합계가 마디 길이와 맞지 않습니다.")


def _rounded(values: list[float]) -> list[float]:
    return [round(value, 6) for value in values]
