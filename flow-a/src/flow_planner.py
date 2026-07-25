import argparse
import json
import math
import re
from pathlib import Path
from typing import Callable


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

ONSET = {
    "ㄱ": "g", "ㄲ": "kk", "ㄴ": "n", "ㄷ": "d", "ㄸ": "tt", "ㄹ": "rx",
    "ㅁ": "m", "ㅂ": "b", "ㅃ": "pp", "ㅅ": "sc", "ㅆ": "s", "ㅇ": "",
    "ㅈ": "jh", "ㅉ": "jj", "ㅊ": "ch", "ㅋ": "k", "ㅌ": "t", "ㅍ": "p", "ㅎ": "hh",
}
VOWEL = {
    "ㅏ": "a", "ㅐ": "e", "ㅑ": "ia", "ㅒ": "ie", "ㅓ": "eo", "ㅔ": "e",
    "ㅕ": "ieo", "ㅖ": "ie", "ㅗ": "o", "ㅘ": "oa", "ㅙ": "oe", "ㅚ": "oe",
    "ㅛ": "io", "ㅜ": "u", "ㅝ": "uo", "ㅞ": "oe", "ㅟ": "ui", "ㅠ": "iu",
    "ㅡ": "eu", "ㅢ": "i", "ㅣ": "i",
}
CODA = {
    "": "", "ㄱ": "kcl", "ㄲ": "kcl", "ㄳ": "kcl", "ㄴ": "n", "ㄵ": "n",
    "ㄶ": "n", "ㄷ": "tcl", "ㄹ": "l", "ㄺ": "kcl", "ㄻ": "m", "ㄼ": "pcl",
    "ㄽ": "l", "ㄾ": "l", "ㄿ": "pcl", "ㅀ": "l", "ㅁ": "m", "ㅂ": "pcl",
    "ㅄ": "pcl", "ㅅ": "tcl", "ㅆ": "tcl", "ㅇ": "ng", "ㅈ": "tcl",
    "ㅊ": "tcl", "ㅋ": "kcl", "ㅌ": "tcl", "ㅍ": "pcl", "ㅎ": "tcl",
}
SNARE_SLOTS = {"boom_bap": (4, 12), "trap": (8,)}
PITCH_HZ = {"potg": 190.0, "kitane": 205.0, "rang": 145.0, "lunar": 195.0}


def load_lyrics(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) not in (8, 16):
        raise ValueError(f"가사는 정확히 8줄 또는 16줄이어야 합니다. 현재 {len(lines)}줄입니다.")
    return lines


def get_bpm(analysis: dict) -> float:
    payload = analysis.get("bpm")
    if isinstance(payload, dict):
        value = payload.get("fixed_integer", payload.get("selected", payload.get("librosa_raw")))
    else:
        value = payload
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("비트 분석 JSON에서 유효한 BPM을 찾지 못했습니다.")
    return float(value)


def get_complete_bars(analysis: dict, count: int) -> list[list[dict]]:
    grid = analysis.get("absolute_grid")
    if not isinstance(grid, list):
        raise ValueError("비트 분석 JSON에 absolute_grid 배열이 필요합니다.")

    grouped: dict[int, list[dict]] = {}
    for entry in grid:
        if not isinstance(entry, dict):
            raise ValueError("absolute_grid 항목은 객체여야 합니다.")
        bar = entry.get("bar")
        if not isinstance(bar, int):
            raise ValueError("absolute_grid 항목에 정수 bar가 필요합니다.")
        grouped.setdefault(bar, []).append(entry)

    bars = []
    for bar_number in sorted(grouped):
        entries = sorted(grouped[bar_number], key=lambda item: item["slot"])
        if len(entries) == 16:
            bars.append(entries)
        elif bars:
            break
    if len(bars) < count:
        raise ValueError(f"완전한 16슬롯 마디가 {count}개 필요하지만 {len(bars)}개만 있습니다.")
    return bars[:count]


def hangul_to_phonemes(syllable: str) -> list[str]:
    code = ord(syllable) - 0xAC00
    onset = CHOSEONG[code // 588]
    vowel = JUNGSEONG[(code % 588) // 28]
    coda = JONGSEONG[code % 28]
    return [phone for phone in (ONSET[onset], VOWEL[vowel], CODA[coda]) if phone]


def normalize_korean(text: str) -> str:
    try:
        from g2pk2 import G2p
    except ImportError as exc:
        raise RuntimeError("한국어 발음 변환을 위해 requirements.txt의 g2pk2를 설치해야 합니다.") from exc
    return G2p()(text)


def extract_syllables(text: str, normalizer: Callable[[str], str]) -> tuple[str, list[str]]:
    unsupported = re.sub(r"[가-힣\s.,!?~'\"()\[\]{}:;·…-]", "", text)
    if unsupported:
        raise ValueError(f"현재 초안은 한글 가사만 지원합니다. 지원하지 않는 문자: {unsupported[:20]}")
    normalized = normalizer(text)
    return normalized, re.findall(r"[가-힣]", normalized)


def choose_slots(syllable_count: int, snare_slots: tuple[int, ...]) -> list[int]:
    if syllable_count > 16:
        raise ValueError("현재 슬롯 기반 초안은 한 마디 최대 16음절을 지원합니다.")
    if syllable_count == 1:
        return [snare_slots[0]]

    slots = [round(index * 15 / syllable_count) for index in range(syllable_count)]
    used = set(slots)
    for snare in snare_slots:
        nearest_index = min(range(len(slots)), key=lambda index: abs(slots[index] - snare))
        candidate = snare
        if candidate not in used or slots[nearest_index] == candidate:
            used.discard(slots[nearest_index])
            slots[nearest_index] = candidate
            used.add(candidate)
    return sorted(slots)


def plan_bar(
    text: str,
    grid: list[dict],
    genre: str,
    normalizer: Callable[[str], str],
    fallback_slot_sec: float,
) -> dict:
    normalized, syllables = extract_syllables(text, normalizer)
    slots = choose_slots(len(syllables), SNARE_SLOTS[genre])
    bar_start = float(grid[0]["time_sec"])
    bar_end = bar_start + fallback_slot_sec * 16
    placements = []

    for index, (syllable, slot) in enumerate(zip(syllables, slots)):
        onset = float(grid[slot]["time_sec"])
        next_onset = (
            float(grid[slots[index + 1]]["time_sec"])
            if index + 1 < len(slots)
            else bar_end
        )
        available = next_onset - onset
        duration = min(available * 0.9, fallback_slot_sec * 2)
        rest = max(0.0, available - duration)
        placements.append(
            {
                "syllable": syllable,
                "slot": slot,
                "onsetSec": round(onset, 6),
                "durationSec": round(duration, 6),
                "restAfterSec": round(rest, 6),
                "accent": any(abs(slot - snare) <= 1 for snare in SNARE_SLOTS[genre]),
                "phonemes": hangul_to_phonemes(syllable),
            }
        )

    return {
        "bar": int(grid[0]["bar"]),
        "text": text,
        "pronunciation": normalized,
        "startSec": round(bar_start, 6),
        "endSec": round(bar_end, 6),
        "syllableCount": len(syllables),
        "densitySyllablesPerSec": round(len(syllables) / (bar_end - bar_start), 3),
        "snareSlots": list(SNARE_SLOTS[genre]),
        "placements": placements,
    }


def build_flow_plan(
    analysis: dict,
    lyrics: list[str],
    genre: str,
    voicebank: str,
    normalizer: Callable[[str], str] = normalize_korean,
) -> dict:
    if genre not in SNARE_SLOTS:
        raise ValueError("genre는 boom_bap 또는 trap이어야 합니다.")
    bpm = get_bpm(analysis)
    bars = get_complete_bars(analysis, len(lyrics))
    fallback_slot_sec = 60.0 / bpm / 4.0
    planned_bars = [
        plan_bar(text, grid, genre, normalizer, fallback_slot_sec)
        for text, grid in zip(lyrics, bars)
    ]
    return {
        "version": 1,
        "sourceAudio": analysis.get("audio_file"),
        "bpm": bpm,
        "timeSignature": analysis.get("time_signature", "4/4"),
        "gridResolution": 16,
        "genre": genre,
        "voicebank": voicebank,
        "bars": planned_bars,
    }


def _split_duration(duration: float, phoneme_count: int) -> list[float]:
    if phoneme_count == 1:
        return [duration]
    weights = [0.22] + [0.58] + ([0.20] if phoneme_count == 3 else [])
    total = sum(weights)
    values = [round(duration * weight / total, 6) for weight in weights[:phoneme_count]]
    values[-1] = round(values[-1] + duration - sum(values), 6)
    return values


def to_diffsinger_ds(plan: dict, base_f0_hz: float) -> list[dict[str, str]]:
    sections = []
    for bar in plan["bars"]:
        ph_seq = []
        ph_dur = []
        ph_num = []
        note_seq = []
        note_dur = []
        note_slur = []
        cursor = bar["startSec"]

        for placement in bar["placements"]:
            gap = placement["onsetSec"] - cursor
            if gap > 0.000001:
                ph_seq.append("SP")
                ph_dur.append(gap)
                ph_num.append(1)
                note_seq.append("rest")
                note_dur.append(gap)
                note_slur.append(0)

            phones = placement["phonemes"]
            phone_durations = _split_duration(placement["durationSec"], len(phones))
            ph_seq.extend(phones)
            ph_dur.extend(phone_durations)
            ph_num.append(len(phones))
            note_seq.append("C4")
            note_dur.append(placement["durationSec"])
            note_slur.append(0)
            cursor = placement["onsetSec"] + placement["durationSec"]

        tail = bar["endSec"] - cursor
        if tail > 0.000001:
            ph_seq.append("SP")
            ph_dur.append(tail)
            ph_num.append(1)
            note_seq.append("rest")
            note_dur.append(tail)
            note_slur.append(0)

        sections.append(
            {
                "offset": f'{bar["startSec"]:.6f}',
                "text": bar["pronunciation"],
                "ph_seq": " ".join(ph_seq),
                "ph_dur": " ".join(f"{value:.6f}" for value in ph_dur),
                "ph_num": " ".join(str(value) for value in ph_num),
                "note_seq": " ".join(note_seq),
                "note_dur": " ".join(f"{value:.6f}" for value in note_dur),
                "note_slur": " ".join(str(value) for value in note_slur),
                "f0_seq": " ".join(
                    f"{base_f0_hz:.3f}" if symbol != "SP" else "0.000"
                    for symbol in ph_seq
                ),
                "f0_timestep": "0.010",
            }
        )
    return sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Korean DiffSinger Flow Planner")
    parser.add_argument("--beat-analysis", required=True, type=Path)
    parser.add_argument("--lyrics", required=True, type=Path)
    parser.add_argument("--genre", required=True, choices=tuple(SNARE_SLOTS))
    parser.add_argument("--voicebank", default="potg")
    parser.add_argument("--base-f0-hz", type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = json.loads(args.beat_analysis.read_text(encoding="utf-8"))
    plan = build_flow_plan(analysis, load_lyrics(args.lyrics), args.genre, args.voicebank)
    base_f0_hz = args.base_f0_hz or PITCH_HZ.get(args.voicebank, 190.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "flow-plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "score.ds").write_text(
        json.dumps(to_diffsinger_ds(plan, base_f0_hz), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output_dir / "flow-plan.json")
    print(args.output_dir / "score.ds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
