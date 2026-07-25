"""비트 분석 결과와 한국어 가사로 보컬 플로우 플랜을 생성한다.

실행 예시:
    python flow_planner.py --beat ../beat/outputs/120_trap_advanced_analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SLOTS_PER_BAR = 16
CONTENT_PREFIXES = ("NN", "VV", "VA", "MAG", "XR", "SL", "SN", "NP")
FUNCTION_PREFIXES = ("J", "E", "X", "S")
_kiwi: Any | None = None
_kiwi_checked = False


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path.resolve()}")
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 값은 객체여야 합니다.")
    return data


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def is_syllable(char: str) -> bool:
    return len(char) == 1 and "가" <= char <= "힣"


def syllable_count(text: str) -> int:
    return sum(is_syllable(char) for char in text) + len(re.findall(r"[A-Za-z0-9]+", text))


def decompose(char: str) -> dict[str, str] | None:
    if not is_syllable(char):
        return None
    onset = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
    vowel = ("ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ ㅠ ㅡ ㅢ ㅣ").split()
    coda = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")
    value = ord(char) - ord("가")
    return {"char": char, "onset": onset[value // 588], "vowel": vowel[value % 588 // 28], "coda": coda[value % 28]}


def morphology(text: str) -> list[dict[str, Any]]:
    global _kiwi, _kiwi_checked
    if not _kiwi_checked:
        _kiwi_checked = True
        try:
            from kiwipiepy import Kiwi
            _kiwi = Kiwi()
        except ImportError:
            pass
    if _kiwi is None:
        # Kiwi가 없을 때도 플래너를 실행할 수 있도록 어절을 기본 단위로 쓴다.
        return [{"form": m.group(), "tag": "NNG", "start": m.start(), "end": m.end()} for m in re.finditer(r"\S+", text)]
    return [{"form": t.form, "tag": t.tag, "start": t.start, "end": t.start + t.len} for t in _kiwi.tokenize(text)]


def split_line(text: str, threshold: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if syllable_count(text) < threshold:
        return [text]
    words = text.split()
    # 두 세그먼트는 각각 마디의 앞·뒤 8슬롯을 사용한다.
    candidates = []
    for index in range(1, len(words)):
        left, right = " ".join(words[:index]), " ".join(words[index:])
        if syllable_count(left) <= 8 and syllable_count(right) <= 8:
            candidates.append((abs(syllable_count(left) - syllable_count(right)), left, right))
    if candidates:
        _, left, right = min(candidates)
        return [left, right]
    return [text]


def allocate(text: str, start: int, end: int) -> list[dict[str, Any]]:
    chars = [(index, char) for index, char in enumerate(text) if is_syllable(char)]
    slots = end - start
    if len(chars) > slots:
        raise ValueError(f"음절 {len(chars)}개를 슬롯 {slots}개에 배치할 수 없습니다.")
    durations = [1] * len(chars)
    # 남는 슬롯은 끝 음절부터 나눠, 문장 마무리를 조금 더 길게 만든다.
    for index in range(slots - len(chars)):
        durations[-1 - (index % len(durations))] += 1
    tokens = morphology(text)
    cursor, result = start, []
    for number, ((char_index, char), duration) in enumerate(zip(chars, durations)):
        token = next((item for item in tokens if item["start"] <= char_index < item["end"]), None)
        tag = token["tag"] if token else None
        stress = 1.0 if tag and tag.startswith(CONTENT_PREFIXES) else 0.6
        result.append({"text": char, "syllable_index": number, "char_index": char_index,
                       "pos": tag, "stress": stress, "phonology": decompose(char),
                       "start_slot_in_bar": cursor, "duration_slots": duration,
                       "end_slot_in_bar_exclusive": cursor + duration})
        cursor += duration
    return result


def assign_pitch(syllables: list[dict[str, Any]], is_last: bool) -> None:
    for syllable in syllables:
        pos = syllable["pos"] or ""
        syllable["midi_note"] = 60 if pos.startswith(FUNCTION_PREFIXES) else 62
        syllable["is_slur"] = False
    if is_last and syllables:
        # 마지막 음절은 낮춰 구절의 종지를 만든다.
        syllables[-1]["midi_note"] = 58


def build_plan(lyrics: list[str], beat: dict[str, Any], verse_start_bar: int, threshold: int) -> dict[str, Any]:
    # 비트 분석의 절대 격자를 마디 단위로 다시 묶는다.
    grid_by_bar: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in beat["absolute_grid"]:
        grid_by_bar[int(point["bar"])].append(point)
    for grid in grid_by_bar.values():
        grid.sort(key=lambda point: point["slot"])
    slot_duration = 60 / float(beat["bpm"]["fixed_integer"]) / 4
    lines, warnings = [], []
    for line_index, text in enumerate(lyrics, 1):
        bar = verse_start_bar + line_index - 1
        grid = grid_by_bar.get(bar)
        if not grid or len(grid) < SLOTS_PER_BAR:
            raise ValueError(f"{bar}마디의 16분음표 그리드가 없습니다.")
        segments_text = split_line(text, threshold)
        layouts = [(0, 16)] if len(segments_text) == 1 else [(0, 8), (8, 16)]
        segments = []
        for index, (segment_text, (start, end)) in enumerate(zip(segments_text, layouts), 1):
            try:
                syllables = allocate(segment_text, start, end)
            except ValueError as error:
                warnings.append({"type": "overflow", "bar": bar, "text": segment_text, "message": str(error)})
                syllables = []
            assign_pitch(syllables, index == len(segments_text))
            for item in syllables:
                item["start_absolute_slot"] = grid[item["start_slot_in_bar"]]["slot"]
                item["end_absolute_slot_exclusive"] = item["start_absolute_slot"] + item["duration_slots"]
                item["start_sec"] = grid[item["start_slot_in_bar"]]["time_sec"]
                item["end_sec"] = item["start_sec"] + item["duration_slots"] * slot_duration
            segments.append({"segment_id": f"bar{bar:03d}_seg{index:02d}", "text": segment_text,
                             "start_slot_in_bar": start, "end_slot_in_bar_exclusive": end,
                             "syllables": syllables})
        lines.append({"line_index": line_index, "bar": bar, "original_text": text, "segments": segments, "rests": []})
    return {"metadata": {"source_audio_file": beat["audio_file"], "bpm": beat["bpm"]["fixed_integer"],
                            "time_signature": beat["time_signature"], "language": "ko", "slot_unit": "16th_note",
                            "verse_start_bar": verse_start_bar, "line_count": len(lines)},
            "lines": lines, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description="비트 분석과 가사를 보컬 플로우 플랜으로 변환합니다.")
    parser.add_argument("--beat", type=Path, required=True, help="beat_analyze.py가 만든 JSON 경로")
    parser.add_argument("--lyrics", type=Path, default=Path("lyrics.json"), help="가사 JSON 경로")
    parser.add_argument("--output", type=Path, default=Path("outputs/flow_plan.json"), help="저장할 플랜 경로")
    parser.add_argument("--split-threshold", type=int, default=13, help="이 음절 수 이상이면 2절 분할을 시도")
    args = parser.parse_args()
    # TODO(API): 가사와 비트 분석 JSON은 이후 API 요청 본문으로 받는다.
    lyric_data, beat = load_json(args.lyrics), load_json(args.beat)
    plan = build_plan(lyric_data["lyrics"], beat, int(lyric_data.get("verse_start_bar", 9)), args.split_threshold)
    save_json(plan, args.output)
    print(f"완료: {args.output.resolve()} | {len(plan['lines'])}줄 | 경고 {len(plan['warnings'])}건")


if __name__ == "__main__":
    main()
