"""Generate a Korean rap flow plan from beat analysis and lyrics."""

from collections import defaultdict
from pathlib import Path
from typing import Any
import json
import math
import re

from kiwipiepy import Kiwi

from diffsinger_export import export_diffsinger

kiwi = Kiwi()

BASE_DIR = Path(__file__).resolve().parent
BEAT_ANALYSIS_PATH = BASE_DIR.parent / "beat" / "outputs" / "120_trap_advanced_analysis.json"
LYRICS_PATH = BASE_DIR / "lyrics.json"
OUTPUT_PATH = BASE_DIR / "outputs" / "flow_plan.json"

VERSE_START_BAR = 9
SLOTS_PER_BAR = 16
SPLIT_THRESHOLD = 13

ONE_SEGMENT_LAYOUT = (0, 15)
# 두 segment는 16개 마디 slot을 빈틈없이 나눠 overflow를 막는다.
TWO_SEGMENT_LAYOUT = ((0, 8), (8, 16))

# Kiwi 품사 기준
CONTENT_POS_PREFIXES = (
    "NN",  # 명사
    "VV",  # 동사
    "VA",  # 형용사
    "MAG", # 일반부사
    "XR",  # 어근
    "SL",  # 외국어
    "SN",  # 숫자
)

FUNCTION_POS_PREFIXES = (
    "J",   # 조사
    "E",   # 어미
    "X",   # 접사
    "SP", "SS", "SF", "SE", "SO", "SW",
)

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path.resolve()}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 구조는 객체(dict)여야 합니다.")

    return data


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


CHOSEONG = [
    "ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ",
    "ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"
]

JUNGSEONG = [
    "ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ",
    "ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"
]

JONGSEONG = [
    "", "ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ",
    "ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ",
    "ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"
]

VOWEL_FAMILY = {
    "ㅏ":"A", "ㅑ":"A", "ㅘ":"A",
    "ㅐ":"AE", "ㅒ":"AE", "ㅙ":"AE",
    "ㅓ":"EO", "ㅕ":"EO", "ㅝ":"EO",
    "ㅔ":"E", "ㅖ":"E", "ㅞ":"E",
    "ㅗ":"O", "ㅛ":"O", "ㅚ":"O",
    "ㅜ":"U", "ㅠ":"U", "ㅟ":"U",
    "ㅡ":"EU", "ㅢ":"EU",
    "ㅣ":"I",
}

CODA_FAMILY = {
    "":"OPEN",
    "ㄱ":"K", "ㄲ":"K", "ㄳ":"K", "ㅋ":"K",
    "ㄴ":"N", "ㄵ":"N", "ㄶ":"N",
    "ㄷ":"T", "ㅅ":"T", "ㅆ":"T", "ㅈ":"T", "ㅊ":"T", "ㅌ":"T", "ㅎ":"T",
    "ㄹ":"L", "ㄺ":"L", "ㄻ":"L", "ㄼ":"L", "ㄽ":"L", "ㄾ":"L", "ㄿ":"L", "ㅀ":"L",
    "ㅁ":"M",
    "ㅂ":"P", "ㅄ":"P", "ㅍ":"P",
    "ㅇ":"NG",
}


def is_hangul_syllable(char: str) -> bool:
    return len(char) == 1 and "가" <= char <= "힣"


def decompose_hangul(char: str) -> dict[str, str] | None:
    if not is_hangul_syllable(char):
        return None

    code = ord(char) - ord("가")
    cho_index = code // 588
    jung_index = (code % 588) // 28
    jong_index = code % 28

    return {
        "char": char,
        "onset": CHOSEONG[cho_index],
        "vowel": JUNGSEONG[jung_index],
        "coda": JONGSEONG[jong_index],
    }


def extract_hangul_syllables(text: str) -> list[str]:
    return [char for char in text if is_hangul_syllable(char)]


def count_syllables(text: str) -> int:
    korean_count = len(extract_hangul_syllables(text))
    non_korean = re.sub(r"[가-힣]", " ", text)
    latin_or_number_count = len(re.findall(r"[A-Za-z0-9]+", non_korean))
    return korean_count + latin_or_number_count


def make_rhyme_key(text: str, tail_length: int = 2) -> dict[str, Any]:
    syllables = extract_hangul_syllables(text)

    if not syllables:
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        tail = tokens[-1] if tokens else ""
        return {
            "surface_tail": tail,
            "exact": tail,
            "vowel": tail,
            "family": tail,
        }

    tail = syllables[-tail_length:]
    decomposed = [decompose_hangul(char) for char in tail]
    decomposed = [item for item in decomposed if item is not None]

    exact = "-".join(
        f"{item['vowel']}{item['coda'] or 'Ø'}"
        for item in decomposed
    )

    vowel = "-".join(item["vowel"] for item in decomposed)

    family = "-".join(
        f"{VOWEL_FAMILY.get(item['vowel'], item['vowel'])}"
        f"/{CODA_FAMILY.get(item['coda'], item['coda'])}"
        for item in decomposed
    )

    return {
        "surface_tail": "".join(tail),
        "exact": exact,
        "vowel": vowel,
        "family": family,
    }


def analyze_korean(text: str) -> dict[str, Any]:
    tokens = kiwi.tokenize(text)

    morphemes = []
    for index, token in enumerate(tokens):
        morphemes.append({
            "index": index,
            "form": token.form,
            "tag": token.tag,
            "start": token.start,
            "length": token.len,
            "end": token.start + token.len,
            "is_content": token.tag.startswith(CONTENT_POS_PREFIXES),
            "is_function": token.tag.startswith(FUNCTION_POS_PREFIXES),
        })

    eojeols = []
    for match in re.finditer(r"\S+", text):
        surface = match.group(0)
        start = match.start()
        end = match.end()

        included = [
            morpheme for morpheme in morphemes
            if morpheme["start"] < end and morpheme["end"] > start
        ]

        eojeols.append({
            "surface": surface,
            "start": start,
            "end": end,
            "syllable_count": count_syllables(surface),
            "morphemes": included,
            "rhyme": make_rhyme_key(surface),
        })

    return {
        "text": text,
        "morphemes": morphemes,
        "eojeols": eojeols,
        "syllable_count": count_syllables(text),
        "rhyme": make_rhyme_key(text),
    }


def split_score(left: str, right: str, boundary_tag: str | None) -> float:
    left_count = count_syllables(left)
    right_count = count_syllables(right)

    balance_penalty = abs(left_count - right_count)

    # 조사나 어미 직후는 자연스러운 호흡 경계일 가능성이 높음
    boundary_bonus = 0.0
    if boundary_tag:
        if boundary_tag.startswith("E"):
            boundary_bonus = 2.0
        elif boundary_tag.startswith("J"):
            boundary_bonus = 1.0

    punctuation_bonus = 1.5 if re.search(r"[,/;|]$", left.strip()) else 0.0

    return balance_penalty - boundary_bonus - punctuation_bonus


def split_lyric_korean(text: str, threshold: int = SPLIT_THRESHOLD) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return [""]

    if count_syllables(text) < threshold:
        return [text]

    analysis = analyze_korean(text)
    eojeols = analysis["eojeols"]

    if len(eojeols) < 2:
        return [text]

    candidates = []

    for split_index in range(1, len(eojeols)):
        left_end = eojeols[split_index - 1]["end"]
        left = text[:left_end].strip(" ,/;|")
        right = text[left_end:].strip(" ,/;|")

        previous_morphemes = eojeols[split_index - 1]["morphemes"]
        boundary_tag = previous_morphemes[-1]["tag"] if previous_morphemes else None

        score = split_score(left, right, boundary_tag)
        candidates.append((score, left, right))

    candidates.sort(key=lambda item: item[0])
    _, left, right = candidates[0]

    return [left, right]


def find_covering_morpheme(
    absolute_char_index: int,
    morphemes: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for morpheme in morphemes:
        if morpheme["start"] <= absolute_char_index < morpheme["end"]:
            return morpheme
    return None


def build_syllable_units(text: str) -> list[dict[str, Any]]:
    analysis = analyze_korean(text)
    morphemes = analysis["morphemes"]

    syllables = []

    for char_index, char in enumerate(text):
        if not is_hangul_syllable(char):
            continue

        morpheme = find_covering_morpheme(char_index, morphemes)
        tag = morpheme["tag"] if morpheme else None

        if tag and tag.startswith(CONTENT_POS_PREFIXES):
            stress = 1.0
            role = "content"
        elif tag and tag.startswith(FUNCTION_POS_PREFIXES):
            stress = 0.55
            role = "function"
        else:
            stress = 0.75
            role = "other"

        syllables.append({
            "text": char,
            "char_index": char_index,
            "morpheme": morpheme["form"] if morpheme else None,
            # 형태소의 실제 시작 위치로 단어 시작을 구분한다.
            "is_word_start": bool(morpheme and char_index == morpheme["start"]),
            "pos": tag,
            "role": role,
            "stress": stress,
            "phonology": decompose_hangul(char),
        })

    # 어절 첫 음절 약간 강조
    eojeol_start_positions = {
        eojeol["start"]
        for eojeol in analysis["eojeols"]
    }

    for syllable in syllables:
        if syllable["char_index"] in eojeol_start_positions:
            syllable["stress"] = min(1.2, syllable["stress"] + 0.15)

    # 구절 마지막 라임 음절 강조
    if syllables:
        syllables[-1]["stress"] = min(1.3, syllables[-1]["stress"] + 0.25)

    return syllables


def group_grid_by_bar(
    absolute_grid: list[dict[str, Any]]
) -> dict[int, list[dict[str, Any]]]:
    grouped = defaultdict(list)

    for item in absolute_grid:
        grouped[int(item["bar"])].append(item)

    for bar_number in grouped:
        grouped[bar_number].sort(key=lambda item: int(item["slot"]))

    return dict(grouped)


def estimate_slot_duration(absolute_grid: list[dict[str, Any]]) -> float:
    intervals = []

    for current, following in zip(absolute_grid, absolute_grid[1:]):
        interval = (
            float(following["time_sec"])
            - float(current["time_sec"])
        )

        if interval > 0:
            intervals.append(interval)

    if not intervals:
        raise ValueError("slot 길이를 계산할 수 없습니다.")

    intervals.sort()
    middle = len(intervals) // 2

    if len(intervals) % 2 == 1:
        return intervals[middle]

    return (intervals[middle - 1] + intervals[middle]) / 2


def allocate_syllables_to_slots(
    text: str,
    start_slot: int,
    end_slot_exclusive: int,
) -> list[dict[str, Any]]:
    syllables = build_syllable_units(text)
    available_slots = end_slot_exclusive - start_slot

    if not syllables:
        return []

    # 한 slot보다 많은 음절은 겹치지 않게 배치할 수 없다.
    if len(syllables) > available_slots:
        raise ValueError(
            f"overflow: {len(syllables)} syllables for {available_slots} slots"
        )

    total_weight = sum(max(item["stress"], 0.1) for item in syllables)

    raw_durations = [
        available_slots * max(item["stress"], 0.1) / total_weight
        for item in syllables
    ]

    durations = [max(1, int(math.floor(value))) for value in raw_durations]

    current_total = sum(durations)

    # 초과 시 낮은 강세 음절부터 줄이기
    while current_total > available_slots:
        candidates = [
            index for index, duration in enumerate(durations)
            if duration > 1
        ]

        if not candidates:
            break

        target = min(
            candidates,
            key=lambda index: syllables[index]["stress"]
        )
        durations[target] -= 1
        current_total -= 1

    # 남는 slot은 높은 강세 음절부터 추가
    while current_total < available_slots:
        target = max(
            range(len(syllables)),
            key=lambda index: (
                raw_durations[index] - durations[index],
                syllables[index]["stress"],
            )
        )
        durations[target] += 1
        current_total += 1

    result = []
    cursor = start_slot

    for index, (syllable, duration) in enumerate(
        zip(syllables, durations)
    ):
        result.append({
            **syllable,
            "syllable_index": index,
            "start_slot_in_bar": cursor,
            "duration_slots": duration,
            "end_slot_in_bar_exclusive": cursor + duration,
        })
        cursor += duration

    return result


def assign_pitch(
    syllables: list[dict[str, Any]],
    is_last_segment: bool,
) -> None:
    """
    규칙 기반 Pitch Accent 할당
    
    규칙:
    1. 내용어(NN*, VV*, VA*, MAG, SL) 첫 음절 → MIDI 62
    2. 조사(J*), 어미(E*), 접사(X*), 기호(S*) → 항상 60
    3. stress >= 1.2 → MIDI + 1 (최대 63)
    4. 구절 마지막 음절(마지막 segment) → MIDI 58
    """
    if not syllables:
        return
    
    # 기본값: 모든 음절 MIDI 60
    for syllable in syllables:
        syllable["midi_note"] = 60
        syllable["is_slur"] = False
    
    for syllable in syllables:
        pos = syllable["pos"]
        
        # 규칙 2: 조사, 어미, 접사, 기호는 항상 60
        if pos and (
            pos.startswith("J") or
            pos.startswith("E") or
            (pos.startswith("X") and not pos.startswith("XR")) or
            pos.startswith("S")
        ):
            continue
        
        # 규칙 1: 내용어의 실제 첫 음절만 62. NP는 예시의 '난/내'를 위해 포함.
        if syllable["is_word_start"] and pos and (
            pos.startswith(("NN", "VV", "VA", "MAG", "SL", "XR", "NP"))
        ):
            syllable["midi_note"] = 62
    
    # 규칙 3: stress >= 1.2이면 +1 (최대 63)
    for syllable in syllables:
        if syllable["stress"] >= 1.2:
            syllable["midi_note"] = min(63, syllable["midi_note"] + 1)
    
    # 규칙 4: 마지막 segment의 마지막 음절 → 58
    if is_last_segment and syllables:
        syllables[-1]["midi_note"] = 58


def make_segment(
    segment_id: str,
    line_index: int,
    bar_number: int,
    segment_index: int,
    text: str,
    bar_grid: list[dict[str, Any]],
    start_slot_in_bar: int,
    end_slot_in_bar_exclusive: int,
    slot_duration: float,
    is_last_segment: bool,
) -> dict[str, Any]:
    analysis = analyze_korean(text)
    available_slots = end_slot_in_bar_exclusive - start_slot_in_bar
    overflow_error = None
    try:
        syllables = allocate_syllables_to_slots(
            text, start_slot_in_bar, end_slot_in_bar_exclusive
        )
    except ValueError as error:
        # 불가능한 배치는 시간축에 쓰지 않아 다음 segment와 충돌하지 않는다.
        syllables = []
        overflow_error = str(error)
    # 배치가 끝난 음절에 pitch 정보를 추가한다.
    assign_pitch(syllables, is_last_segment=is_last_segment)

    # bar 내부 slot(0~15) 기준으로 grid를 조회한다.
    grid_by_slot = {index: item for index, item in enumerate(bar_grid)}
    for syllable in syllables:
        start_grid = grid_by_slot[syllable["start_slot_in_bar"]]
        syllable["start_absolute_slot"] = int(start_grid["slot"])
        syllable["end_absolute_slot_exclusive"] = (
            syllable["start_absolute_slot"] + syllable["duration_slots"]
        )
        syllable["start_sec"] = float(start_grid["time_sec"])
        syllable["end_sec"] = (
            syllable["start_sec"] + syllable["duration_slots"] * slot_duration
        )

    start_grid = grid_by_slot[start_slot_in_bar]
    return {
        "segment_id": segment_id,
        "line_index": line_index,
        "bar": bar_number,
        "segment_index": segment_index,
        "text": text,
        "morphemes": analysis["morphemes"],
        "eojeols": analysis["eojeols"],
        "rhyme": analysis["rhyme"],
        "syllable_count": analysis["syllable_count"],
        "syllables": syllables,
        "unallocated_syllables": analysis["syllable_count"] if overflow_error else 0,
        "overflow_error": overflow_error,
        "start_slot_in_bar": start_slot_in_bar,
        "end_slot_in_bar_exclusive": end_slot_in_bar_exclusive,
        "start_absolute_slot": int(start_grid["slot"]),
        "end_absolute_slot_exclusive": int(start_grid["slot"]) + available_slots,
        "available_slots": available_slots,
        "start_sec": float(start_grid["time_sec"]),
        "end_sec": float(start_grid["time_sec"]) + available_slots * slot_duration,
        "allocated_duration_sec": available_slots * slot_duration,
        "density": "overflow" if overflow_error else "normal",
        "density_ratio": analysis["syllable_count"] / available_slots if available_slots else 0,
    }


def create_rest_events(
    syllables: list[dict[str, Any]],
    bar_grid: list[dict[str, Any]],
    slot_duration: float,
) -> list[dict[str, Any]]:
    rests, cursor = [], 0
    for syllable in sorted(syllables, key=lambda item: item["start_slot_in_bar"]):
        start = syllable["start_slot_in_bar"]
        if cursor < start:
            grid = bar_grid[cursor]
            rests.append({
                "type": "rest",
                "start_absolute_slot": int(grid["slot"]),
                "end_absolute_slot_exclusive": int(grid["slot"]) + start - cursor,
                "duration_slots": start - cursor,
                "start_sec": float(grid["time_sec"]),
                "end_sec": float(grid["time_sec"]) + (start - cursor) * slot_duration,
            })
        cursor = max(cursor, syllable["end_slot_in_bar_exclusive"])
    if cursor < len(bar_grid):
        grid = bar_grid[cursor]
        rests.append({
            "type": "rest",
            "start_absolute_slot": int(grid["slot"]),
            "end_absolute_slot_exclusive": int(grid["slot"]) + len(bar_grid) - cursor,
            "duration_slots": len(bar_grid) - cursor,
            "start_sec": float(grid["time_sec"]),
            "end_sec": float(grid["time_sec"]) + (len(bar_grid) - cursor) * slot_duration,
        })
    return rests


def validate_flow_plan(flow_plan: dict[str, Any]) -> dict[str, Any]:
    errors, warnings = [], []
    for line in flow_plan["lines"]:
        expected = line["analysis"]["syllable_count"]
        placed = [s for segment in line["segments"] for s in segment["syllables"]]
        if len(placed) != expected:
            errors.append(f"line {line['line_index']}: expected {expected}, placed {len(placed)}")
        ordered = sorted(placed, key=lambda item: item["start_absolute_slot"])
        previous_end = None
        for syllable in ordered:
            start, end = syllable["start_absolute_slot"], syllable["end_absolute_slot_exclusive"]
            if start < 0 or end < 0 or start >= end:
                errors.append(f"invalid syllable range: {syllable['text']}")
            if not 0 <= syllable["start_slot_in_bar"] < syllable["end_slot_in_bar_exclusive"] <= 16:
                errors.append(f"bar range overflow: {syllable['text']}")
            if previous_end is not None and start < previous_end:
                errors.append(f"syllable overlap in bar {line['bar']}")
            previous_end = max(previous_end or end, end)
            if not 58 <= syllable["midi_note"] <= 63:
                errors.append(f"invalid MIDI note: {syllable['midi_note']}")
        for segment in line["segments"]:
            if segment["density"] == "overflow":
                warnings.append(f"{segment['segment_id']}: {segment['overflow_error']}")
            elif segment["syllables"] and (
                segment["syllables"][-1]["end_absolute_slot_exclusive"]
#                 != segment["end_absolute_slot_exclusive"]
            ):
                errors.append(f"segment end mismatch: {segment['segment_id']}")
        for rest in line.get("rests", []):
            if rest["start_absolute_slot"] >= rest["end_absolute_slot_exclusive"]:
                errors.append(f"invalid rest range in bar {line['bar']}")
            for syllable in placed:
                if (rest["start_absolute_slot"] < syllable["end_absolute_slot_exclusive"]
                    and syllable["start_absolute_slot"] < rest["end_absolute_slot_exclusive"]):
                    errors.append(f"rest overlaps syllable in bar {line['bar']}")
                    break
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def rebalance_two_segments(text: str, segments: list[str]) -> list[str]:
    # 8+8 slot layout에 맞는 어절 경계를 우선 선택해 overflow를 예방한다.
    if len(segments) != 2:
        return segments
    eojeols = analyze_korean(text)["eojeols"]
    candidates = []
    for split_index in range(1, len(eojeols)):
        left = text[:eojeols[split_index - 1]["end"]].strip()
        right = text[eojeols[split_index]["start"]:].strip()
        left_count, right_count = count_syllables(left), count_syllables(right)
        if left_count <= 8 and right_count <= 8:
            candidates.append((abs(left_count - right_count), left, right))
    if not candidates:
        return segments
    _, left, right = min(candidates, key=lambda item: item[0])
    return [left, right]


def create_flow_plan(
    lyrics: list[str],
    beat_analysis: dict[str, Any],
    verse_start_bar: int,
    split_threshold: int,
) -> dict[str, Any]:
    absolute_grid = beat_analysis["absolute_grid"]
    grid_by_bar = group_grid_by_bar(absolute_grid)
    slot_duration = estimate_slot_duration(absolute_grid)
    lines, warnings = [], []

    for line_index, original_text in enumerate(lyrics, start=1):
        bar_number = verse_start_bar + line_index - 1
        bar_grid = grid_by_bar[bar_number]
        lyric_segments = split_lyric_korean(original_text, split_threshold)
        lyric_segments = rebalance_two_segments(original_text, lyric_segments)
        layouts = [ONE_SEGMENT_LAYOUT] if len(lyric_segments) == 1 else TWO_SEGMENT_LAYOUT
        segments = []

        for segment_index, (segment_text, layout) in enumerate(zip(lyric_segments, layouts), start=1):
            start_slot, end_slot = layout
            segment = make_segment(
                segment_id=f"bar{bar_number:03d}_seg{segment_index:02d}",
                line_index=line_index, bar_number=bar_number, segment_index=segment_index,
                text=segment_text, bar_grid=bar_grid, start_slot_in_bar=start_slot,
                end_slot_in_bar_exclusive=end_slot, slot_duration=slot_duration,
                is_last_segment=(segment_index == len(lyric_segments)),
            )
            if segment["density"] == "overflow":
                warnings.append({
                    "type": "overflow", "segment_id": segment["segment_id"],
                    "syllables": segment["syllable_count"],
                    "available_slots": segment["available_slots"],
                })
            segments.append(segment)

        # 마디 전체를 기준으로 음절 사이의 빈 slot을 중립 rest로 보존한다.
        bar_syllables = [s for segment in segments for s in segment["syllables"]]
        rests = create_rest_events(bar_syllables, bar_grid, slot_duration)
        lines.append({
            "line_index": line_index, "bar": bar_number,
            "original_text": original_text, "analysis": analyze_korean(original_text),
            "split_count": len(lyric_segments), "segments": segments, "rests": rests,
        })

    bpm = beat_analysis["bpm"]["fixed_integer"]
    flow_plan = {
        "metadata": {
            "source_audio_file": beat_analysis["audio_file"],
            "time_signature": beat_analysis["time_signature"],
            "bpm": bpm, "downbeat_offset_sec": beat_analysis["downbeat_offset_sec"],
            "language": "ko", "morph_analyzer": "kiwipiepy",
            "rhyme_method": "hangul_vowel_coda_family",
            "stress_method": "pos_and_phrase_boundary_heuristic",
            "slot_unit": "16th_note", "slot_duration_sec": slot_duration,
            "verse_start_bar": verse_start_bar,
            "verse_end_bar": verse_start_bar + len(lyrics) - 1,
            "line_count": len(lines),
            "segment_count": sum(len(line["segments"]) for line in lines),
            "split_threshold_syllables": split_threshold,
        },
        "lines": lines, "warnings": warnings,
    }
    # 생성 직후 타임라인과 MIDI 범위를 자동 검증한다.
    flow_plan["validation"] = validate_flow_plan(flow_plan)
    return flow_plan


def main() -> None:
    beat_analysis = load_json(BEAT_ANALYSIS_PATH)
    lyric_data = load_json(LYRICS_PATH)
    lyrics = lyric_data["lyrics"]
    verse_start_bar = lyric_data.get("verse_start_bar", VERSE_START_BAR)
    flow_plan = create_flow_plan(
        lyrics=lyrics,
        beat_analysis=beat_analysis,
        verse_start_bar=verse_start_bar,
        split_threshold=SPLIT_THRESHOLD,
    )

    save_json(flow_plan, OUTPUT_PATH)
    ds_output_path = OUTPUT_PATH.with_name("diffsinger_input.ds")
    ds_sections = export_diffsinger(flow_plan)
    save_json(flow_plan, OUTPUT_PATH)
    with ds_output_path.open("w", encoding="utf-8") as file:
        json.dump(ds_sections, file, ensure_ascii=False, indent=2)

    print(f"Flow plan: {OUTPUT_PATH.resolve()}")
    print(f"DiffSinger input: {ds_output_path.resolve()}")


if __name__ == "__main__":
    main()


