"""Merged Korean rap flow planner.

Flow-A supplies G2P, POTG phonemes, timing output, and CLI conventions.
Flow-B supplies Kiwi analysis, rhyme/stress features, slot allocation, MIDI
pitch, and micro timing. This module keeps the existing planners unchanged and
connects their responsibilities through an explicit alignment model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from flow_planner import get_bpm, hangul_to_phonemes, load_lyrics, normalize_korean


SLOTS_PER_BAR = 16
SPLIT_THRESHOLD = 13
MICRO_OFFSET_MAX_MS = 15.0
SNARE_CONFIDENCE_MIN = 0.65
MICRO_OFFSET_RULES = {
    "boom_bap": {"normal": 0.00, "stress": 0.06, "snare": 0.08},
    "trap": {"normal": 0.00, "stress": -0.05, "snare": -0.06},
}

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
VOWEL_FAMILY = {
    "ㅏ": "A", "ㅑ": "A", "ㅘ": "A",
    "ㅐ": "AE", "ㅒ": "AE", "ㅙ": "AE",
    "ㅓ": "EO", "ㅕ": "EO", "ㅝ": "EO",
    "ㅔ": "E", "ㅖ": "E", "ㅞ": "E",
    "ㅗ": "O", "ㅛ": "O", "ㅚ": "O",
    "ㅜ": "U", "ㅠ": "U", "ㅟ": "U",
    "ㅡ": "EU", "ㅢ": "EU", "ㅣ": "I",
}
CODA_FAMILY = {
    "": "OPEN",
    "ㄱ": "K", "ㄲ": "K", "ㄳ": "K", "ㅋ": "K",
    "ㄴ": "N", "ㄵ": "N", "ㄶ": "N",
    "ㄷ": "T", "ㅅ": "T", "ㅆ": "T", "ㅈ": "T", "ㅊ": "T", "ㅌ": "T", "ㅎ": "T",
    "ㄹ": "L", "ㄺ": "L", "ㄻ": "L", "ㄼ": "L", "ㄽ": "L", "ㄾ": "L", "ㄿ": "L", "ㅀ": "L",
    "ㅁ": "M", "ㅂ": "P", "ㅄ": "P", "ㅍ": "P", "ㅇ": "NG",
}


def _get_kiwi() -> Any:
    try:
        from kiwipiepy import Kiwi
    except ImportError as exc:
        raise RuntimeError(
            "Kiwi 분석을 위해 requirements.txt의 kiwipiepy를 설치해야 합니다."
        ) from exc
    return Kiwi()


def _is_hangul(char: str) -> bool:
    return len(char) == 1 and "가" <= char <= "힣"


def _hangul_chars(text: str) -> list[tuple[int, str]]:
    return [(index, char) for index, char in enumerate(text) if _is_hangul(char)]


def _decompose_hangul(char: str) -> tuple[str, str, str]:
    code = ord(char) - 0xAC00
    return (
        CHOSEONG[code // 588],
        JUNGSEONG[(code % 588) // 28],
        JONGSEONG[code % 28],
    )


def make_rhyme_key(text: str, tail_length: int = 2) -> dict[str, str]:
    tail = [char for _, char in _hangul_chars(text)][-tail_length:]
    decomposed = [_decompose_hangul(char) for char in tail]
    return {
        "surfaceTail": "".join(tail),
        "exact": "-".join(f"{vowel}{coda or 'Ø'}" for _, vowel, coda in decomposed),
        "vowel": "-".join(vowel for _, vowel, _ in decomposed),
        "family": "-".join(
            f"{VOWEL_FAMILY.get(vowel, vowel)}/{CODA_FAMILY.get(coda, coda)}"
            for _, vowel, coda in decomposed
        ),
    }


def analyze_korean(text: str, kiwi: Any | None = None) -> dict[str, Any]:
    kiwi = kiwi or _get_kiwi()
    morphemes = []
    for index, token in enumerate(kiwi.tokenize(text)):
        end = token.start + token.len
        morphemes.append(
            {
                "id": index,
                "form": token.form,
                "tag": token.tag,
                "start": token.start,
                "end": end,
                "isContent": token.tag.startswith(CONTENT_POS_PREFIXES),
                "isFunction": token.tag.startswith(FUNCTION_POS_PREFIXES),
            }
        )

    eojeols = []
    for index, match in enumerate(re.finditer(r"\S+", text)):
        included = [
            item["id"]
            for item in morphemes
            if item["start"] < match.end() and item["end"] > match.start()
        ]
        eojeols.append(
            {
                "id": index,
                "surface": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "morphemeIds": included,
            }
        )
    return {"text": text, "morphemes": morphemes, "eojeols": eojeols}


def _source_syllables(text: str, analysis: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for source_index, (char_index, char) in enumerate(_hangul_chars(text)):
        morpheme = next(
            (
                item
                for item in analysis["morphemes"]
                if item["start"] <= char_index < item["end"]
            ),
            None,
        )
        eojeol = next(
            (
                item
                for item in analysis["eojeols"]
                if item["start"] <= char_index < item["end"]
            ),
            None,
        )
        tag = morpheme["tag"] if morpheme else None
        if tag and tag.startswith(CONTENT_POS_PREFIXES):
            stress, role = 1.0, "content"
        elif tag and tag.startswith(FUNCTION_POS_PREFIXES):
            stress, role = 0.55, "function"
        else:
            stress, role = 0.75, "other"

        reasons = [role]
        is_word_start = bool(eojeol and char_index == eojeol["start"])
        if is_word_start:
            stress = min(1.2, stress + 0.15)
            reasons.append("eojeol_start")

        result.append(
            {
                "sourceIndex": source_index,
                "charIndex": char_index,
                "text": char,
                "morphemeId": morpheme["id"] if morpheme else None,
                "eojeolId": eojeol["id"] if eojeol else None,
                "pos": tag,
                "role": role,
                "stress": stress,
                "stressReasons": reasons,
                "isWordStart": is_word_start,
            }
        )
    return result


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


def align_original_pronunciation(
    text: str,
    pronunciation: str,
    analysis: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = _source_syllables(text, analysis)
    spoken = [char for _, char in _hangul_chars(pronunciation)]
    operations = _align_sequences([item["text"] for item in sources], spoken)
    alignment = []
    units = []

    for operation, source_index, spoken_index in operations:
        source = sources[source_index] if source_index is not None else None
        pronounced = spoken[spoken_index] if spoken_index is not None else None
        alignment.append(
            {
                "operation": operation,
                "originalIndex": source_index,
                "pronouncedIndex": spoken_index,
                "original": source["text"] if source else None,
                "pronounced": pronounced,
                "morphemeId": source["morphemeId"] if source else None,
                "eojeolId": source["eojeolId"] if source else None,
            }
        )
        if pronounced is None:
            continue

        inherited = source
        if inherited is None and sources:
            previous = [item for item in units if item.get("sourceIndex") is not None]
            inherited = sources[previous[-1]["sourceIndex"]] if previous else sources[0]
        units.append(
            {
                "sourceIndex": inherited["sourceIndex"] if inherited else None,
                "original": inherited["text"] if inherited else None,
                "pronounced": pronounced,
                "morphemeId": inherited["morphemeId"] if inherited else None,
                "eojeolId": inherited["eojeolId"] if inherited else None,
                "pos": inherited["pos"] if inherited else None,
                "role": inherited["role"] if inherited else "other",
                "stress": inherited["stress"] if inherited else 0.75,
                "stressReasons": list(inherited["stressReasons"]) if inherited else ["other"],
                "isWordStart": inherited["isWordStart"] if inherited else False,
            }
        )

    if units:
        units[-1]["stress"] = min(1.3, float(units[-1]["stress"]) + 0.25)
        units[-1]["stressReasons"].append("phrase_final_rhyme")
    return alignment, units


def split_lyric(text: str, threshold: int = SPLIT_THRESHOLD) -> list[str]:
    if len(_hangul_chars(text)) < threshold:
        return [text.strip()]
    matches = list(re.finditer(r"\S+", text))
    candidates = []
    for index in range(1, len(matches)):
        left = text[: matches[index - 1].end()].strip()
        right = text[matches[index].start() :].strip()
        left_count, right_count = len(_hangul_chars(left)), len(_hangul_chars(right))
        overflow = max(0, left_count - 8) + max(0, right_count - 8)
        candidates.append((overflow, abs(left_count - right_count), left, right))
    if not candidates:
        return [text.strip()]
    _, _, left, right = min(candidates, key=lambda item: (item[0], item[1]))
    return [left, right]


def _allocate_slot_durations(units: list[dict[str, Any]], available_slots: int) -> list[int]:
    if len(units) > available_slots:
        raise ValueError(f"overflow: {len(units)} syllables for {available_slots} slots")
    weights = [max(float(unit["stress"]), 0.1) for unit in units]
    total = sum(weights)
    raw = [available_slots * weight / total for weight in weights]
    durations = [max(1, math.floor(value)) for value in raw]
    while sum(durations) > available_slots:
        candidates = [index for index, value in enumerate(durations) if value > 1]
        if not candidates:
            break
        durations[min(candidates, key=lambda index: weights[index])] -= 1
    while sum(durations) < available_slots:
        target = max(
            range(len(units)),
            key=lambda index: (raw[index] - durations[index], weights[index]),
        )
        durations[target] += 1
    return durations


def assign_pitch(units: list[dict[str, Any]], is_last_segment: bool) -> None:
    for unit in units:
        unit["midiNote"] = 60
        pos = unit["pos"]
        is_function = pos and (
            pos.startswith("J")
            or pos.startswith("E")
            or (pos.startswith("X") and not pos.startswith("XR"))
            or pos.startswith("S")
        )
        if not is_function and unit["isWordStart"] and pos and pos.startswith(
            ("NN", "VV", "VA", "MAG", "SL", "XR", "NP")
        ):
            unit["midiNote"] = 62
        if float(unit["stress"]) >= 1.2:
            unit["midiNote"] = min(63, unit["midiNote"] + 1)
    if is_last_segment and units:
        units[-1]["midiNote"] = 58


def _complete_bars(
    analysis: dict[str, Any],
    count: int,
    verse_start_bar: int,
) -> list[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in analysis.get("absolute_grid", []):
        grouped.setdefault(int(entry["bar"]), []).append(entry)
    bars = []
    for bar_number in range(verse_start_bar, verse_start_bar + count):
        entries = sorted(grouped.get(bar_number, []), key=lambda item: int(item["slot"]))
        if len(entries) != SLOTS_PER_BAR:
            raise ValueError(f"{bar_number}마디에 완전한 16슬롯 grid가 필요합니다.")
        bars.append(entries)
    return bars


def _slot_duration(analysis: dict[str, Any]) -> float:
    grid = analysis["absolute_grid"]
    intervals = sorted(
        float(right["time_sec"]) - float(left["time_sec"])
        for left, right in zip(grid, grid[1:])
        if float(right["time_sec"]) > float(left["time_sec"])
    )
    return intervals[len(intervals) // 2] if intervals else 60.0 / get_bpm(analysis) / 4.0


def build_snare_map(analysis: dict[str, Any]) -> dict[int, dict[str, float]]:
    result = {}
    events = analysis.get("snare_detection", {}).get("events", [])
    for event in events if isinstance(events, list) else []:
        grid_slot = event.get("grid_slot")
        confidence = event.get("confidence", 0.0)
        if not isinstance(grid_slot, int) or not isinstance(confidence, (int, float)):
            continue
        if confidence < SNARE_CONFIDENCE_MIN:
            continue
        candidate = {
            "confidence": float(confidence),
            "originalTimeSec": float(event.get("original_time", 0.0)),
            "snappedTimeSec": float(event.get("snapped_time", 0.0)),
        }
        current = result.get(grid_slot)
        if current is None or candidate["confidence"] > current["confidence"]:
            result[grid_slot] = candidate
    return result


def calculate_micro_offset(
    unit: dict[str, Any],
    genre: str,
    slot_duration: float,
    snare_map: dict[int, dict[str, float]],
) -> tuple[float, str, dict[str, float] | None]:
    """Preserve Flow-B's genre/stress/snare micro-timing rule."""
    rules = MICRO_OFFSET_RULES[genre]
    snare = snare_map.get(int(unit["absoluteSlot"]))
    if snare is not None:
        ratio, reason = rules["snare"], "snare"
    elif float(unit["stress"]) >= 1.0:
        ratio, reason = rules["stress"], "stress"
    else:
        ratio, reason = rules["normal"], "none"
    offset_ms = slot_duration * 1000.0 * ratio
    return (
        round(max(-MICRO_OFFSET_MAX_MS, min(MICRO_OFFSET_MAX_MS, offset_ms)), 3),
        reason,
        snare,
    )


def _midi_to_note_name(midi_note: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return names[midi_note % 12] + str(midi_note // 12 - 1)


def _split_phoneme_duration(duration: float, count: int) -> list[float]:
    weights = {1: [1.0], 2: [0.22, 0.58], 3: [0.22, 0.58, 0.20]}[count]
    total = sum(weights)
    values = [round(duration * weight / total, 6) for weight in weights]
    values[-1] = round(values[-1] + duration - sum(values), 6)
    return values


def build_flow_plan(
    analysis: dict[str, Any],
    lyrics: list[str],
    genre: str,
    voicebank: str,
    *,
    verse_start_bar: int = 1,
    normalizer: Callable[[str], str] = normalize_korean,
    kiwi: Any | None = None,
) -> dict[str, Any]:
    if genre not in MICRO_OFFSET_RULES:
        raise ValueError("genre는 boom_bap 또는 trap이어야 합니다.")
    kiwi = kiwi or _get_kiwi()
    if normalizer is normalize_korean:
        try:
            from g2pk2 import G2p
        except ImportError as exc:
            raise RuntimeError(
                "한국어 발음 변환을 위해 requirements.txt의 g2pk2를 설치해야 합니다."
            ) from exc
        normalizer = G2p()
    bars = _complete_bars(analysis, len(lyrics), verse_start_bar)
    slot_duration = _slot_duration(analysis)
    snare_map = build_snare_map(analysis)
    planned_bars = []

    for line_index, (text, grid) in enumerate(zip(lyrics, bars), start=1):
        segment_texts = split_lyric(text)
        layouts = [(0, 16)] if len(segment_texts) == 1 else [(0, 8), (8, 16)]
        segments = []
        placements = []

        for segment_index, (segment_text, layout) in enumerate(
            zip(segment_texts, layouts), start=1
        ):
            linguistic = analyze_korean(segment_text, kiwi=kiwi)
            pronunciation = normalizer(segment_text)
            alignment, units = align_original_pronunciation(
                segment_text, pronunciation, linguistic
            )
            duration_slots = _allocate_slot_durations(units, layout[1] - layout[0])
            assign_pitch(units, segment_index == len(segment_texts))
            cursor = layout[0]
            segment_placements = []

            for unit, slots in zip(units, duration_slots):
                grid_entry = grid[cursor]
                grid_start = float(grid_entry["time_sec"])
                original_duration = slots * slot_duration
                placement = {
                    **unit,
                    "segmentIndex": segment_index,
                    "slotInBar": cursor,
                    "absoluteSlot": int(grid_entry["slot"]),
                    "durationSlots": slots,
                    "gridStartSec": round(grid_start, 6),
                    "gridEndSec": round(grid_start + original_duration, 6),
                }
                offset_ms, reason, snare = calculate_micro_offset(
                    placement, genre, slot_duration, snare_map
                )
                placement["microOffsetMs"] = offset_ms
                placement["microOffsetReason"] = reason
                placement["finalStartSec"] = round(grid_start + offset_ms / 1000.0, 6)
                placement["snareReference"] = snare
                segment_placements.append(placement)
                placements.append(placement)
                cursor += slots

            segments.append(
                {
                    "segmentIndex": segment_index,
                    "originalText": segment_text,
                    "pronunciation": pronunciation,
                    "rhyme": make_rhyme_key(pronunciation),
                    "morphemes": linguistic["morphemes"],
                    "eojeols": linguistic["eojeols"],
                    "alignment": alignment,
                    "startSlotInBar": layout[0],
                    "endSlotInBarExclusive": layout[1],
                    "placements": segment_placements,
                }
            )

        placements.sort(key=lambda item: item["finalStartSec"])
        bar_start = float(grid[0]["time_sec"])
        bar_end = bar_start + slot_duration * SLOTS_PER_BAR
        rests = []
        cursor_time = bar_start
        for index, placement in enumerate(placements):
            start = max(bar_start, float(placement["finalStartSec"]))
            next_start = (
                max(start, float(placements[index + 1]["finalStartSec"]))
                if index + 1 < len(placements)
                else bar_end
            )
            if start > cursor_time + 0.000001:
                rests.append(
                    {
                        "startSec": round(cursor_time, 6),
                        "endSec": round(start, 6),
                        "durationSec": round(start - cursor_time, 6),
                    }
                )
            original_duration = float(placement["gridEndSec"]) - float(
                placement["gridStartSec"]
            )
            duration = max(0.001, min(original_duration, next_start - start))
            end = min(bar_end, start + duration)
            placement["finalStartSec"] = round(start, 6)
            placement["finalEndSec"] = round(end, 6)
            placement["finalDurationSec"] = round(end - start, 6)
            placement["restAfterSec"] = round(max(0.0, next_start - end), 6)
            placement["phonemes"] = hangul_to_phonemes(placement["pronounced"])
            placement["phonemeDurations"] = _split_phoneme_duration(
                placement["finalDurationSec"], len(placement["phonemes"])
            )
            cursor_time = end
        if cursor_time < bar_end - 0.000001:
            rests.append(
                {
                    "startSec": round(cursor_time, 6),
                    "endSec": round(bar_end, 6),
                    "durationSec": round(bar_end - cursor_time, 6),
                }
            )

        planned_bars.append(
            {
                "bar": int(grid[0]["bar"]),
                "lineIndex": line_index,
                "originalText": text,
                "startSec": round(bar_start, 6),
                "endSec": round(bar_end, 6),
                "segments": segments,
                "placements": placements,
                "rests": rests,
            }
        )

    return {
        "version": 2,
        "metadata": {
            "sourceAudio": analysis.get("audio_file"),
            "bpm": get_bpm(analysis),
            "timeSignature": analysis.get("time_signature", "4/4"),
            "genre": genre,
            "voicebank": voicebank,
            "gridResolution": 16,
            "slotDurationSec": round(slot_duration, 6),
            "verseStartBar": verse_start_bar,
            "snareConfidenceMin": SNARE_CONFIDENCE_MIN,
            "snareEventCount": len(snare_map),
            "microOffsetRules": MICRO_OFFSET_RULES[genre],
            "microOffsetMaxMs": MICRO_OFFSET_MAX_MS,
        },
        "bars": planned_bars,
    }


def to_diffsinger_ds(plan: dict[str, Any]) -> list[dict[str, str]]:
    sections = []
    for bar in plan["bars"]:
        ph_seq: list[str] = []
        ph_dur: list[float] = []
        ph_num: list[int] = []
        note_seq: list[str] = []
        note_dur: list[float] = []
        note_slur: list[int] = []
        cursor = float(bar["startSec"])
        pronounced_text = "".join(
            placement["pronounced"] for placement in bar["placements"]
        )

        for placement in bar["placements"]:
            start = float(placement["finalStartSec"])
            if start > cursor + 0.000001:
                gap = start - cursor
                ph_seq.append("SP")
                ph_dur.append(gap)
                ph_num.append(1)
                note_seq.append("rest")
                note_dur.append(gap)
                note_slur.append(0)
            ph_seq.extend(placement["phonemes"])
            ph_dur.extend(placement["phonemeDurations"])
            ph_num.append(len(placement["phonemes"]))
            note_seq.append(_midi_to_note_name(int(placement["midiNote"])))
            note_dur.append(float(placement["finalDurationSec"]))
            note_slur.append(0)
            cursor = float(placement["finalEndSec"])

        if cursor < float(bar["endSec"]) - 0.000001:
            gap = float(bar["endSec"]) - cursor
            ph_seq.append("SP")
            ph_dur.append(gap)
            ph_num.append(1)
            note_seq.append("rest")
            note_dur.append(gap)
            note_slur.append(0)

        sections.append(
            {
                "offset": f'{float(bar["startSec"]):.6f}',
                "text": pronounced_text,
                "original_text": str(bar["originalText"]),
                "ph_seq": " ".join(ph_seq),
                "ph_dur": " ".join(f"{value:.6f}" for value in ph_dur),
                "ph_num": " ".join(str(value) for value in ph_num),
                "note_seq": " ".join(note_seq),
                "note_dur": " ".join(f"{value:.6f}" for value in note_dur),
                "note_slur": " ".join(str(value) for value in note_slur),
            }
        )
    return sections


def validate_outputs(plan: dict[str, Any], score: list[dict[str, str]]) -> None:
    if len(plan["bars"]) != len(score):
        raise ValueError("flow plan과 DiffSinger section 수가 다릅니다.")
    for bar, section in zip(plan["bars"], score):
        placements = bar["placements"]
        for left, right in zip(placements, placements[1:]):
            if float(left["finalEndSec"]) > float(right["finalStartSec"]) + 0.000001:
                raise ValueError(f"{bar['bar']}마디 음절이 겹칩니다.")
        phones = section["ph_seq"].split()
        durations = [float(value) for value in section["ph_dur"].split()]
        ph_num = [int(value) for value in section["ph_num"].split()]
        if len(phones) != len(durations) or sum(ph_num) != len(phones):
            raise ValueError(f"{bar['bar']}마디 DiffSinger 음소 매핑이 맞지 않습니다.")
        total = sum(durations)
        expected = float(bar["endSec"]) - float(bar["startSec"])
        if not math.isclose(total, expected, abs_tol=1e-4):
            raise ValueError(f"{bar['bar']}마디 duration 합이 마디 길이와 다릅니다.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merged Korean DiffSinger Flow Planner")
    parser.add_argument("--beat-analysis", required=True, type=Path)
    parser.add_argument("--lyrics", required=True, type=Path)
    parser.add_argument("--genre", required=True, choices=tuple(MICRO_OFFSET_RULES))
    parser.add_argument("--voicebank", default="potg")
    parser.add_argument("--verse-start-bar", type=int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = json.loads(args.beat_analysis.read_text(encoding="utf-8"))
    plan = build_flow_plan(
        analysis,
        load_lyrics(args.lyrics),
        args.genre,
        args.voicebank,
        verse_start_bar=args.verse_start_bar,
    )
    score = to_diffsinger_ds(plan)
    validate_outputs(plan, score)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flow_path = args.output_dir / "flow-plan.json"
    score_path = args.output_dir / "score.ds"
    flow_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    score_path.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    print(flow_path)
    print(score_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
