"""Export an existing Flow Planner timeline as DiffSinger sections.

The exporter deliberately consumes the planner's timestamps, MIDI notes, slur
flags, and rests as-is; it never derives them from BPM or slots.
"""

from __future__ import annotations

import math
import re
from typing import Any

try:  # g2pk2 is preferred when the notebook environment provides it.
    from g2pk2 import G2p
except ImportError:  # The maintained PyPI package exposes the same G2p API.
    from g2pk import G2p


CHOSEONG = ("ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")
JUNGSEONG = ("ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ")
JONGSEONG = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")

ONSET_PHONEMES = {
    "ㄱ": "g", "ㄲ": "kk", "ㄴ": "n", "ㄷ": "d", "ㄸ": "tt", "ㄹ": "rx", "ㅁ": "m", "ㅂ": "b", "ㅃ": "pp", "ㅅ": "s", "ㅆ": "sc", "ㅇ": "", "ㅈ": "jh", "ㅉ": "jj", "ㅊ": "ch", "ㅋ": "k", "ㅌ": "t", "ㅍ": "p", "ㅎ": "hh",
}
VOWEL_PHONEMES = {
    "ㅏ": "a", "ㅐ": "e", "ㅑ": "ia", "ㅒ": "ie", "ㅓ": "eo", "ㅔ": "e", "ㅕ": "ieo", "ㅖ": "ie", "ㅗ": "o", "ㅘ": "oa", "ㅙ": "oe", "ㅚ": "oe", "ㅛ": "io", "ㅜ": "u", "ㅝ": "uo", "ㅞ": "oe", "ㅟ": "ui", "ㅠ": "iu", "ㅡ": "eu", "ㅢ": "ui", "ㅣ": "i",
}
CODA_PHONEMES = {
    "": "", "ㄱ": "kcl", "ㄲ": "kcl", "ㄳ": "kcl", "ㄴ": "n", "ㄵ": "n", "ㄶ": "n", "ㄷ": "tcl", "ㄹ": "l", "ㄺ": "kcl", "ㄻ": "m", "ㄼ": "pcl", "ㄽ": "l", "ㄾ": "l", "ㄿ": "pcl", "ㅀ": "l", "ㅁ": "m", "ㅂ": "pcl", "ㅄ": "pcl", "ㅅ": "tcl", "ㅆ": "tcl", "ㅇ": "ng", "ㅈ": "tcl", "ㅊ": "tcl", "ㅋ": "kcl", "ㅌ": "tcl", "ㅍ": "pcl", "ㅎ": "tcl",
}
SILENT_ONSET_COMPOUNDS = {"ㅘ": ("w", "a"), "ㅙ": ("w", "e"), "ㅚ": ("w", "e"), "ㅝ": ("w", "o"), "ㅞ": ("w", "e"), "ㅟ": ("w", "i"), "ㅢ": ("", "ui")}
PALATAL_VOWELS = {"ㅑ", "ㅕ", "ㅛ", "ㅠ", "ㅒ", "ㅖ", "ㅟ"}
VOWELS = frozenset(VOWEL_PHONEMES.values())
LETTER_TO_POTG = {"a": "a", "b": "b", "c": "k", "d": "d", "e": "e", "f": "f", "g": "g", "h": "hh", "i": "i", "j": "jh", "k": "k", "l": "l", "m": "m", "n": "n", "o": "o", "p": "p", "q": "k", "r": "r", "s": "s", "t": "t", "u": "u", "v": "v", "w": "w", "x": "k", "y": "i", "z": "z"}


def midi_to_note_name(midi_note: int) -> str:
    if not isinstance(midi_note, int) or not 0 <= midi_note <= 127:
        raise ValueError(f"MIDI note must be an integer from 0 to 127: {midi_note!r}")
    return ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[midi_note % 12] + str(midi_note // 12 - 1)


def syllable_to_potg_phonemes(syllable: str) -> list[str]:
    """Map one Hangul syllable (or one Latin character) to POTG phonemes."""
    if len(syllable) != 1:
        raise ValueError(f"Expected one syllable, got {syllable!r}")
    if "가" <= syllable <= "힣":
        code = ord(syllable) - ord("가")
        onset, vowel, coda = CHOSEONG[code // 588], JUNGSEONG[(code % 588) // 28], JONGSEONG[code % 28]
        onset_phone, vowel_phone = ONSET_PHONEMES[onset], VOWEL_PHONEMES[vowel]
        if onset == "ㅇ" and vowel in SILENT_ONSET_COMPOUNDS:
            onset_phone, vowel_phone = SILENT_ONSET_COMPOUNDS[vowel]
        elif onset == "ㅅ" and vowel in PALATAL_VOWELS:
            onset_phone = "sh"
        elif onset == "ㅆ" and vowel in PALATAL_VOWELS:
            onset_phone = "sy"
        return [phone for phone in (onset_phone, vowel_phone, CODA_PHONEMES[coda]) if phone]
    if syllable.isascii() and syllable.isalpha():
        return [LETTER_TO_POTG[syllable.lower()]]
    raise ValueError(f"Unsupported pronunciation syllable: {syllable!r}")


def allocate_phoneme_durations(phonemes: list[str], syllable_duration: float) -> list[float]:
    if not phonemes or not math.isfinite(syllable_duration) or syllable_duration <= 0:
        raise ValueError("A syllable needs one or more phonemes and a positive finite duration")
    weights = [0.58 if phone in VOWELS else 0.30 if phone in {"kcl", "tcl", "pcl", "ng"} else 0.18 for phone in phonemes]
    total = sum(weights)
    durations = [round(syllable_duration * weight / total, 6) for weight in weights]
    durations[-1] = round(durations[-1] + (syllable_duration - sum(durations)), 6)
    if any(duration <= 0 for duration in durations):
        raise ValueError(f"Invalid phoneme durations for {phonemes}: {durations}")
    return durations


def _pronounced_syllables(text: str, g2p: G2p) -> tuple[str, list[str]]:
    pronunciation = g2p(text)
    source = [char for char in text if "가" <= char <= "힣" or char.isascii() and char.isalpha()]
    spoken = [char for char in pronunciation if "가" <= char <= "힣" or char.isascii() and char.isalpha()]
    if len(source) != len(spoken):
        raise ValueError(f"G2P syllable count changed ({len(source)} -> {len(spoken)}) for segment {text!r}; explicit alignment is required")
    return pronunciation, spoken


def build_diffsinger_section(segment: dict[str, Any], *, g2p: G2p | None = None) -> dict[str, str]:
    g2p = g2p or G2p()
    syllables = segment.get("syllables", [])
    if not syllables:
        raise ValueError(f"{_location(segment)} has no allocated syllables")
    pronunciation, spoken = _pronounced_syllables(str(segment["text"]), g2p)
    if len(spoken) != len(syllables):
        raise ValueError(f"{_location(segment)}: planner syllable count {len(syllables)} differs from G2P count {len(spoken)}")

    phonemes: list[str] = []
    durations: list[float] = []
    ph_num: list[int] = []
    note_seq: list[str] = []
    note_dur: list[float] = []
    note_slur: list[str] = []
    for source_syllable, pronounced_syllable in zip(syllables, spoken):
        duration = float(source_syllable["end_sec"]) - float(source_syllable["start_sec"])
        phones = syllable_to_potg_phonemes(pronounced_syllable)
        phone_durations = allocate_phoneme_durations(phones, duration)
        phonemes.extend(phones)
        durations.extend(phone_durations)
        ph_num.append(len(phones))
        note_seq.append(midi_to_note_name(source_syllable["midi_note"]))
        note_dur.append(duration)
        note_slur.append("1" if source_syllable["is_slur"] else "0")

    return {
        "offset": f"{float(syllables[0]['start_sec']):.6f}", "text": pronunciation,
        "original_text": str(segment["text"]), "ph_seq": " ".join(phonemes),
        "ph_dur": " ".join(f"{duration:.6f}" for duration in durations),
        "ph_num": " ".join(map(str, ph_num)), "note_seq": " ".join(note_seq),
        "note_dur": " ".join(f"{duration:.6f}" for duration in note_dur),
        "note_slur": " ".join(note_slur),
    }


def export_diffsinger(flow_plan: dict[str, Any]) -> list[dict[str, str]]:
    g2p = G2p()
    sections = []
    for line in flow_plan["lines"]:
        for segment in line["segments"]:
            if segment.get("syllables"):
                section = build_diffsinger_section(segment, g2p=g2p)
                segment["diffsinger"] = section
                sections.append(section)
    sections.sort(key=lambda section: float(section["offset"]))
    validation = validate_diffsinger(flow_plan, sections)
    flow_plan.setdefault("validation", {})["diffsinger"] = validation
    if not validation["valid"]:
        raise ValueError("DiffSinger validation failed: " + "; ".join(validation["errors"]))
    return sections


def validate_diffsinger(flow_plan: dict[str, Any], sections: list[dict[str, str]], tolerance: float = 1e-4) -> dict[str, Any]:
    errors: list[str] = []
    source_segments = [segment for line in flow_plan["lines"] for segment in line["segments"] if segment.get("syllables")]
    if len(source_segments) != len(sections):
        errors.append(f"section count mismatch: {len(source_segments)} != {len(sections)}")
    for segment, section in zip(source_segments, sections):
        location = _location(segment)
        phones, ph_dur, ph_num = section["ph_seq"].split(), [float(value) for value in section["ph_dur"].split()], [int(value) for value in section["ph_num"].split()]
        notes, note_dur, note_slur = section["note_seq"].split(), [float(value) for value in section["note_dur"].split()], section["note_slur"].split()
        if sum(ph_num) != len(phones): errors.append(f"{location}: ph_num does not match ph_seq")
        if len(phones) != len(ph_dur): errors.append(f"{location}: ph_seq/ph_dur length mismatch")
        if not (len(notes) == len(note_dur) == len(note_slur) == len(segment["syllables"])): errors.append(f"{location}: note field length mismatch")
        if any(value <= 0 for value in ph_dur + note_dur): errors.append(f"{location}: non-positive duration")
        if abs(sum(ph_dur) - sum(note_dur)) > tolerance: errors.append(f"{location}: phoneme/note total duration mismatch")
        for source, count, phone_durations, note_duration, note, slur in zip(segment["syllables"], ph_num, _groups(ph_dur, ph_num), note_dur, notes, note_slur):
            original_duration = float(source["end_sec"]) - float(source["start_sec"])
            if abs(sum(phone_durations) - original_duration) > tolerance or abs(note_duration - original_duration) > tolerance: errors.append(f"{location}: syllable duration changed")
            if note != midi_to_note_name(source["midi_note"]) or slur != ("1" if source["is_slur"] else "0"): errors.append(f"{location}: MIDI or slur changed")
    if any(float(left["offset"]) > float(right["offset"]) for left, right in zip(sections, sections[1:])): errors.append("sections are not sorted by offset")
    return {"valid": not errors, "errors": errors, "warnings": []}


def _groups(values: list[float], counts: list[int]) -> list[list[float]]:
    groups, cursor = [], 0
    for count in counts:
        groups.append(values[cursor:cursor + count]); cursor += count
    return groups


def _location(segment: dict[str, Any]) -> str:
    return f"line_index={segment.get('line_index')}, bar={segment.get('bar')}, segment_id={segment.get('segment_id')}"
