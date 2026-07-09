import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.lyric_prompts import TARGET_BARS, build_api_user_prompt
from app.paths import DATA_DIR

DATA_PATH = DATA_DIR / "merged_final_dataset_analyzed.csv"
OUTPUT_PATH = DATA_DIR / "prepared_dataset_v2.jsonl"

MIN_RHYME_SCORE = 0.22
MIN_KOREAN_RATIO = 0.35
MIN_MEAN_LINE_LENGTH = 7
MAX_MEAN_LINE_LENGTH = 28
MAX_LINE_LENGTH_STDEV = 9
MAX_SHORT_LINES = 1
MAX_ENDING_WORD_COUNT = 2
MAX_REPEATED_BIGRAMS = 5
MAX_REPEATED_TRIGRAMS = 1

VOWELS = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
    "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
CODAS = [
    None, "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ",
    "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ",
    "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
VOWEL_GROUPS = {
    "ㅐ": "ㅔ", "ㅔ": "ㅐ", "ㅖ": "ㅔ", "ㅒ": "ㅐ",
    "ㅗ": "ㅜ", "ㅜ": "ㅗ",
    "ㅡ": "ㅣ", "ㅣ": "ㅡ",
    "ㅏ": "ㅑ", "ㅑ": "ㅏ",
    "ㅓ": "ㅕ", "ㅕ": "ㅓ",
    "ㅙ": "ㅐ", "ㅚ": "ㅔ", "ㅞ": "ㅔ",
}
CODA_GROUPS = {
    "ㄴ": "nasal", "ㅁ": "nasal", "ㅇ": "nasal",
    "ㄱ": "stop", "ㅂ": "stop", "ㄷ": "stop", "ㅅ": "stop",
    "ㅋ": "stop", "ㅌ": "stop", "ㅍ": "stop",
    "ㄹ": "liquid",
}


def clean_lines(lyrics: str) -> list[str]:
    cleaned = []
    for line in lyrics.split("\n"):
        line = line.strip()
        if not line or re.match(r"^\[.*\]$", line):
            continue
        cleaned.append(line)
    return cleaned


def make_chunks(lines: list[str], chunk_size: int = TARGET_BARS) -> list[list[str]]:
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines) - chunk_size + 1, chunk_size)]
    remainder = len(lines) % chunk_size
    if remainder >= chunk_size // 2 and len(lines) >= chunk_size:
        tail = lines[-chunk_size:]
        if not chunks or tail != chunks[-1]:
            chunks.append(tail)
    return chunks


def normalize_text(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"[가-힣A-Za-z0-9]+", text.lower())


def ending_word(line: str) -> str:
    line_tokens = tokens(line)
    return line_tokens[-1] if line_tokens else ""


def line_length(line: str) -> int:
    return len(re.findall(r"[가-힣A-Za-z0-9]", line))


def korean_ratio(lines: list[str]) -> float:
    text = "\n".join(lines)
    letters = re.findall(r"[A-Za-z가-힣]", text)
    if not letters:
        return 0.0
    return len(re.findall(r"[가-힣]", text)) / len(letters)


def hangul_phonemes(text: str) -> list[tuple[str, str | None]]:
    phonemes = []
    for ch in text:
        code = ord(ch)
        if not 0xAC00 <= code <= 0xD7A3:
            continue
        idx = code - 0xAC00
        vowel = VOWELS[(idx % 588) // 28]
        coda = CODAS[idx % 28]
        phonemes.append((vowel, coda))
    return phonemes


def syllable_rhyme_score(a: tuple[str, str | None], b: tuple[str, str | None]) -> float:
    v1, c1 = a
    v2, c2 = b
    if v1 == v2:
        vowel_score = 1.0
    elif VOWEL_GROUPS.get(v1) == v2:
        vowel_score = 0.8
    else:
        vowel_score = 0.0

    if c1 == c2:
        coda_score = 1.0
    elif c1 and c2 and CODA_GROUPS.get(c1) == CODA_GROUPS.get(c2):
        coda_score = 0.7
    elif not c1 and not c2:
        coda_score = 1.0
    else:
        coda_score = 0.0

    return vowel_score * 0.8 + coda_score * 0.2


def line_rhyme_score(line1: str, line2: str) -> float:
    p1 = hangul_phonemes(line1)
    p2 = hangul_phonemes(line2)
    if not p1 or not p2:
        return 0.0

    weights = [1.0, 0.5, 0.3]
    count = min(len(p1), len(p2), len(weights))
    total = 0.0
    for i in range(1, count + 1):
        total += syllable_rhyme_score(p1[-i], p2[-i]) * weights[i - 1]
    return total / sum(weights[:count])


def repeated_ngram_count(lines: list[str], n: int) -> int:
    sequence = []
    for line in lines:
        sequence.extend(tokens(line))
        sequence.append("<LINE>")
    ngrams = [
        tuple(sequence[i:i + n])
        for i in range(len(sequence) - n + 1)
        if "<LINE>" not in sequence[i:i + n]
    ]
    return sum(count - 1 for count in Counter(ngrams).values() if count > 1)


def chunk_features(lines: list[str]) -> dict[str, float | int]:
    lengths = [line_length(line) for line in lines]
    endings = [ending_word(line) for line in lines]
    normalized_lines = [normalize_text(line) for line in lines]
    rhyme_scores = [line_rhyme_score(lines[i], lines[i + 1]) for i in range(len(lines) - 1)]
    return {
        "rhyme_score": sum(rhyme_scores) / len(rhyme_scores),
        "duplicate_lines": len(lines) - len(set(normalized_lines)),
        "max_ending_word_count": max(Counter(endings).values()) if endings else len(lines),
        "korean_ratio": korean_ratio(lines),
        "short_lines": sum(1 for length in lengths if length <= 3),
        "mean_line_length": statistics.mean(lengths),
        "line_length_stdev": statistics.pstdev(lengths),
        "repeated_bigrams": repeated_ngram_count(lines, 2),
        "repeated_trigrams": repeated_ngram_count(lines, 3),
    }


def rejection_reason(features: dict[str, float | int]) -> str | None:
    if features["rhyme_score"] < MIN_RHYME_SCORE:
        return "low_rhyme"
    if features["duplicate_lines"] > 0:
        return "line_repeat"
    if features["max_ending_word_count"] > MAX_ENDING_WORD_COUNT:
        return "ending_repeat"
    if features["korean_ratio"] < MIN_KOREAN_RATIO:
        return "low_korean_ratio"
    if features["short_lines"] > MAX_SHORT_LINES:
        return "too_many_short_lines"
    if not MIN_MEAN_LINE_LENGTH <= features["mean_line_length"] <= MAX_MEAN_LINE_LENGTH:
        return "bad_mean_line_length"
    if features["line_length_stdev"] > MAX_LINE_LENGTH_STDEV:
        return "uneven_breathing"
    if features["repeated_trigrams"] > MAX_REPEATED_TRIGRAMS:
        return "phrase_repeat"
    if features["repeated_bigrams"] > MAX_REPEATED_BIGRAMS:
        return "word_repeat"
    return None


def build_user_prompt(bpm: float) -> str:
    return build_api_user_prompt(bpm=bpm, bars=TARGET_BARS)


def build_record(lines: list[str], bpm: float) -> dict:
    return {
        "messages": [
            {"role": "user", "content": build_user_prompt(bpm)},
            {"role": "assistant", "content": "\n".join(lines) + "\n[End]"},
        ]
    }


def prepare() -> tuple[list[dict], Counter]:
    records = []
    stats = Counter()
    seen_chunks: set[str] = set()

    with DATA_PATH.open(encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            lyrics = row.get("lyrics", "")
            try:
                bpm = float(row.get("bpm", "0") or 0)
            except ValueError:
                bpm = 0.0

            for chunk in make_chunks(clean_lines(lyrics)):
                stats["candidate_chunks"] += 1
                if bpm <= 0:
                    stats["bad_bpm"] += 1
                    continue

                chunk_key = "\n".join(normalize_text(line) for line in chunk)
                if chunk_key in seen_chunks:
                    stats["duplicate_chunk"] += 1
                    continue
                seen_chunks.add(chunk_key)

                features = chunk_features(chunk)
                reason = rejection_reason(features)
                if reason:
                    stats[reason] += 1
                    continue

                records.append(build_record(chunk, bpm))

    stats["kept"] = len(records)
    return records, stats


def main() -> None:
    records, stats = prepare()
    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"saved: {OUTPUT_PATH} ({len(records)} samples)")
    print("stats:")
    for key, value in stats.most_common():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
