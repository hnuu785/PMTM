import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.lyric_prompts import TARGET_BARS, build_messages
from app.genre_rules import get_genre_rules
from app.paths import DATA_DIR
from app.rhyme_scoring.phonetics_utils import get_phonemes, count_syllables
from app.rhyme_scoring.rhyme_engine import calculate_rhyme_density

DATA_PATH = DATA_DIR / "merged_final_dataset_analyzed.csv"
OUTPUT_PATH = DATA_DIR / "prepared_dataset_v3.jsonl"

MIN_RHYME_SCORE = 0.35
MIN_KOREAN_RATIO = 0.35
MIN_MEAN_LINE_LENGTH = 6
MAX_MEAN_LINE_LENGTH = 28
MAX_LINE_LENGTH_STDEV = 9
MAX_SHORT_LINES = 1
MAX_ENDING_WORD_COUNT = 2
MAX_REPEATED_BIGRAMS = 5
MAX_REPEATED_TRIGRAMS = 1




ADLIB_PAREN_RE = re.compile(
    r"\s*\((?:yeah|uh|ah|oh|ay|ye|ooh|woah|skrrt|skrt|yah|ey|billi|milli|heyy?)\)\s*",
    re.IGNORECASE,
)
ADLIB_END_RE = re.compile(
    r"(?:[\s,\.!\?]+(?:\([^\)]+\)|yeah|uh|ah|oh|ay|ye|ooh|woah|skrrt|skrt|yah|ey))+[\s,\.!\?]*$",
    re.IGNORECASE,
)
ONLY_ADLIB_RE = re.compile(
    r"^(?:[\s,\.!\?\(\)]|(?:yeah|uh|ah|oh|ay|ye|ooh|woah|skrrt|skrt|yah|ey))+$",
    re.IGNORECASE,
)


def clean_lines(lyrics: str) -> list[str]:
    cleaned = []
    for line in lyrics.split("\n"):
        line = line.strip()
        if not line or re.match(r"^\[.*\]$", line) or ONLY_ADLIB_RE.match(line):
            continue
        line = ADLIB_PAREN_RE.sub(" ", line).strip()
        line = ADLIB_END_RE.sub("", line).strip()
        line = re.sub(r"\s+", " ", line).strip()
        if not line or ONLY_ADLIB_RE.match(line):
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
    return count_syllables(line)


def korean_ratio(lines: list[str]) -> float:
    text = "\n".join(lines)
    letters = re.findall(r"[A-Za-z가-힣]", text)
    if not letters:
        return 0.0
    return len(re.findall(r"[가-힣]", text)) / len(letters)





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


def chunk_features(lines: list[str], bpm: float | None = None) -> dict[str, float | int]:
    lengths = [line_length(line) for line in lines]
    endings = [ending_word(line) for line in lines]
    normalized_lines = [normalize_text(line) for line in lines]
    rhyme_score = calculate_rhyme_density(lines, bpm=bpm)
    return {
        "rhyme_score": rhyme_score,
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


def build_record(lines: list[str], genre: str, bpm: float) -> dict:
    formatted_lines = []
    for i, line in enumerate(lines, 1):
        syllables = count_syllables(line)
        formatted_lines.append(f"{i}. ({syllables}음절) {line}")

    assistant_content = "\n".join(formatted_lines)

    return {
        "messages": build_messages(
            bpm=bpm,
            bars=len(lines),
            assistant=assistant_content
        )
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

            if bpm <= 0:
                stats["bad_bpm"] += 1
                continue

            genre, target_lines, min_s, max_s = get_genre_rules(bpm)

            for chunk in make_chunks(clean_lines(lyrics), chunk_size=target_lines):
                stats["candidate_chunks"] += 1

                chunk_key = "\n".join(normalize_text(line) for line in chunk)
                if chunk_key in seen_chunks:
                    stats["duplicate_chunk"] += 1
                    continue
                seen_chunks.add(chunk_key)

                features = chunk_features(chunk, bpm=bpm)
                reason = rejection_reason(features)
                if reason:
                    stats[reason] += 1
                    continue

                # 장르별 ±1음절 허용 범위 (붐뱁 7~17음절, 트랩 5~15음절) 필터링
                syllables_list = [count_syllables(line) for line in chunk]
                if not all(min_s - 1 <= s <= max_s + 1 for s in syllables_list):
                    stats["syllable_mismatch"] += 1
                    continue


                records.append(build_record(chunk, genre, bpm))

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
