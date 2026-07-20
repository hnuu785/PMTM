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
from app.rhyme_scoring.phonetics_utils import get_phonemes
from app.rhyme_scoring.loanword_stopwords import ENGLISH_RHYME_STOPWORDS
from app.rhyme_scoring.rhyme_engine import get_line_rhyme_score

DATA_PATH = DATA_DIR / "merged_final_dataset_analyzed.csv"
OUTPUT_PATH = DATA_DIR / "prepared_dataset_v3.jsonl"

MIN_RHYME_SCORE = 0.22
MIN_KOREAN_RATIO = 0.35
MIN_MEAN_LINE_LENGTH = 7
MAX_MEAN_LINE_LENGTH = 28
MAX_LINE_LENGTH_STDEV = 9
MAX_SHORT_LINES = 1
MAX_ENDING_WORD_COUNT = 2
MAX_REPEATED_BIGRAMS = 5
MAX_REPEATED_TRIGRAMS = 1




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


def int_to_korean(n: int) -> str:
    if n == 0:
        return "영"
    units = ["", "십", "백", "천"]
    big_units = ["", "만", "억", "조"]
    digits = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
    
    num_str = str(n)
    parts = []
    rev_str = num_str[::-1]
    for i in range(0, len(rev_str), 4):
        chunk = rev_str[i:i+4]
        chunk_val = ""
        for j, digit in enumerate(chunk):
            d = int(digit)
            if d > 0:
                if d == 1 and j > 0:
                    digit_name = ""
                else:
                    digit_name = digits[d]
                chunk_val = digit_name + units[j] + chunk_val
        if chunk_val:
            parts.append(chunk_val + big_units[i // 4])
    
    return "".join(reversed(parts))


def count_syllables(line: str) -> int:
    tokens = re.findall(r"[가-힣]+|[A-Za-z]+|\d+", line)
    total_syllables = 0
    for tok in tokens:
        if "가" <= tok[0] <= "힣":
            total_syllables += len(tok)
        elif tok.lower() in ENGLISH_RHYME_STOPWORDS:
            total_syllables += 1
        elif tok.isdigit():
            try:
                korean_num = int_to_korean(int(tok))
                total_syllables += len(korean_num)
            except Exception:
                total_syllables += len(tok)
        else:
            try:
                phonemes = get_phonemes(tok)
                if phonemes:
                    total_syllables += len(phonemes)
                else:
                    vowels = re.findall(r"[aeiouyAEIOUY]", tok)
                    total_syllables += max(1, len(vowels))
            except Exception:
                vowels = re.findall(r"[aeiouyAEIOUY]", tok)
                total_syllables += max(1, len(vowels))
    return total_syllables


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


def chunk_features(lines: list[str]) -> dict[str, float | int]:
    lengths = [line_length(line) for line in lines]
    endings = [ending_word(line) for line in lines]
    normalized_lines = [normalize_text(line) for line in lines]
    rhyme_scores = [get_line_rhyme_score(lines[i], lines[i + 1]) for i in range(len(lines) - 1)]
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


def build_user_prompt(genre: str) -> str:
    bpm = 90.0 if genre == "붐뱁" else 140.0
    return build_api_user_prompt(bpm=bpm, bars=TARGET_BARS)


def build_record(lines: list[str], genre: str) -> dict:
    formatted_lines = []
    for i, line in enumerate(lines, 1):
        formatted_lines.append(f"{i}. {line}")

    assistant_content = "\n".join(formatted_lines)

    return {
        "messages": [
            {"role": "user", "content": build_user_prompt(genre)},
            {"role": "assistant", "content": assistant_content},
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

                judgment_bpm = bpm * 2.0 if 60.0 <= bpm < 80.0 else bpm
                genre = "붐뱁" if judgment_bpm < 110 else "트랩"
                
                # 엄격 모드: 모든 마디의 음절 수가 반드시 지침 범위(±1음절 허용) 내여야 함
                syllables_list = [count_syllables(line) for line in chunk]
                if genre == "붐뱁":
                    if not all(9 <= s <= 15 for s in syllables_list):
                        stats["syllable_mismatch"] += 1
                        continue
                else: # 트랩
                    if not all(13 <= s <= 19 for s in syllables_list):
                        stats["syllable_mismatch"] += 1
                        continue

                records.append(build_record(chunk, genre))

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
