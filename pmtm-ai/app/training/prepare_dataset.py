import pandas as pd
from datasets import Dataset
import re

from app.lyric_prompts import DEFAULT_GENRE, DEFAULT_MOOD, TARGET_BARS, build_api_messages
from app.paths import DATA_DIR

DATA_PATH = str(DATA_DIR / "merged_final_dataset_analyzed.csv")
OUTPUT_PATH = str(DATA_DIR / "prepared_dataset.jsonl")


def clean_lines(lyrics: str) -> list[str]:
    lines = lyrics.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        # 섹션 태그([Verse], [Hook] 등)와 빈 줄 제거
        if not line or re.match(r"^\[.*\]$", line):
            continue
        cleaned.append(line)
    return cleaned


def make_chunks(lines: list[str], chunk_size: int) -> list[list[str]]:
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines) - chunk_size + 1, chunk_size)]
    # 꼬리 줄 손실 방지: 마지막 chunk와 겹치더라도 끝 chunk 추가
    remainder = len(lines) % chunk_size
    if remainder >= chunk_size // 2 and len(lines) >= chunk_size:
        tail = lines[-chunk_size:]
        if tail != chunks[-1]:
            chunks.append(tail)
    return chunks


def format_assistant(chunk: list[str]) -> str:
    body = "\n".join(chunk)
    return f"{body}\n[End]"


def prepare(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        lyrics = str(row.get("lyrics", ""))
        lines = clean_lines(lyrics)

        if len(lines) < TARGET_BARS:
            continue

        seen: set[str] = set()

        for chunk in make_chunks(lines, TARGET_BARS):
            assistant = format_assistant(chunk)
            if assistant not in seen:
                seen.add(assistant)
                records.append({
                    "messages": build_api_messages(
                        bpm=float(row["bpm"]),
                        genre=DEFAULT_GENRE,
                        mood=DEFAULT_MOOD,
                        bars=TARGET_BARS,
                        assistant=assistant,
                    )
                })

    return records


def main():
    df = pd.read_csv(DATA_PATH)
    records = prepare(df)
    dataset = Dataset.from_list(records)
    dataset.to_json(OUTPUT_PATH, force_ascii=False)
    print(f"저장 완료: {OUTPUT_PATH}  ({len(records)}개 샘플)")


if __name__ == "__main__":
    main()
