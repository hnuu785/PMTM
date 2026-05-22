import pandas as pd
from datasets import Dataset
import re

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


def format_audio(row: pd.Series) -> str:
    return (
        f"BPM: {row['bpm']:.0f} | "
        f"에너지: {row['energy']:.2f} | "
        f"댄서빌리티: {row['danceability']:.2f} | "
        f"라우드니스: {row['loudness']:.1f}dB | "
        f"밸런스: {row['valence']:.2f}"
    )


def format_chunk(artist: str, audio: str, chunk: list[str]) -> str:
    n = len(chunk)
    body = "\n".join(chunk)
    return f"아티스트: {artist}\n{audio}\n[Verse {n}마디]\n{body}\n[End]"


def prepare(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        artist = str(row["artist"])
        lyrics = str(row.get("lyrics", ""))
        lines = clean_lines(lyrics)

        if len(lines) < 8:
            continue

        audio = format_audio(row)
        sizes = [16, 8] if len(lines) >= 16 else [8]
        seen: set[str] = set()

        for size in sizes:
            for chunk in make_chunks(lines, size):
                text = format_chunk(artist, audio, chunk)
                if text not in seen:
                    seen.add(text)
                    records.append({"text": text})

    return records


def main():
    df = pd.read_csv(DATA_PATH)
    records = prepare(df)
    dataset = Dataset.from_list(records)
    dataset.to_json(OUTPUT_PATH, force_ascii=False)
    print(f"저장 완료: {OUTPUT_PATH}  ({len(records)}개 샘플)")


if __name__ == "__main__":
    main()
