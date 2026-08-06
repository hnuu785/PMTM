import os
import sys
import pandas as pd
from tqdm import tqdm

from app.paths import DATA_DIR, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from app.rhyme_scoring.rhyme_engine import analyze_bar_end_rhyme

INPUT_PATH = str(DATA_DIR / "merged_final_dataset_analyzed.csv")
OUTPUT_PATH = str(DATA_DIR / "merged_final_dataset_analyzed.csv")
BACKUP_OLD_COL = "rhyme_coverage_old"


def compute_coverage(lyrics: str) -> float:
    lines = [ln.strip() for ln in str(lyrics).split("\n") if ln.strip()]
    return analyze_bar_end_rhyme(lines).coverage_score


def analyze():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    print(f"Reading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    # 기존 점수를 백업 컬럼으로 옮기고 다시 계산
    if "rhyme_coverage" in df.columns:
        df[BACKUP_OLD_COL] = df["rhyme_coverage"]

    coverages = []
    for lyrics in tqdm(df["lyrics"], total=len(df), desc="rhyme"):
        coverages.append(compute_coverage(lyrics))
    df["rhyme_coverage"] = coverages

    # 상위 10
    top = df.sort_values(by="rhyme_coverage", ascending=False).head(10)
    print("\n" + "=" * 60)
    print("  TOP 10 RHYME COVERAGE")
    print("=" * 60)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        old = r.get(BACKUP_OLD_COL, float("nan"))
        print(f"{i:2}. [{r['artist']:>15}] {r['title']:<30} coverage={r['rhyme_coverage']*100:5.2f}  old={old*100:5.2f}")

    # 분포 비교
    print("\n" + "=" * 60)
    print("  distribution")
    print("=" * 60)
    print(f"  new: mean={df['rhyme_coverage'].mean():.4f}  std={df['rhyme_coverage'].std():.4f}  median={df['rhyme_coverage'].median():.4f}")
    if BACKUP_OLD_COL in df.columns:
        print(f"  old: mean={df[BACKUP_OLD_COL].mean():.4f}  std={df[BACKUP_OLD_COL].std():.4f}  median={df[BACKUP_OLD_COL].median():.4f}")
        delta = df["rhyme_coverage"] - df[BACKUP_OLD_COL]
        print(f"  Δ : mean={delta.mean():+.4f}  std={delta.std():.4f}")

    # 점수 순위 변화 (상위 50 교집합)
    if BACKUP_OLD_COL in df.columns:
        top_new = set(df.sort_values("rhyme_coverage", ascending=False).head(50).index)
        top_old = set(df.sort_values(BACKUP_OLD_COL, ascending=False).head(50).index)
        overlap = len(top_new & top_old)
        print(f"  TOP-50 overlap (new vs old): {overlap}/50")

    # 아티스트별 상위 10
    artist_avg = df.groupby("artist")["rhyme_coverage"].mean().sort_values(ascending=False).head(10)
    print("\n" + "=" * 60)
    print("  TOP 10 ARTISTS")
    print("=" * 60)
    for artist, score in artist_avg.items():
        print(f"  {artist:20} : {score * 100:5.2f}")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    analyze()
