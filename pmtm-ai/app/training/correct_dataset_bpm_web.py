import json
import sys
from pathlib import Path
import pandas as pd

# 파일 경로
DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "merged_final_dataset_analyzed.csv"
JSON_PATH = Path("/Users/cho/.gemini/antigravity/brain/49b4423b-76a4-49b7-814e-1733570187ce/scratch/bpm_search_results.json")

# 기존 수동 보정 13곡 (Why 83 BPM 포함)
MANUAL_BPM_MAP = {
    ("Why", "Supreme Team"): 83.0,
    ("Unknowingly", "Supreme Team"): 111.0,
    ("LOVE STORY", "Epik High"): 94.0,
    ("One (feat. 지선)", "Epik High"): 135.0,
    ("Don't Hate Me", "Epik High"): 155.0,
    ("Eternal Sunshine", "Epik High"): 97.0,
    ("SoulMate (feat. IU)", "ZICO"): 97.0,
    ("Eureka (Feat. Zion.T)", "ZICO"): 120.0,
    ("VENI VIPIC VICI (Feat. DJ Wegun)", "ZICO"): 105.0, # 대소문자 대비
    ("VENI VIDI VICI (Feat. DJ Wegun)", "ZICO"): 105.0,
    ("wish", "CHANGMO"): 130.0,
    ("Uck (Prod. by CHANGMO)", "SUPERBEE"): 140.0,
    ("LET IT BURN (Prod. The Quiett)", "Ahn Byeong Woong"): 92.0,
    ("NO ROOF (feat. Colde, Khakii)", "Ahn Byeong Woong"): 172.0,
}

def clean_str(s: str) -> str:
    # 대소문자, 괄호 안 내용, 특수문자, 띄어쓰기를 지워 유연하게 매칭
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
    s = re.sub(r"[^a-z0-9가-힣]", "", s)
    return s

import re

def main():
    if not DATA_PATH.exists():
        print(f"Error: Dataset not found at {DATA_PATH}")
        sys.exit(1)
    if not JSON_PATH.exists():
        print(f"Error: JSON results not found at {JSON_PATH}")
        sys.exit(1)
        
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print("Loading JSON search results...")
    with JSON_PATH.open("r", encoding="utf-8") as f:
        search_results = json.load(f)
        
    # JSON 파일 데이터 파싱 및 정제 키 매핑 사전 구축
    json_bpm_map = {}
    for key, real_bpm in search_results.items():
        parts = key.split(" - ", 1)
        if len(parts) == 2:
            artist, title = parts[0].strip(), parts[1].strip()
            json_bpm_map[(clean_str(title), clean_str(artist))] = real_bpm
            
    # 기존 수동 맵도 정제 키 매핑에 추가 (Why 83 BPM 보장)
    for (title, artist), real_bpm in MANUAL_BPM_MAP.items():
        json_bpm_map[(clean_str(title), clean_str(artist))] = real_bpm
        
    corrected_count = 0
    corrected_log = []
    
    print("Applying BPM corrections (Manual + Web Search results only)...")
    for idx, row in df.iterrows():
        title = row["title"]
        artist = row["artist"]
        old_bpm = row["bpm"]
        
        match_key = (clean_str(title), clean_str(artist))
        if match_key in json_bpm_map:
            real_bpm = json_bpm_map[match_key]
            
            # 구글 검색 결과 덮어씀 (BPM 값에 오차가 있고 결측치거나 빈 값이면 교정)
            if pd.isna(old_bpm) or abs(old_bpm - real_bpm) > 0.01:
                df.at[idx, "bpm"] = real_bpm
                corrected_log.append(f"{artist} - {title}: {old_bpm} -> {real_bpm:.3f}")
                corrected_count += 1

    print(f"\nTotal Corrected: {corrected_count} rows")
    print("\n--- Corrected Example Logs ---")
    unique_logs = list(set(corrected_log))
    for log in unique_logs[:45]:
        print(f"  {log}")
        
    print("\nSaving corrected dataset...")
    df.to_csv(DATA_PATH, index=False)
    print("Successfully updated dataset with manual + web search results!")

if __name__ == "__main__":
    main()
