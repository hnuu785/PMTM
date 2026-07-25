from pathlib import Path
import json

import librosa
import numpy as np


# TODO(API): 음원 업로드 API가 생기면 요청으로 받은 파일 경로를 사용한다.
AUDIO_PATH = "90_boombap.mp3"
OUTPUT_DIR = Path("outputs")
SAMPLE_RATE = 44100
HOP_LENGTH = 512
BEATS_PER_BAR = 4


def to_float(value) -> float:
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def build_absolute_grid(bpm: int, offset_sec: float, duration_sec: float) -> list[dict]:
    """BPM과 첫 박을 기준으로 16분음표 격자를 만든다."""
    grid = []
    if bpm <= 0:
        return grid

    beat_interval = 60.0 / bpm
    sixteenth_interval = beat_interval / 4.0
    
    global_slot = 0
    labels = ["1", "e", "&", "a"]
    current_time = offset_sec

    while current_time <= duration_sec:
        beat_index = global_slot // 4
        bar_index = (beat_index // BEATS_PER_BAR) + 1
        beat_in_bar = (beat_index % BEATS_PER_BAR) + 1
        subdivision = global_slot % 4

        grid.append(
            {
                "slot": global_slot,
                "time_sec": round(current_time, 4),
                "bar": bar_index,
                "beat_in_bar": beat_in_bar,
                "subdivision": subdivision,
                "position_label": labels[subdivision],
            }
        )
        global_slot += 1
        current_time = offset_sec + (global_slot * sixteenth_interval)

    return grid


def snap_to_grid(onset_times: np.ndarray, grid: list[dict]) -> list[dict]:
    """실제 온셋을 가장 가까운 16분음표 격자에 맞춘다."""
    snapped_onsets = []
    if not grid or len(onset_times) == 0:
        return snapped_onsets

    grid_times = np.array([g["time_sec"] for g in grid])

    for t in onset_times:
        time_val = float(t)
        closest_idx = int(np.argmin(np.abs(grid_times - time_val)))
        
        snapped_onsets.append({
            "original_time": round(time_val, 4),
            "snapped_time": grid[closest_idx]["time_sec"],
            "grid_slot": grid[closest_idx]["slot"],
            "bar": grid[closest_idx]["bar"],
            "position": f"{grid[closest_idx]['beat_in_bar']}.{grid[closest_idx]['position_label']}"
        })

    return snapped_onsets


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = Path(AUDIO_PATH)

    if not audio_path.exists():
        raise FileNotFoundError(f"MP3 파일을 찾을 수 없습니다: {audio_path.resolve()}")

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError("오디오 데이터가 비어 있습니다.")

    duration_sec = float(librosa.get_duration(y=y, sr=sr))

    # 악기 성격별 타격을 보존하기 위해 저·중·고역 온셋을 분리한다.
    onset_envelope_low = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH, fmax=150
    )
    onset_frames_low = librosa.onset.onset_detect(
        onset_envelope=onset_envelope_low, sr=sr, hop_length=HOP_LENGTH
    )
    onset_times_low = librosa.frames_to_time(onset_frames_low, sr=sr, hop_length=HOP_LENGTH)

    onset_envelope_mid = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH, fmin=150, fmax=3000
    )
    onset_frames_mid = librosa.onset.onset_detect(
        onset_envelope=onset_envelope_mid, sr=sr, hop_length=HOP_LENGTH
    )
    onset_times_mid = librosa.frames_to_time(onset_frames_mid, sr=sr, hop_length=HOP_LENGTH)

    onset_envelope_high = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH, fmin=3000
    )
    onset_frames_high = librosa.onset.onset_detect(
        onset_envelope=onset_envelope_high, sr=sr, hop_length=HOP_LENGTH
    )
    onset_times_high = librosa.frames_to_time(onset_frames_high, sr=sr, hop_length=HOP_LENGTH)

    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    librosa_tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope, sr=sr, hop_length=HOP_LENGTH
    )
    
    raw_bpm = to_float(librosa_tempo)
    # 이후 슬롯 계산은 정수 BPM을 기준으로 한다.
    fixed_bpm = int(round(raw_bpm))

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    if len(beat_times) > 0:
        base_offset = float(beat_times[0])
        # 첫 비트 근처의 킥을 찾으면 실제 첫 박으로 보정한다.
        nearby_kicks = [t for t in onset_times_low if abs(t - base_offset) < 0.2]
        offset_sec = float(nearby_kicks[0]) if nearby_kicks else base_offset
    else:
        offset_sec = 0.0

    absolute_grid = build_absolute_grid(bpm=fixed_bpm, offset_sec=offset_sec, duration_sec=duration_sec)

    snapped_low = snap_to_grid(onset_times_low, absolute_grid)
    snapped_mid = snap_to_grid(onset_times_mid, absolute_grid)
    snapped_high = snap_to_grid(onset_times_high, absolute_grid)

    result = {
        "audio_file": str(audio_path),
        "sample_rate": int(sr),
        "duration_sec": round(duration_sec, 4),
        "time_signature": "4/4",
        "bpm": {
            "librosa_raw": round(raw_bpm, 4),
            "fixed_integer": fixed_bpm
        },
        "downbeat_offset_sec": round(offset_sec, 4),
        "grid_slots_count": len(absolute_grid),
        "absolute_grid": absolute_grid,
        "onsets_quantized": {
            "low_kick": snapped_low,
            "mid_snare": snapped_mid,
            "high_hat": snapped_high
        }
    }

    json_path = OUTPUT_DIR / f"{audio_path.stem}_advanced_analysis.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("Advanced Beat Analysis Complete (Absolute Grid & Quantize)")
    print(f"Audio          : {audio_path.name}")
    print(f"Raw BPM        : {raw_bpm:.2f} -> Fixed BPM: {fixed_bpm}")
    print(f"Offset (Start) : {offset_sec:.4f} sec")
    print(f"Total Grid     : {len(absolute_grid)} slots (16th notes)")
    print(f"Onsets Found   : Low({len(onset_times_low)}), Mid({len(onset_times_mid)}), High({len(onset_times_high)})")
    print(f"JSON Saved     : {json_path.resolve()}")


if __name__ == "__main__":
    main()
