#!/usr/bin/env python3
"""Advanced beat analysis and rule-based snare candidate detection.

Usage from a terminal:
    python beat_analysis.py input.mp3
    python beat_analysis.py input.mp3 --output-dir outputs --snare-threshold 0.70

Usage from Python / FastAPI:
    from beat_analysis import analyze_audio
    result = analyze_audio("input.mp3", save_files=False)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_HOP_LENGTH = 512
DEFAULT_BEATS_PER_BAR = 4
DEFAULT_SNARE_CONFIDENCE_THRESHOLD = 0.65


def to_float(value: Any) -> float:
    """Convert a scalar or NumPy-like value to a Python float."""
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def normalize_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-8:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def build_absolute_grid(
    bpm: float,
    offset_sec: float,
    duration_sec: float,
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
) -> list[dict[str, Any]]:
    """Build an absolute 16th-note grid from BPM and the first beat offset."""
    grid: list[dict[str, Any]] = []
    if bpm <= 0:
        return grid

    beat_interval = 60.0 / bpm
    sixteenth_interval = beat_interval / 4.0
    labels = ["1", "e", "&", "a"]
    global_slot = 0

    while True:
        current_time = offset_sec + global_slot * sixteenth_interval
        if current_time > duration_sec:
            break

        beat_index = global_slot // 4
        bar_index = (beat_index // beats_per_bar) + 1
        beat_in_bar = (beat_index % beats_per_bar) + 1
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

    return grid


def snap_to_grid(
    onset_times: np.ndarray,
    grid: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Snap real onset times to their nearest 16th-note grid positions."""
    snapped_onsets: list[dict[str, Any]] = []
    if not grid or len(onset_times) == 0:
        return snapped_onsets

    grid_times = np.array([item["time_sec"] for item in grid], dtype=float)

    for onset_time in onset_times:
        time_value = float(onset_time)
        closest_index = int(np.argmin(np.abs(grid_times - time_value)))
        item = grid[closest_index]
        snapped_onsets.append(
            {
                "original_time": round(time_value, 4),
                "snapped_time": item["time_sec"],
                "grid_slot": item["slot"],
                "bar": item["bar"],
                "position": f"{item['beat_in_bar']}.{item['position_label']}",
            }
        )

    return snapped_onsets


def nearest_frame_value(feature: np.ndarray, frame: int) -> float:
    if feature.size == 0:
        return 0.0

    safe_frame = int(np.clip(frame, 0, len(feature) - 1))
    return float(feature[safe_frame])


def detect_snare_candidates(
    y_percussive: np.ndarray,
    sr: int,
    mid_onset_frames: np.ndarray,
    absolute_grid: list[dict[str, Any]],
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    confidence_threshold: float = DEFAULT_SNARE_CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Score middle-band onsets as rule-based snare candidates."""
    if len(mid_onset_frames) == 0:
        return []

    stft_magnitude = np.abs(
        librosa.stft(y_percussive, n_fft=2048, hop_length=hop_length)
    )
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)

    low_mask = frequencies < 180
    mid_mask = (frequencies >= 180) & (frequencies < 3000)
    high_mask = frequencies >= 3000

    low_energy = normalize_array(np.mean(stft_magnitude[low_mask], axis=0))
    mid_energy = normalize_array(np.mean(stft_magnitude[mid_mask], axis=0))
    high_energy = normalize_array(np.mean(stft_magnitude[high_mask], axis=0))
    centroid = normalize_array(
        librosa.feature.spectral_centroid(S=stft_magnitude, sr=sr)[0]
    )
    zero_crossing_rate = normalize_array(
        librosa.feature.zero_crossing_rate(
            y_percussive,
            hop_length=hop_length,
        )[0]
    )
    onset_envelope = normalize_array(
        librosa.onset.onset_strength(
            y=y_percussive,
            sr=sr,
            hop_length=hop_length,
        )
    )

    grid_times = np.array(
        [item["time_sec"] for item in absolute_grid],
        dtype=float,
    )
    events: list[dict[str, Any]] = []

    for frame_value in mid_onset_frames:
        frame = int(frame_value)
        time_sec = float(
            librosa.frames_to_time(frame, sr=sr, hop_length=hop_length)
        )

        mid_score = nearest_frame_value(mid_energy, frame)
        high_score = nearest_frame_value(high_energy, frame)
        low_score = nearest_frame_value(low_energy, frame)
        attack_score = nearest_frame_value(onset_envelope, frame)
        centroid_score = nearest_frame_value(centroid, frame)
        noise_score = nearest_frame_value(zero_crossing_rate, frame)

        grid_item: dict[str, Any] | None = None
        grid_distance = 0.0
        backbeat_score = 0.0

        if len(grid_times) > 0:
            grid_index = int(np.argmin(np.abs(grid_times - time_sec)))
            grid_item = absolute_grid[grid_index]
            grid_distance = abs(time_sec - grid_item["time_sec"])

            if (
                grid_item["beat_in_bar"] in (2, 4)
                and grid_item["subdivision"] == 0
            ):
                backbeat_score = 1.0

        raw_score = (
            0.28 * mid_score
            + 0.17 * high_score
            + 0.22 * attack_score
            + 0.10 * centroid_score
            + 0.08 * noise_score
            + 0.15 * backbeat_score
            - 0.12 * low_score
        )
        confidence = float(np.clip(raw_score, 0.0, 1.0))

        event: dict[str, Any] = {
            "original_time": round(time_sec, 4),
            "confidence": round(confidence, 4),
            "is_snare_candidate": confidence >= confidence_threshold,
            "features": {
                "mid_energy": round(mid_score, 4),
                "high_energy": round(high_score, 4),
                "low_energy": round(low_score, 4),
                "onset_strength": round(attack_score, 4),
                "spectral_centroid": round(centroid_score, 4),
                "zero_crossing_rate": round(noise_score, 4),
                "backbeat_bonus": round(backbeat_score, 4),
            },
        }

        if grid_item is not None:
            event.update(
                {
                    "snapped_time": grid_item["time_sec"],
                    "grid_slot": grid_item["slot"],
                    "bar": grid_item["bar"],
                    "position": (
                        f"{grid_item['beat_in_bar']}."
                        f"{grid_item['position_label']}"
                    ),
                    "grid_distance_sec": round(grid_distance, 4),
                }
            )

        events.append(event)

    events.sort(key=lambda item: item["original_time"])
    return events


def analyze_audio(
    audio_path_value: str | Path,
    *,
    output_dir: str | Path = "outputs",
    preview_dir: str | Path | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
    snare_confidence_threshold: float = DEFAULT_SNARE_CONFIDENCE_THRESHOLD,
    save_json: bool = True,
    save_preview: bool = True,
) -> dict[str, Any]:
    """Analyze one audio file and optionally save JSON and preview WAV files.

    This is the main function intended for direct use from a server or API.
    The returned dictionary is JSON serializable and also contains output paths
    in the ``output_files`` field.
    """
    audio_path = Path(audio_path_value).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    preview_output_path = (
        Path(preview_dir).expanduser().resolve()
        if preview_dir is not None
        else output_path
    )

    if not audio_path.exists():
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"오디오 경로가 파일이 아닙니다: {audio_path}")
    if not 0.0 <= snare_confidence_threshold <= 1.0:
        raise ValueError("snare_confidence_threshold는 0.0~1.0이어야 합니다.")
    if sample_rate <= 0 or hop_length <= 0 or beats_per_bar <= 0:
        raise ValueError("sample_rate, hop_length, beats_per_bar는 양수여야 합니다.")

    if save_json:
        output_path.mkdir(parents=True, exist_ok=True)
    if save_preview:
        preview_output_path.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError("오디오 데이터가 비어 있습니다.")

    duration_sec = float(librosa.get_duration(y=y, sr=sr))
    y_percussive = librosa.effects.percussive(
        y,
        hop_length=hop_length,
        margin=2.0,
    )

    onset_envelope_low = librosa.onset.onset_strength(
        y=y_percussive,
        sr=sr,
        hop_length=hop_length,
        fmax=150,
    )
    onset_frames_low = librosa.onset.onset_detect(
        onset_envelope=onset_envelope_low,
        sr=sr,
        hop_length=hop_length,
    )
    onset_times_low = librosa.frames_to_time(
        onset_frames_low,
        sr=sr,
        hop_length=hop_length,
    )

    onset_envelope_mid = librosa.onset.onset_strength(
        y=y_percussive,
        sr=sr,
        hop_length=hop_length,
        fmin=150,
        fmax=3000,
    )
    onset_frames_mid = librosa.onset.onset_detect(
        onset_envelope=onset_envelope_mid,
        sr=sr,
        hop_length=hop_length,
    )
    onset_times_mid = librosa.frames_to_time(
        onset_frames_mid,
        sr=sr,
        hop_length=hop_length,
    )

    onset_envelope_high = librosa.onset.onset_strength(
        y=y_percussive,
        sr=sr,
        hop_length=hop_length,
        fmin=3000,
    )
    onset_frames_high = librosa.onset.onset_detect(
        onset_envelope=onset_envelope_high,
        sr=sr,
        hop_length=hop_length,
    )
    onset_times_high = librosa.frames_to_time(
        onset_frames_high,
        sr=sr,
        hop_length=hop_length,
    )

    onset_envelope = librosa.onset.onset_strength(
        y=y_percussive,
        sr=sr,
        hop_length=hop_length,
    )
    librosa_tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
    )

    raw_bpm = to_float(librosa_tempo)
    fixed_bpm = int(round(raw_bpm))
    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=hop_length,
    )

    if len(beat_times) > 0:
        base_offset = float(beat_times[0])
        nearby_kicks = [
            float(onset_time)
            for onset_time in onset_times_low
            if abs(float(onset_time) - base_offset) < 0.2
        ]
        offset_sec = (
            min(nearby_kicks, key=lambda value: abs(value - base_offset))
            if nearby_kicks
            else base_offset
        )
    else:
        offset_sec = 0.0

    grid_bpm = raw_bpm if raw_bpm > 0 else float(fixed_bpm)
    absolute_grid = build_absolute_grid(
        bpm=grid_bpm,
        offset_sec=offset_sec,
        duration_sec=duration_sec,
        beats_per_bar=beats_per_bar,
    )

    snapped_low = snap_to_grid(onset_times_low, absolute_grid)
    snapped_mid = snap_to_grid(onset_times_mid, absolute_grid)
    snapped_high = snap_to_grid(onset_times_high, absolute_grid)

    all_snare_scores = detect_snare_candidates(
        y_percussive=y_percussive,
        sr=sr,
        mid_onset_frames=onset_frames_mid,
        absolute_grid=absolute_grid,
        hop_length=hop_length,
        confidence_threshold=snare_confidence_threshold,
    )
    detected_snares = [
        event for event in all_snare_scores if event["is_snare_candidate"]
    ]

    result: dict[str, Any] = {
        "audio_file": str(audio_path),
        "sample_rate": int(sr),
        "duration_sec": round(duration_sec, 4),
        "time_signature": f"{beats_per_bar}/4",
        "bpm": {
            "librosa_raw": round(raw_bpm, 4),
            "fixed_integer": fixed_bpm,
        },
        "downbeat_offset_sec": round(offset_sec, 4),
        "grid_slots_count": len(absolute_grid),
        "absolute_grid": absolute_grid,
        "onsets_quantized": {
            "low_kick": snapped_low,
            "mid_snare": snapped_mid,
            "high_hat": snapped_high,
        },
        "snare_detection": {
            "method": "librosa_hpss_rule_based",
            "confidence_threshold": snare_confidence_threshold,
            "candidate_count": len(detected_snares),
            "events": detected_snares,
            "all_mid_onset_scores": all_snare_scores,
        },
        "output_files": {
            "json": None,
            "preview_wav": None,
        },
    }

    if save_json:
        json_path = output_path / f"{audio_path.stem}_advanced_analysis.json"
        result["output_files"]["json"] = str(json_path)
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

    if save_preview:
        snare_times = [event["original_time"] for event in detected_snares]
        clicks = librosa.clicks(
            times=snare_times,
            sr=sr,
            length=len(y),
            click_freq=1500.0,
            click_duration=0.03,
        )
        preview = np.clip(y + 0.35 * clicks, -1.0, 1.0)
        preview_path = preview_output_path / f"{audio_path.stem}_snare_preview.wav"
        sf.write(preview_path, preview, sr)
        result["output_files"]["preview_wav"] = str(preview_path)

        # JSON was written before the preview path was known, so update it.
        if save_json:
            json_path = Path(result["output_files"]["json"])
            with json_path.open("w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2)

    return result


def print_summary(result: dict[str, Any]) -> None:
    bpm = result["bpm"]
    onsets = result["onsets_quantized"]
    output_files = result["output_files"]

    print("Advanced Beat Analysis Complete")
    print(f"Audio           : {Path(result['audio_file']).name}")
    print(
        f"Raw BPM         : {bpm['librosa_raw']:.2f} "
        f"-> Fixed BPM: {bpm['fixed_integer']}"
    )
    print(f"Offset (Start)  : {result['downbeat_offset_sec']:.4f} sec")
    print(f"Total Grid      : {result['grid_slots_count']} slots (16th notes)")
    print(
        "Onsets Found    : "
        f"Low({len(onsets['low_kick'])}), "
        f"Mid({len(onsets['mid_snare'])}), "
        f"High({len(onsets['high_hat'])})"
    )
    print(
        "Snare Candidates: "
        f"{result['snare_detection']['candidate_count']}"
    )

    if output_files["json"]:
        print(f"JSON Saved      : {output_files['json']}")
    if output_files["preview_wav"]:
        print(f"Preview Saved   : {output_files['preview_wav']}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "오디오의 BPM, 16분음표 grid, 저·중·고역 onset과 "
            "스네어 후보를 분석합니다."
        )
    )
    parser.add_argument(
        "audio_path",
        help="분석할 MP3/WAV 등 오디오 파일 경로",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="JSON과 미리듣기 WAV 저장 폴더 (기본값: outputs)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"분석 sample rate (기본값: {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=DEFAULT_HOP_LENGTH,
        help=f"STFT/onset hop length (기본값: {DEFAULT_HOP_LENGTH})",
    )
    parser.add_argument(
        "--beats-per-bar",
        type=int,
        default=DEFAULT_BEATS_PER_BAR,
        help=f"한 마디의 박자 수 (기본값: {DEFAULT_BEATS_PER_BAR})",
    )
    parser.add_argument(
        "--snare-threshold",
        type=float,
        default=DEFAULT_SNARE_CONFIDENCE_THRESHOLD,
        help=(
            "스네어 후보 confidence 임계값 0~1 "
            f"(기본값: {DEFAULT_SNARE_CONFIDENCE_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="분석 JSON 파일을 저장하지 않음",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="스네어 클릭 미리듣기 WAV를 저장하지 않음",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="완성된 결과 JSON을 표준 출력에 표시",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        result = analyze_audio(
            audio_path_value=args.audio_path,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            beats_per_bar=args.beats_per_bar,
            snare_confidence_threshold=args.snare_threshold,
            save_json=not args.no_json,
            save_preview=not args.no_preview,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(status=1, message=f"오류: {error}\n")

    print_summary(result)

    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
