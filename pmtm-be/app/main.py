import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import get_settings
from app.demo_pipeline import DEMO_STATUS_KEY_PREFIX
from app.demo_pipeline import parse_status_payload
from app.demo_pipeline import run_demo_generation
from app.demo_pipeline import sanitize_prompt_field
from app.demo_pipeline import validate_demo_length
from app.demo_pipeline import validate_vocal_start_bars
from app.demo_pipeline import validate_voice
from app.schemas import BeatAnalysisResponse, DemoGenerateResponse, DemoStatusResponse
from app.schemas import LyricGenerateRequest, LyricGenerateResponse, LyricModel, RhymeAnalyzeRequest
from app.schemas import RhymeHighlightRange, RhymeLineAnalysis

settings = get_settings()
MAX_BEAT_UPLOAD_BYTES = 20 * 1024 * 1024
RHYME_GROUP_THRESHOLD = 0.50
RHYME_SYLLABLE_HIGHLIGHT_THRESHOLD = 0.50
TARGET_LYRIC_BARS = 8
EXP_005_SFT_ADAPTER = "exp-005/sft_rap_qwen"
EXP_005_GRPO_ADAPTER = "exp-005/grpo_rap_qwen"
DEMO_STORAGE_ROOT = Path(__file__).resolve().parents[1] / "storage" / "demos"
SUPPORTED_BEAT_CONTENT_TYPES = {
    "audio/aac",
    "audio/flac",
    "audio/mp3",
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/x-m4a",
    "audio/x-wav",
}

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Backend API for the PMTM lyric generation service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.post("/api/v1/lyrics/generate", response_model=LyricGenerateResponse)
def generate_lyrics(payload: LyricGenerateRequest) -> LyricGenerateResponse:
    bpm = payload.bpm
    lyrics, notes = _generate_verse_for_model(bpm, payload.llm)
    lyric_lines = _extract_lyric_lines(lyrics)

    return LyricGenerateResponse(
        title=f"{bpm} BPM Verse",
        bpm=bpm,
        llm=payload.llm,
        lyrics=lyrics,
        notes=notes,
        rhymeAnalysis=_analyze_rhyme_lines(lyric_lines),
    )


@app.post("/api/v1/lyrics/generate-from-beat", response_model=LyricGenerateResponse)
async def generate_lyrics_from_beat(
    llm: LyricModel = Form("qwen-local"),
    beat: UploadFile = File(...),
) -> LyricGenerateResponse:
    beat_path = await _save_uploaded_beat(beat)
    try:
        bpm = _analyze_beat_bpm(beat_path)
    finally:
        try:
            os.unlink(beat_path)
        except FileNotFoundError:
            pass

    lyrics, notes = _generate_verse_for_model(bpm, llm)
    notes = ["librosa tempo 분석값을 BPM으로 사용했습니다.", *notes]
    lyric_lines = _extract_lyric_lines(lyrics)

    return LyricGenerateResponse(
        title=f"{bpm} BPM Verse",
        bpm=bpm,
        llm=llm,
        lyrics=lyrics,
        notes=notes,
        rhymeAnalysis=_analyze_rhyme_lines(lyric_lines),
    )


@app.post("/api/v1/beats/analyze", response_model=BeatAnalysisResponse)
async def analyze_beat(beat: UploadFile = File(...)) -> BeatAnalysisResponse:
    file_name = beat.filename or "beat"
    beat_path = await _save_uploaded_beat(beat)
    try:
        return _analyze_beat(beat_path, file_name)
    finally:
        try:
            os.unlink(beat_path)
        except FileNotFoundError:
            pass


@app.post("/api/v1/lyrics/analyze-rhyme", response_model=list[RhymeLineAnalysis])
def analyze_rhyme(payload: RhymeAnalyzeRequest) -> list[RhymeLineAnalysis]:
    return _analyze_rhyme_lines(payload.lines)


@app.post("/api/v1/demos/generate-from-beat", response_model=DemoGenerateResponse)
async def generate_demo_from_beat(
    llm: LyricModel = Form("qwen-local"),
    genre: str = Form("Korean hip-hop"),
    mood: str = Form("confident"),
    demoLengthSec: int = Form(30),
    voice: str = Form("verse"),
    vocalStartBars: int = Form(4),
    beat: UploadFile = File(...),
) -> DemoGenerateResponse:
    try:
        demo_length_sec = validate_demo_length(demoLengthSec)
        voice_preset = validate_voice(voice)
        vocal_start_bars = validate_vocal_start_bars(vocalStartBars)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex
    work_dir = DEMO_STORAGE_ROOT / job_id
    work_dir.mkdir(parents=True, exist_ok=False)
    beat_path = await _save_uploaded_beat(beat, target_dir=work_dir, filename_prefix="input")
    genre_value = sanitize_prompt_field(genre, "Korean hip-hop")
    mood_value = sanitize_prompt_field(mood, "confident")
    _enqueue_demo_job(
        job_id,
        beat_path,
        str(work_dir),
        llm,
        genre_value,
        mood_value,
        demo_length_sec,
        voice_preset,
        vocal_start_bars,
    )
    return DemoGenerateResponse(jobId=job_id, status="queued")


@app.get("/api/v1/demos/{job_id}", response_model=DemoStatusResponse)
def get_demo_status(job_id: str) -> DemoStatusResponse:
    redis_client = _get_redis_client()
    data = redis_client.hgetall(f"{DEMO_STATUS_KEY_PREFIX}{job_id}")
    if not data:
        raise HTTPException(status_code=404, detail="데모 작업을 찾을 수 없습니다.")
    payload = parse_status_payload(data)
    _sync_failed_rq_status(redis_client, job_id, payload)
    worker_count = _get_demo_worker_count(redis_client)
    payload["workerAvailable"] = worker_count > 0
    payload["workerCount"] = worker_count
    if payload["status"] == "queued" and worker_count == 0:
        payload["notes"] = [
            *payload["notes"],
            "데모 생성 워커가 실행 중이 아닙니다. rq worker demo-generation을 시작해주세요.",
        ]
    return DemoStatusResponse(**payload)


@app.get("/api/v1/demos/{job_id}/audio")
def get_demo_audio(job_id: str) -> FileResponse:
    work_dir = DEMO_STORAGE_ROOT / job_id
    mp3_path = work_dir / "demo.mp3"
    wav_path = work_dir / "demo.wav"
    if mp3_path.exists():
        return FileResponse(mp3_path, media_type="audio/mpeg", filename=f"pmtm-demo-{job_id}.mp3")
    if wav_path.exists():
        return FileResponse(wav_path, media_type="audio/wav", filename=f"pmtm-demo-{job_id}.wav")
    raise HTTPException(status_code=404, detail="데모 오디오 파일을 찾을 수 없습니다.")


def _generate_verse_for_model(
    bpm: int,
    llm: LyricModel,
    *,
    genre: str = "Korean hip-hop",
    mood: str = "confident",
    bars: int = 8,
) -> tuple[str, list[str]]:
    bars = TARGET_LYRIC_BARS
    if llm == "openai":
        return _generate_openai_verse(bpm, genre=genre, mood=mood, bars=bars), [
            f"{settings.openai_model} 생성 결과입니다.",
            "OpenAI Responses API를 사용했습니다.",
        ]
    if llm == "qwen-exp-005-sft":
        return _generate_qwen_verse(bpm, EXP_005_SFT_ADAPTER, genre=genre, mood=mood, bars=bars), [
            "exp-005 SFT LoRA 어댑터 생성 결과입니다.",
            "Qwen/Qwen2.5-3B-Instruct 베이스 모델에 exp-005/sft_rap_qwen 어댑터를 적용했습니다.",
        ]
    if llm == "qwen-exp-005-grpo":
        return _generate_qwen_verse(bpm, EXP_005_GRPO_ADAPTER, genre=genre, mood=mood, bars=bars), [
            "exp-005 GRPO LoRA 어댑터 생성 결과입니다.",
            "Qwen/Qwen2.5-3B-Instruct 베이스 모델에 exp-005/grpo_rap_qwen 어댑터를 적용했습니다.",
        ]
    return _generate_qwen_verse(bpm, genre=genre, mood=mood, bars=bars), [
        "Qwen/Qwen2.5-3B-Instruct 베이스 모델 생성 결과입니다.",
        "LoRA 어댑터를 사용하지 않은 Qwen Instruct 추론입니다.",
    ]


def _extract_lyric_lines(lyrics: str) -> list[str]:
    return [
        line.strip()
        for line in lyrics.splitlines()
        if line.strip() and not line.strip().lower().startswith("[verse")
    ]


def _analyze_rhyme_lines(lines: list[str]) -> list[RhymeLineAnalysis]:
    clean_lines = [line.strip() for line in lines[:32]]
    line_count = len(clean_lines)
    parents = list(range(line_count))
    scores = [0.0] * line_count
    best_match_indexes: list[int | None] = [None] * line_count

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    get_line_rhyme_score, calculate_syllable_score, get_phonemes = _load_rhyme_analysis_funcs()
    for left in range(line_count):
        for right in range(left + 1, line_count):
            score = get_line_rhyme_score(clean_lines[left], clean_lines[right])
            if score > scores[left]:
                scores[left] = score
                best_match_indexes[left] = right
            if score > scores[right]:
                scores[right] = score
                best_match_indexes[right] = left
            if score >= RHYME_GROUP_THRESHOLD:
                union(left, right)

    root_counts: dict[int, int] = {}
    for index in range(line_count):
        root = find(index)
        root_counts[root] = root_counts.get(root, 0) + 1

    group_ids: dict[int, int] = {}
    next_group_id = 0
    analyses: list[RhymeLineAnalysis] = []
    for index, line in enumerate(clean_lines):
        root = find(index)
        rhyme_group = None
        if root_counts[root] > 1:
            if root not in group_ids:
                group_ids[root] = next_group_id
                next_group_id += 1
            rhyme_group = group_ids[root]

        highlight_start, highlight_end = _find_rhyme_highlight(line)
        highlight_ranges: list[RhymeHighlightRange] = []
        best_match_index = best_match_indexes[index]
        if rhyme_group is not None and best_match_index is not None:
            highlight_ranges = _find_rhyme_highlight_ranges(
                line,
                clean_lines[best_match_index],
                calculate_syllable_score,
                get_phonemes,
            )
        analyses.append(
            RhymeLineAnalysis(
                text=line,
                rhymeGroup=rhyme_group,
                score=round(scores[index], 4),
                highlightStart=highlight_start,
                highlightEnd=highlight_end,
                highlightRanges=highlight_ranges,
            )
        )

    return analyses


def _load_rhyme_score_func():
    get_line_rhyme_score, _, _ = _load_rhyme_analysis_funcs()
    return get_line_rhyme_score


def _load_rhyme_analysis_funcs():
    project_root = Path(__file__).resolve().parents[2]
    scoring_root = project_root / "pmtm-ai" / "app" / "rhyme_scoring"
    if str(scoring_root) not in sys.path:
        sys.path.insert(0, str(scoring_root))

    from phonetics_utils import get_phonemes
    from rhyme_engine import calculate_syllable_score, get_line_rhyme_score

    return get_line_rhyme_score, calculate_syllable_score, get_phonemes


def _find_rhyme_highlight(line: str) -> tuple[int | None, int | None]:
    matches = list(re.finditer(r"[가-힣A-Za-z]+", line))
    if not matches:
        return None, None
    match = matches[-1]
    return match.start(), match.end()


def _find_rhyme_highlight_ranges(
    line: str,
    match_line: str,
    calculate_syllable_score,
    get_phonemes,
) -> list[RhymeHighlightRange]:
    token_match = _find_last_rhyme_token(line)
    if token_match is None:
        return []

    if re.fullmatch(r"[A-Za-z]+", token_match.group(0)):
        return [RhymeHighlightRange(start=token_match.start(), end=token_match.end())]

    syllable_ranges = _find_hangul_syllable_ranges(line)
    line_phonemes = get_phonemes(line)
    match_phonemes = get_phonemes(match_line)
    compare_count = min(len(line_phonemes), len(match_phonemes), len(syllable_ranges), 3)
    if compare_count <= 0:
        return []

    ranges: list[tuple[int, int]] = []
    for offset in range(1, compare_count + 1):
        score = calculate_syllable_score(line_phonemes[-offset], match_phonemes[-offset])
        if score >= RHYME_SYLLABLE_HIGHLIGHT_THRESHOLD:
            ranges.append(syllable_ranges[-offset])

    return [
        RhymeHighlightRange(start=start, end=end)
        for start, end in _merge_ranges(sorted(ranges))
    ]


def _find_last_rhyme_token(line: str) -> re.Match[str] | None:
    matches = list(re.finditer(r"[가-힣A-Za-z]+", line))
    return matches[-1] if matches else None


def _find_hangul_syllable_ranges(line: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"[가-힣]", line)]


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []

    merged: list[tuple[int, int]] = [ranges[0]]
    for start, end in ranges[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


async def _save_uploaded_beat(
    beat: UploadFile,
    target_dir: Path | None = None,
    filename_prefix: str = "beat",
) -> str:
    if beat.content_type not in SUPPORTED_BEAT_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="지원하지 않는 오디오 형식입니다.")

    suffix = Path(beat.filename or "").suffix
    if target_dir:
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = str(target_dir / f"{filename_prefix}{suffix or '.audio'}")
        tmp_file = open(tmp_path, "wb")
    else:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    with tmp_file as tmp:
        total = 0
        while chunk := await beat.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_BEAT_UPLOAD_BYTES:
                tmp_path = tmp.name
                tmp.close()
                os.unlink(tmp_path)
                raise HTTPException(status_code=400, detail="비트 파일은 20MB 이하로 업로드해주세요.")
            tmp.write(chunk)

        tmp_path = tmp.name

    if total == 0:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail="비트 파일이 비어 있습니다.")

    return tmp_path


def _get_redis_client():
    try:
        import redis
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="redis package is not installed.") from exc

    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _get_demo_worker_count(redis_client) -> int:
    try:
        from rq import Worker
    except ImportError:
        return 0

    try:
        return len(Worker.all(connection=redis_client))
    except Exception:
        return 0


def _sync_failed_rq_status(redis_client, job_id: str, payload: dict) -> None:
    if payload["status"] in {"succeeded", "failed"}:
        return

    try:
        from rq.job import Job
        from rq.job import JobStatus

        job = Job.fetch(job_id, connection=redis_client)
        if job.get_status() != JobStatus.FAILED:
            return

        payload["status"] = "failed"
        payload["progress"] = 1.0
        payload["error"] = job.exc_info or "데모 생성 worker가 작업 중 예기치 않게 종료되었습니다."
        redis_client.hset(
            f"{DEMO_STATUS_KEY_PREFIX}{job_id}",
            mapping={
                "status": payload["status"],
                "progress": str(payload["progress"]),
                "error": payload["error"],
            },
        )
    except Exception:
        return


def _enqueue_demo_job(
    job_id: str,
    beat_path: str,
    work_dir: str,
    llm: LyricModel,
    genre: str,
    mood: str,
    demo_length_sec: int,
    voice: str,
    vocal_start_bars: int,
) -> None:
    try:
        from rq import Queue
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="rq package is not installed.") from exc

    redis_client = _get_redis_client()
    status_key = f"{DEMO_STATUS_KEY_PREFIX}{job_id}"
    redis_client.hset(
        status_key,
        mapping={
            "jobId": job_id,
            "status": "queued",
            "progress": "0.0",
            "notes": json.dumps(["데모 생성 작업이 대기열에 등록되었습니다."], ensure_ascii=False),
        },
    )
    redis_client.expire(status_key, 60 * 60)

    queue = Queue("demo-generation", connection=redis_client)
    queue.enqueue(
        run_demo_generation,
        job_id,
        beat_path,
        work_dir,
        llm,
        genre,
        mood,
        demo_length_sec,
        voice,
        vocal_start_bars,
        job_id=job_id,
        job_timeout=60 * 10,
    )


def _analyze_beat_bpm(file_path: str) -> int:
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(file_path, sr=None, mono=True, duration=60)
        if len(y) == 0:
            raise ValueError("empty audio")

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_value = float(np.asarray(tempo).reshape(-1)[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="BPM 분석 실패") from exc

    if not math.isfinite(tempo_value):
        raise HTTPException(status_code=400, detail="BPM 분석 실패")

    bpm = round(tempo_value)
    if bpm < 40 or bpm > 220:
        raise HTTPException(status_code=400, detail="BPM 분석 실패")

    return bpm


def _analyze_beat(file_path: str, file_name: str) -> BeatAnalysisResponse:
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(file_path, sr=None, mono=True, duration=60)
        if len(y) == 0:
            raise ValueError("empty audio")

        hop_length = 512
        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            onset_envelope=onset_envelope,
            hop_length=hop_length,
        )
        tempo_value = float(np.asarray(tempo).reshape(-1)[0])
        if not math.isfinite(tempo_value):
            raise ValueError("invalid tempo")

        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        _, percussive = librosa.effects.hpss(y)
        percussive_onset_envelope = librosa.onset.onset_strength(
            y=percussive,
            sr=sr,
            hop_length=hop_length,
        )
        percussive_onset_frames = librosa.onset.onset_detect(
            onset_envelope=percussive_onset_envelope,
            sr=sr,
            hop_length=hop_length,
        )
        drum_entry_frame = _select_sustained_onset(
            percussive_onset_frames,
            percussive_onset_envelope[percussive_onset_frames],
            round(2 * sr / hop_length),
        )
        drum_entry_sec = (
            float(librosa.frames_to_time(drum_entry_frame, sr=sr, hop_length=hop_length))
            if drum_entry_frame is not None
            else 0.0
        )
        first_beat_sec = (
            min((float(value) for value in beat_times), key=lambda value: abs(value - drum_entry_sec))
            if len(beat_times) > 0
            else drum_entry_sec
        )
        first_bar_beat_times, first_bar_end_sec = _build_first_bar(
            beat_times,
            first_beat_sec,
            tempo_value,
        )

        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=hop_length,
        )
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

        frame_times = librosa.frames_to_time(
            np.arange(len(onset_envelope)),
            sr=sr,
            hop_length=hop_length,
        )
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        waveform_times = np.arange(len(y), dtype=float) / sr

        return BeatAnalysisResponse(
            fileName=file_name,
            durationSec=round(float(librosa.get_duration(y=y, sr=sr)), 3),
            sampleRate=int(sr),
            sampleCount=int(len(y)),
            tempo=round(tempo_value, 2),
            timeSignature="4/4",
            timeSignatureSource="assumed",
            introStartSec=0.0,
            introEndSec=round(first_beat_sec, 4),
            drumEntrySec=round(drum_entry_sec, 4),
            firstBeatSec=round(first_beat_sec, 4),
            firstBarStartSec=round(first_beat_sec, 4),
            firstBarEndSec=round(first_bar_end_sec, 4),
            firstBarBeatTimes=_rounded_list(first_bar_beat_times),
            beatTimes=_rounded_list(beat_times),
            onsetTimes=_rounded_list(librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)),
            waveform=_downsample_series(waveform_times, y, 700),
            rms=_downsample_series(rms_times, rms, 400),
            onsetStrength=_downsample_series(frame_times, onset_envelope, 400),
            spectral={
                "rmsMean": round(float(np.mean(rms)), 6),
                "rmsMax": round(float(np.max(rms)), 6),
                "zeroCrossingRateMean": round(float(np.mean(zero_crossing_rate)), 6),
                "centroidMeanHz": round(float(np.mean(centroid)), 2),
                "bandwidthMeanHz": round(float(np.mean(bandwidth)), 2),
                "rolloffMeanHz": round(float(np.mean(rolloff)), 2),
            },
            chroma=[round(float(value), 6) for value in np.mean(chroma, axis=1)],
            mfcc=[round(float(value), 4) for value in np.mean(mfcc, axis=1)],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="비트 분석 실패") from exc


def _rounded_list(values) -> list[float]:
    return [round(float(value), 4) for value in values]


def _select_sustained_onset(onset_frames, strengths, horizon_frames: int) -> int | None:
    if len(onset_frames) == 0:
        return None

    strength_values = [float(value) for value in strengths]
    threshold = max(strength_values) * 0.25
    frame_values = [int(value) for value in onset_frames]
    for index, frame in enumerate(frame_values):
        if strength_values[index] < threshold:
            continue
        following_count = sum(frame <= candidate <= frame + horizon_frames for candidate in frame_values)
        if following_count >= 3:
            return frame

    return frame_values[0]


def _build_first_bar(beat_times, first_beat_sec: float, tempo: float) -> tuple[list[float], float]:
    values = [float(value) for value in beat_times]
    beat_interval = 60 / tempo if tempo > 0 else 0.0
    first_index = (
        min(range(len(values)), key=lambda index: abs(values[index] - first_beat_sec))
        if values
        else 0
    )
    bar_beats = [
        values[first_index + offset]
        if first_index + offset < len(values)
        else first_beat_sec + beat_interval * offset
        for offset in range(4)
    ]
    bar_end = (
        values[first_index + 4]
        if first_index + 4 < len(values)
        else first_beat_sec + beat_interval * 4
    )
    return bar_beats, bar_end


def _downsample_series(times, values, max_points: int) -> list[dict[str, float]]:
    import numpy as np

    value_count = min(len(times), len(values))
    if value_count == 0:
        return []
    indexes = np.linspace(0, value_count - 1, min(value_count, max_points), dtype=int)
    return [
        {"time": round(float(times[index]), 4), "value": round(float(values[index]), 6)}
        for index in indexes
    ]


def _generate_qwen_verse(
    bpm: int,
    adapter: str | None = None,
    *,
    genre: str = "Korean hip-hop",
    mood: str = "confident",
    bars: int = 8,
) -> str:
    bars = TARGET_LYRIC_BARS
    project_root = Path(__file__).resolve().parents[2]
    ai_root = project_root / "pmtm-ai"
    python_path = Path(settings.qwen_python_path)
    if not python_path.is_absolute():
        python_path = project_root / "pmtm-be" / python_path

    if not python_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Qwen Python runtime not found: {python_path}",
        )

    try:
        command = [
                str(python_path),
                "-m",
                "app.inference.generate_for_api",
                "--bpm",
                str(bpm),
                "--genre",
                genre,
                "--mood",
                mood,
                "--bars",
                str(bars),
                "--max-new-tokens",
                "180",
                "--temperature",
                "0.85",
                "--top-p",
                "0.92",
        ]
        if adapter:
            adapter_path = ai_root / "models" / adapter
            if not adapter_path.exists():
                raise HTTPException(
                    status_code=503,
                    detail=f"Qwen adapter not found: {adapter_path}",
                )
            command.extend(["--adapter", str(adapter_path)])

        completed = subprocess.run(
            command,
            cwd=ai_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=settings.qwen_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Qwen generation timed out.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "Qwen generation failed."
        raise HTTPException(status_code=502, detail=detail[-2000:]) from exc

    generated = completed.stdout.strip()
    if not generated:
        raise HTTPException(status_code=502, detail="Qwen returned an empty lyric.")

    lines = [line.strip() for line in generated.splitlines() if line.strip()]
    if lines and lines[0].lower().startswith("[verse"):
        lines = lines[1:]
    return "[Verse]\n" + "\n".join(lines[:bars])


def _generate_openai_verse(
    bpm: int,
    *,
    genre: str = "Korean hip-hop",
    mood: str = "confident",
    bars: int = 8,
) -> str:
    bars = TARGET_LYRIC_BARS
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured.",
        )

    payload = _build_openai_payload(bpm, genre=genre, mood=mood, bars=bars)
    data = _request_openai_response(payload)

    generated = _extract_openai_text(data)
    if not generated:
        raise HTTPException(status_code=502, detail=_describe_empty_openai_response(data))

    lines = [line.strip() for line in generated.splitlines() if line.strip()]
    if lines and lines[0].lower().startswith("[verse"):
        lines = lines[1:]
    return "[Verse]\n" + "\n".join(lines[:bars])


def _build_openai_payload(
    bpm: int,
    *,
    genre: str = "Korean hip-hop",
    mood: str = "confident",
    bars: int = 8,
) -> dict:
    bars = TARGET_LYRIC_BARS
    payload: dict = {
        "model": settings.openai_model,
        "instructions": (
            f"You write Korean rap lyrics. Return only a {bars}-line verse in Korean. "
            "Use natural Korean hip-hop phrasing, with English words only as occasional ad-libs. "
            "Do not include title, explanation, numbering, or markdown."
        ),
        "input": (
            f"BPM {bpm}, 장르 {genre}, 분위기 {mood}에 맞는 한국어 랩 {bars}마디 벌스를 써줘. "
            "각 줄은 한 마디처럼 호흡이 맞아야 하고, 전체 가사의 대부분은 한국어여야 해."
        ),
        "max_output_tokens": 500,
    }

    if settings.openai_model.startswith("gpt-5"):
        payload["reasoning"] = {"effort": "minimal"}
        payload["text"] = {"verbosity": "low"}

    return payload


def _request_openai_response(payload: dict) -> dict:
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=settings.openai_timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=detail[-2000:]) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="OpenAI generation timed out.") from exc
    return data


def _extract_openai_text(data: dict) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()

    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _describe_empty_openai_response(data: dict) -> str:
    status = data.get("status")
    incomplete_details = data.get("incomplete_details")
    if incomplete_details:
        return f"OpenAI returned no lyric text. status={status}, incomplete_details={incomplete_details}"

    output_types: list[str] = []
    for item in data.get("output", []):
        item_type = item.get("type")
        if isinstance(item_type, str):
            output_types.append(item_type)
        for content in item.get("content", []):
            content_type = content.get("type")
            if isinstance(content_type, str):
                output_types.append(content_type)

    if output_types:
        return f"OpenAI returned no lyric text. status={status}, output_types={output_types}"

    return f"OpenAI returned no lyric text. status={status}"
