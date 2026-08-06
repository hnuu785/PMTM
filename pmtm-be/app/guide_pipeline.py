import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.beat_integration import run_advanced_beat_analysis
from app.config import get_settings
from app.demo_pipeline import DEMO_JOB_TIMEOUT_SECONDS, DEMO_STATUS_KEY_PREFIX, mix_demo_audio
from app.flow_adapter import build_flow_plan, write_diffsinger_ds, write_flow_plan
from app.rvc_adapter import render_rvc
from app.schemas import DemoStatus
from app.utils.redis import update_redis_status


MIX_TAIL_SECONDS = 0.1


@dataclass(frozen=True)
class VoicebankProfile:
    id: str
    label: str
    directory: str
    base_f0_hz: float


VOICEBANK_PROFILES = {
    profile.id: profile
    for profile in (
        VoicebankProfile("potg", "POTG", "potg", 190.0),
        VoicebankProfile("kitane", "KITANE", "kitane", 205.0),
        VoicebankProfile("rang", "RANG", "rang", 145.0),
        VoicebankProfile("lunar", "LUNAR", "lunar", 195.0),
    )
}


def validate_voicebank(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in VOICEBANK_PROFILES:
        supported = ", ".join(VOICEBANK_PROFILES)
        raise ValueError(f"지원하는 DiffSinger 보이스뱅크는 {supported}입니다.")
    return normalized


def validate_bpm(value: int) -> int:
    if value < 40 or value > 220:
        raise ValueError("BPM은 40부터 220 사이여야 합니다.")
    return value


def validate_first_bar_start(value: float) -> float:
    if not math.isfinite(value) or value < 0 or value > 600:
        raise ValueError("첫 마디 시작 시각은 0초부터 600초 사이여야 합니다.")
    return value


def validate_guide_flow(lyrics: str, bpm: int, first_bar_start_sec: float, voicebank_id: str, genre: str = "boom_bap") -> None:
    profile = VOICEBANK_PROFILES[voicebank_id]
    build_flow_plan(
        lyrics,
        bpm,
        first_bar_start_sec,
        voicebank_id,
        base_f0_hz=profile.base_f0_hz,
        genre=genre,
    )


def list_voicebanks() -> list[dict[str, object]]:
    root = resolve_configured_path(get_settings().diffsinger_voicebank_root)
    return [
        {
            "id": profile.id,
            "label": profile.label,
            "available": (root / profile.directory).is_dir(),
        }
        for profile in VOICEBANK_PROFILES.values()
    ]


def run_guide_demo_generation(
    job_id: str,
    beat_path: str,
    work_dir: str,
    lyrics: str,
    bpm: int,
    first_bar_start_sec: float,
    voicebank_id: str,
    genre: str = "boom_bap",
    rvc_model_id: str | None = None,
    use_index: bool | None = None,
) -> None:
    # pyrefly: ignore [missing-import]
    import redis

    settings = get_settings()
    redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    work_path = Path(work_dir)
    beat_file = Path(beat_path)
    profile = VOICEBANK_PROFILES[voicebank_id]

    try:
        _set_status(
            redis_client,
            job_id,
            "planning",
            progress=0.16,
            bpm=bpm,
            lyrics=lyrics,
            voicebank=voicebank_id,
        )
        beat_analysis = run_advanced_beat_analysis(beat_file) if beat_file.exists() else None
        plan = build_flow_plan(
            lyrics,
            bpm,
            first_bar_start_sec,
            voicebank_id,
            base_f0_hz=profile.base_f0_hz,
            genre=genre,
            beat_analysis=beat_analysis,
        )
        flow_plan_path = work_path / "flow-plan.json"
        score_path = work_path / "score.ds"
        write_flow_plan(plan, flow_plan_path)
        write_diffsinger_ds(plan, score_path, base_f0_hz=profile.base_f0_hz)

        render_duration_sec = first_bar_start_sec + plan.beatMap.barDurationSec * plan.beatMap.barCount
        mix_duration_sec = render_duration_sec + MIX_TAIL_SECONDS
        _trim_beat(beat_file, work_path / "beat_segment.wav", mix_duration_sec)

        _set_status(
            redis_client,
            job_id,
            "rendering",
            progress=0.42,
            bpm=bpm,
            lyrics=lyrics,
            voicebank=voicebank_id,
        )
        raw_vocal_path = work_path / "vocal_raw.wav"
        vocal_path = work_path / "vocal.wav"
        render_diffsinger(score_path, raw_vocal_path, voicebank_id)
        _fit_vocal_to_duration(raw_vocal_path, vocal_path, mix_duration_sec)

        rvc_applied_note = None
        if rvc_model_id:
            _set_status(
                redis_client,
                job_id,
                "converting_rvc",
                progress=0.68,
                bpm=bpm,
                lyrics=lyrics,
                voicebank=voicebank_id,
            )
            rvc_vocal_path = work_path / "vocal_rvc.wav"
            render_rvc(vocal_path, rvc_vocal_path, rvc_model_id=rvc_model_id, use_index=use_index)
            vocal_path = rvc_vocal_path
            index_tag = " (인덱스 적용)" if use_index else ""
            rvc_applied_note = f"RVC 음색 변환 적용: {rvc_model_id}{index_tag}"

        _set_status(
            redis_client,
            job_id,
            "mixing",
            progress=0.86,
            bpm=bpm,
            lyrics=lyrics,
            voicebank=voicebank_id,
        )
        output_path = mix_demo_audio(work_path / "beat_segment.wav", vocal_path, work_path)
        line_count = len([
            line
            for line in lyrics.splitlines()
            if line.strip() and not line.strip().lower().startswith("[verse")
        ])
        bar_count = len(plan.bars)
        rule_desc = "2줄=1마디(트랩)" if line_count == 16 else "1줄=1마디(붐뱁)"
        notes = [
            f"편집된 {line_count}줄({bar_count}마디, {rule_desc})을 렌더링했습니다.",
            f"첫 마디 시작: {first_bar_start_sec:.3f}초",
            f"DiffSinger 보이스뱅크: {profile.label}",
        ]
        if rvc_applied_note:
            notes.append(rvc_applied_note)
        notes.append(f"데모 파일: {output_path.name}")

        _set_status(
            redis_client,
            job_id,
            "succeeded",
            progress=1.0,
            bpm=bpm,
            lyrics=lyrics,
            voicebank=voicebank_id,
            notes=notes,
            audio_url=f"/api/v1/demos/{job_id}/audio",
            vocal_url=f"/api/v1/demos/{job_id}/vocal",
            flow_plan_url=f"/api/v1/demos/{job_id}/flow-plan",
        )
    except Exception as exc:
        _set_status(
            redis_client,
            job_id,
            "failed",
            progress=1.0,
            bpm=bpm,
            lyrics=lyrics,
            voicebank=voicebank_id,
            error=str(exc),
        )


def render_diffsinger(score_path: Path, output_path: Path, voicebank_id: str) -> None:
    settings = get_settings()
    profile = VOICEBANK_PROFILES[voicebank_id]
    project_root = Path(__file__).resolve().parents[2]
    python_path = resolve_configured_path(settings.diffsinger_python_path, project_root / "pmtm-be")
    voicebank_root = resolve_configured_path(settings.diffsinger_voicebank_root, project_root / "pmtm-be")
    voicebank_path = voicebank_root / profile.directory
    renderer_path = project_root / "pmtm-svs" / "render.py"

    if not python_path.is_file():
        raise RuntimeError(f"DiffSinger Python runtime not found: {python_path}")
    if not voicebank_path.is_dir():
        raise RuntimeError(f"DiffSinger voicebank not found: {voicebank_path}")
    if not renderer_path.is_file():
        raise RuntimeError(f"DiffSinger renderer not found: {renderer_path}")

    command = [
        str(python_path),
        str(renderer_path),
        "--score",
        str(score_path),
        "--voice-bank",
        str(voicebank_path),
        "--output",
        str(output_path),
        "--device",
        settings.diffsinger_device,
        "--lang",
        "ko",
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=settings.diffsinger_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("DiffSinger 렌더링 시간이 초과되었습니다.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "DiffSinger rendering failed."
        raise RuntimeError(detail[-4000:]) from exc

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("DiffSinger 렌더러가 보컬 WAV를 생성하지 않았습니다.")


def resolve_configured_path(value: str, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base or Path(__file__).resolve().parents[1]) / path


def _trim_beat(input_path: Path, output_path: Path, duration_sec: float) -> None:
    if not shutil.which("ffmpeg"):
        if input_path.suffix.lower() == ".wav":
            shutil.copyfile(input_path, output_path)
            return
        raise RuntimeError("ffmpeg is required to prepare the beat for mixing.")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-t",
            f"{duration_sec:.6f}",
            "-ac",
            "2",
            "-ar",
            "44100",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _fit_vocal_to_duration(input_path: Path, output_path: Path, duration_sec: float) -> None:
    if not shutil.which("ffmpeg"):
        shutil.copyfile(input_path, output_path)
        return
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-af",
            "apad",
            "-t",
            f"{duration_sec:.6f}",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _set_status(
    redis_client,
    job_id: str,
    status: DemoStatus,
    *,
    progress: float,
    bpm: int | None = None,
    lyrics: str | None = None,
    voicebank: str | None = None,
    notes: list[str] | None = None,
    error: str | None = None,
    audio_url: str | None = None,
    vocal_url: str | None = None,
    flow_plan_url: str | None = None,
) -> None:
    update_redis_status(
        redis_client=redis_client,
        key_prefix=DEMO_STATUS_KEY_PREFIX,
        job_id=job_id,
        status=status,
        progress=progress,
        timeout_seconds=DEMO_JOB_TIMEOUT_SECONDS,
        bpm=bpm,
        lyrics=lyrics,
        voicebank=voicebank,
        notes=notes,
        error=error,
        audio_url=audio_url,
        vocal_url=vocal_url,
        flow_plan_url=flow_plan_url,
    )


def enqueue_guide_demo(
    redis_client,
    job_id: str,
    beat_path: str,
    work_dir: str,
    lyrics: str,
    bpm: int,
    first_bar_start_sec: float,
    voicebank_id: str,
    genre: str = "boom_bap",
    rvc_model_id: str | None = None,
    use_index: bool | None = None,
) -> None:
    # pyrefly: ignore [missing-import]
    from rq import Queue

    line_count = len([
        line
        for line in lyrics.splitlines()
        if line.strip() and not line.strip().lower().startswith("[verse")
    ])
    status_key = f"{DEMO_STATUS_KEY_PREFIX}{job_id}"
    redis_client.hset(
        status_key,
        mapping={
            "jobId": job_id,
            "status": "queued",
            "progress": "0.0",
            "bpm": str(bpm),
            "lyrics": lyrics,
            "voicebank": voicebank_id,
            "notes": json.dumps([f"{line_count}마디 SVS 작업이 대기열에 등록되었습니다."], ensure_ascii=False),
        },
    )
    redis_client.expire(status_key, 60 * 60)
    Queue("demo-generation", connection=redis_client).enqueue(
        run_guide_demo_generation,
        job_id,
        beat_path,
        work_dir,
        lyrics,
        bpm,
        first_bar_start_sec,
        voicebank_id,
        genre,
        rvc_model_id,
        use_index,
        job_id=job_id,
        job_timeout=get_settings().diffsinger_timeout_seconds + 120,
    )
