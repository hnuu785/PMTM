import json
import shutil
import subprocess
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import get_settings
from app.schemas import DemoStatus
from app.schemas import LyricModel


DEMO_STATUS_KEY_PREFIX = "pmtm:demo:"
DEMO_JOB_TIMEOUT_SECONDS = 600
DEMO_LENGTH_SECONDS = {30, 60}
TARGET_LYRIC_BARS = 8
VOICE_PRESETS = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
}


@dataclass(frozen=True)
class LyricBar:
    bar_index: int
    text: str


class VocalProvider(Protocol):
    def synthesize_line(self, text: str, voice: str, bpm: int, bar_duration_sec: float, output_path: Path) -> None:
        ...


class OpenAIVocalProvider:
    def __init__(self) -> None:
        self.settings = get_settings()

    def synthesize_line(self, text: str, voice: str, bpm: int, bar_duration_sec: float, output_path: Path) -> None:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        voice_value: str | dict[str, str]
        if voice.startswith("voice_"):
            voice_value = {"id": voice}
        else:
            voice_value = voice

        payload = {
            "model": self.settings.openai_tts_model,
            "voice": voice_value,
            "input": text,
            "instructions": (
                f"Deliver this as a Korean rap demo at {bpm} BPM. "
                f"Keep the phrase close to {bar_duration_sec:.2f} seconds with clear timing."
            ),
            "response_format": "wav",
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.settings.openai_timeout_seconds) as response:
                output_path.write_bytes(response.read())
            ensure_wav_file(output_path)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(detail[-2000:]) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI speech request failed: {exc}") from exc
        except TimeoutError as exc:
            raise RuntimeError("OpenAI speech generation timed out.") from exc


def get_vocal_provider() -> VocalProvider:
    return OpenAIVocalProvider()


def ensure_wav_file(output_path: Path) -> None:
    if output_path.read_bytes()[:4] == b"RIFF":
        return

    if not shutil.which("ffmpeg"):
        raise RuntimeError("OpenAI speech output was not WAV, and ffmpeg is required to convert it.")

    raw_path = output_path.with_suffix(".raw")
    output_path.replace(raw_path)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_path),
        "-ac",
        "1",
        "-ar",
        "24000",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def run_demo_generation(
    job_id: str,
    beat_path: str,
    work_dir: str,
    llm: LyricModel,
    genre: str,
    mood: str,
    demo_length_sec: int,
    voice: str,
) -> None:
    from app import main

    import redis

    settings = get_settings()
    redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    work_path = Path(work_dir)
    beat_file = Path(beat_path)

    try:
        _set_status(redis_client, job_id, "analyzing", progress=0.12)
        bpm = main._analyze_beat_bpm(str(beat_file))
        _trim_beat_segment(beat_file, work_path / "beat_segment.wav", demo_length_sec)

        _set_status(redis_client, job_id, "writing", progress=0.32, bpm=bpm)
        bars = TARGET_LYRIC_BARS
        lyrics, notes = main._generate_verse_for_model(
            bpm,
            llm,
            genre=genre,
            mood=mood,
            bars=bars,
        )
        lyric_bars = normalize_lyric_bars(lyrics, bars)
        lyrics_payload = {
            "bpm": bpm,
            "genre": genre,
            "mood": mood,
            "bars": [bar.__dict__ for bar in lyric_bars],
            "lyrics": lyrics,
        }
        (work_path / "lyrics.json").write_text(json.dumps(lyrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _set_status(redis_client, job_id, "voicing", progress=0.48, bpm=bpm, lyrics=lyrics, notes=notes)

        provider = get_vocal_provider()
        vocal_path = work_path / "vocal.wav"
        synthesize_vocal_track(provider, lyric_bars, voice, bpm, vocal_path)

        _set_status(redis_client, job_id, "mixing", progress=0.82, bpm=bpm, lyrics=lyrics, notes=notes)
        output_path = mix_demo_audio(work_path / "beat_segment.wav", vocal_path, work_path)
        audio_url = f"/api/v1/demos/{job_id}/audio"
        _set_status(
            redis_client,
            job_id,
            "succeeded",
            progress=1.0,
            bpm=bpm,
            lyrics=lyrics,
            notes=["librosa tempo 분석값을 BPM으로 사용했습니다.", *notes, f"데모 파일: {output_path.name}"],
            audio_url=audio_url,
        )
    except Exception as exc:
        _set_status(redis_client, job_id, "failed", progress=1.0, error=str(exc))


def normalize_lyric_bars(lyrics: str, bars: int) -> list[LyricBar]:
    lines = [
        line.strip()
        for line in lyrics.splitlines()
        if line.strip() and not line.strip().lower().startswith("[verse")
    ]
    selected = lines[:bars]
    while len(selected) < bars:
        selected.append("")
    return [LyricBar(bar_index=index + 1, text=text) for index, text in enumerate(selected)]


def synthesize_vocal_track(provider: VocalProvider, bars: list[LyricBar], voice: str, bpm: int, output_path: Path) -> None:
    bar_duration_sec = 60.0 / bpm * 4.0
    line_paths: list[Path] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for bar in bars:
        line_path = output_path.parent / f"vocal_line_{bar.bar_index:02d}.wav"
        text = bar.text or " "
        provider.synthesize_line(text, voice, bpm, bar_duration_sec, line_path)
        aligned_path = output_path.parent / f"vocal_line_{bar.bar_index:02d}_aligned.wav"
        align_wav_to_duration(line_path, aligned_path, bar_duration_sec)
        line_paths.append(aligned_path)

    concatenate_wavs(line_paths, output_path)


def align_wav_to_duration(input_path: Path, output_path: Path, target_duration_sec: float) -> None:
    with wave.open(str(input_path), "rb") as source:
        params = source.getparams()
        frames = source.readframes(source.getnframes())

    sample_width = params.sampwidth
    frame_width = sample_width * params.nchannels
    target_frames = max(1, round(target_duration_sec * params.framerate))
    current_frames = params.nframes

    if current_frames == target_frames:
        output_path.write_bytes(input_path.read_bytes())
        return

    if current_frames < target_frames:
        silence = b"\x00" * frame_width * (target_frames - current_frames)
        adjusted = frames + silence
    else:
        adjusted = frames[: target_frames * frame_width]

    with wave.open(str(output_path), "wb") as target:
        target.setparams(params._replace(nframes=target_frames))
        target.writeframes(adjusted)


def concatenate_wavs(input_paths: list[Path], output_path: Path) -> None:
    if not input_paths:
        raise RuntimeError("No vocal lines were generated.")

    with wave.open(str(input_paths[0]), "rb") as first:
        params = first.getparams()

    with wave.open(str(output_path), "wb") as target:
        target.setparams(params)
        for input_path in input_paths:
            with wave.open(str(input_path), "rb") as source:
                if source.getparams()[:3] != params[:3]:
                    raise RuntimeError("Generated vocal WAV files have incompatible formats.")
                target.writeframes(source.readframes(source.getnframes()))


def _trim_beat_segment(input_path: Path, output_path: Path, demo_length_sec: int) -> None:
    if shutil.which("ffmpeg"):
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-t",
            str(demo_length_sec),
            "-ac",
            "2",
            "-ar",
            "44100",
            str(output_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return

    if input_path.suffix.lower() == ".wav":
        shutil.copyfile(input_path, output_path)
        return

    raise RuntimeError("ffmpeg is required to trim non-WAV beat files.")


def mix_demo_audio(beat_path: Path, vocal_path: Path, work_dir: Path) -> Path:
    mp3_path = work_dir / "demo.mp3"
    wav_path = work_dir / "demo.wav"
    if shutil.which("ffmpeg"):
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(beat_path),
            "-i",
            str(vocal_path),
            "-filter_complex",
            "[0:a]volume=0.55[beat];[1:a]volume=1.25[vocal];[beat][vocal]amix=inputs=2:duration=first:dropout_transition=0",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(mp3_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return mp3_path

    _mix_wavs_fallback(beat_path, vocal_path, wav_path)
    return wav_path


def _mix_wavs_fallback(beat_path: Path, vocal_path: Path, output_path: Path) -> None:
    with wave.open(str(beat_path), "rb") as beat, wave.open(str(vocal_path), "rb") as vocal:
        if beat.getparams()[:3] != vocal.getparams()[:3]:
            raise RuntimeError("ffmpeg is required to mix WAV files with different formats.")

        params = beat.getparams()
        sample_width = params.sampwidth
        if sample_width != 2:
            raise RuntimeError("Fallback mixer only supports 16-bit WAV files.")

        beat_frames = beat.readframes(beat.getnframes())
        vocal_frames = vocal.readframes(vocal.getnframes())

    mixed = bytearray()
    frame_count = min(len(beat_frames), len(vocal_frames)) // sample_width
    for index in range(frame_count):
        offset = index * sample_width
        beat_sample = int.from_bytes(beat_frames[offset : offset + sample_width], "little", signed=True)
        vocal_sample = int.from_bytes(vocal_frames[offset : offset + sample_width], "little", signed=True)
        value = int(beat_sample * 0.55 + vocal_sample * 1.25)
        value = max(-32768, min(32767, value))
        mixed.extend(value.to_bytes(sample_width, "little", signed=True))

    with wave.open(str(output_path), "wb") as output:
        output.setparams(params)
        output.writeframes(bytes(mixed))


def _set_status(
    redis_client,
    job_id: str,
    status: DemoStatus,
    *,
    progress: float,
    bpm: int | None = None,
    lyrics: str | None = None,
    notes: list[str] | None = None,
    error: str | None = None,
    audio_url: str | None = None,
) -> None:
    payload = {
        "jobId": job_id,
        "status": status,
        "progress": str(max(0.0, min(1.0, progress))),
    }
    if bpm is not None:
        payload["bpm"] = str(bpm)
    if lyrics is not None:
        payload["lyrics"] = lyrics
    if notes is not None:
        payload["notes"] = json.dumps(notes, ensure_ascii=False)
    if error is not None:
        payload["error"] = error
    if audio_url is not None:
        payload["audioUrl"] = audio_url

    redis_client.hset(f"{DEMO_STATUS_KEY_PREFIX}{job_id}", mapping=payload)
    redis_client.expire(f"{DEMO_STATUS_KEY_PREFIX}{job_id}", DEMO_JOB_TIMEOUT_SECONDS)


def parse_status_payload(data: dict[str, str]) -> dict:
    notes_raw = data.get("notes")
    notes: list[str] = []
    if notes_raw:
        try:
            parsed = json.loads(notes_raw)
            if isinstance(parsed, list):
                notes = [str(item) for item in parsed]
        except json.JSONDecodeError:
            notes = []

    bpm = data.get("bpm")
    return {
        "jobId": data.get("jobId", ""),
        "status": data.get("status", "failed"),
        "progress": float(data.get("progress") or 0.0),
        "bpm": int(bpm) if bpm and bpm.isdigit() else None,
        "lyrics": data.get("lyrics"),
        "notes": notes,
        "error": data.get("error"),
        "audioUrl": data.get("audioUrl"),
    }


def validate_demo_length(value: int) -> int:
    if value not in DEMO_LENGTH_SECONDS:
        raise ValueError("데모 길이는 30초 또는 60초만 지원합니다.")
    return value


def validate_voice(value: str) -> str:
    cleaned = value.strip()
    if cleaned in VOICE_PRESETS or cleaned.startswith("voice_"):
        return cleaned
    raise ValueError("지원하지 않는 보이스입니다.")


def sanitize_prompt_field(value: str, fallback: str, max_length: int = 40) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return fallback
    return cleaned[:max_length]
