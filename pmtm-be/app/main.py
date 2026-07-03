import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import LyricGenerateRequest, LyricGenerateResponse, LyricModel, RhymeAnalyzeRequest
from app.schemas import RhymeLineAnalysis

settings = get_settings()
MAX_BEAT_UPLOAD_BYTES = 20 * 1024 * 1024
RHYME_GROUP_THRESHOLD = 0.72
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


@app.post("/api/v1/lyrics/analyze-rhyme", response_model=list[RhymeLineAnalysis])
def analyze_rhyme(payload: RhymeAnalyzeRequest) -> list[RhymeLineAnalysis]:
    return _analyze_rhyme_lines(payload.lines)


def _generate_verse_for_model(bpm: int, llm: LyricModel) -> tuple[str, list[str]]:
    if llm == "openai":
        return _generate_openai_verse(bpm), [
            f"{settings.openai_model} 생성 결과입니다.",
            "OpenAI Responses API를 사용했습니다.",
        ]
    if llm == "qwen-exp-001-sft":
        return _generate_qwen_verse(bpm, adapter="exp-001/sft_rap_qwen"), [
            "Qwen/Qwen2.5-1.5B + exp-001 SFT 어댑터 생성 결과입니다.",
            "pmtm-ai/models/exp-001/sft_rap_qwen을 사용했습니다.",
        ]
    if llm == "qwen-exp-001-grpo":
        return _generate_qwen_verse(bpm, adapter="exp-001/grpo_rap_qwen"), [
            "Qwen/Qwen2.5-1.5B + exp-001 GRPO 어댑터 생성 결과입니다.",
            "pmtm-ai/models/exp-001/grpo_rap_qwen을 사용했습니다.",
        ]
    if llm == "qwen-exp-002-sft":
        return _generate_qwen_verse(bpm, adapter="exp-002/sft_rap_qwen"), [
            "Qwen/Qwen2.5-1.5B + exp-002 SFT 어댑터 생성 결과입니다.",
            "pmtm-ai/models/exp-002/sft_rap_qwen을 사용했습니다.",
        ]
    if llm == "qwen-exp-002-grpo":
        return _generate_qwen_verse(bpm, adapter="exp-002/grpo_rap_qwen"), [
            "Qwen/Qwen2.5-1.5B + exp-002 GRPO 어댑터 생성 결과입니다.",
            "pmtm-ai/models/exp-002/grpo_rap_qwen을 사용했습니다.",
        ]

    return _generate_qwen_verse(bpm), [
        "Qwen/Qwen2.5-1.5B 베이스 모델 생성 결과입니다.",
        "LoRA 어댑터를 사용하지 않은 순수 Qwen 추론입니다.",
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

    get_line_rhyme_score = _load_rhyme_score_func()
    for left in range(line_count):
        for right in range(left + 1, line_count):
            score = get_line_rhyme_score(clean_lines[left], clean_lines[right])
            scores[left] = max(scores[left], score)
            scores[right] = max(scores[right], score)
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
        analyses.append(
            RhymeLineAnalysis(
                text=line,
                rhymeGroup=rhyme_group,
                score=round(scores[index], 4),
                highlightStart=highlight_start,
                highlightEnd=highlight_end,
            )
        )

    return analyses


def _load_rhyme_score_func():
    project_root = Path(__file__).resolve().parents[2]
    scoring_root = project_root / "pmtm-ai" / "app" / "rhyme_scoring"
    if str(scoring_root) not in sys.path:
        sys.path.insert(0, str(scoring_root))

    from rhyme_engine import get_line_rhyme_score

    return get_line_rhyme_score


def _find_rhyme_highlight(line: str) -> tuple[int | None, int | None]:
    matches = list(re.finditer(r"[가-힣A-Za-z]+", line))
    if not matches:
        return None, None
    match = matches[-1]
    return match.start(), match.end()


async def _save_uploaded_beat(beat: UploadFile) -> str:
    if beat.content_type not in SUPPORTED_BEAT_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="지원하지 않는 오디오 형식입니다.")

    suffix = Path(beat.filename or "").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
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


def _generate_qwen_verse(bpm: int, adapter: str | None = None) -> str:
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
                "--base-model",
                "Qwen/Qwen2.5-1.5B",
                "--bpm",
                str(bpm),
                "--bars",
                "8",
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
        else:
            command.extend(
                [
                    "--tokenizer-model",
                    "Qwen/Qwen2.5-1.5B-Instruct",
                ]
            )

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
    return "[Verse]\n" + "\n".join(lines[:8])


def _generate_openai_verse(bpm: int) -> str:
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured.",
        )

    payload = _build_openai_payload(bpm)
    data = _request_openai_response(payload)

    generated = _extract_openai_text(data)
    if not generated:
        raise HTTPException(status_code=502, detail=_describe_empty_openai_response(data))

    lines = [line.strip() for line in generated.splitlines() if line.strip()]
    if lines and lines[0].lower().startswith("[verse"):
        lines = lines[1:]
    return "[Verse]\n" + "\n".join(lines[:8])


def _build_openai_payload(bpm: int) -> dict:
    payload: dict = {
        "model": settings.openai_model,
        "instructions": (
            "You write Korean rap lyrics. Return only an 8-line verse in Korean. "
            "Use natural Korean hip-hop phrasing, with English words only as occasional ad-libs. "
            "Do not include title, explanation, numbering, or markdown."
        ),
        "input": (
            f"BPM {bpm}에 맞는 한국어 랩 8마디 벌스를 써줘. "
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
