import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import LyricGenerateRequest, LyricGenerateResponse

settings = get_settings()

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
    if payload.llm == "openai":
        lyrics = _generate_openai_verse(bpm)
        notes = [
            f"{settings.openai_model} 생성 결과입니다.",
            "OpenAI Responses API를 사용했습니다.",
        ]
    elif payload.llm == "qwen-exp-001-sft":
        lyrics = _generate_qwen_verse(bpm, adapter="exp-001/sft_rap_qwen")
        notes = [
            "Qwen/Qwen2.5-1.5B + exp-001 SFT 어댑터 생성 결과입니다.",
            "pmtm-ai/models/exp-001/sft_rap_qwen을 사용했습니다.",
        ]
    elif payload.llm == "qwen-exp-001-grpo":
        lyrics = _generate_qwen_verse(bpm, adapter="exp-001/grpo_rap_qwen")
        notes = [
            "Qwen/Qwen2.5-1.5B + exp-001 GRPO 어댑터 생성 결과입니다.",
            "pmtm-ai/models/exp-001/grpo_rap_qwen을 사용했습니다.",
        ]
    else:
        lyrics = _generate_qwen_verse(bpm)
        notes = [
            "Qwen/Qwen2.5-1.5B 베이스 모델 생성 결과입니다.",
            "LoRA 어댑터를 사용하지 않은 순수 Qwen 추론입니다.",
        ]

    return LyricGenerateResponse(
        title=f"{bpm} BPM Verse",
        bpm=bpm,
        llm=payload.llm,
        lyrics=lyrics,
        notes=notes,
    )


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

    payload = {
        "model": settings.openai_model,
        "instructions": (
            "You write rap lyrics. Return only an 8-line verse. "
            "Do not include title, explanation, numbering, or markdown."
        ),
        "input": (
            f"Write an 8-bar rap verse for BPM {bpm}. "
            "Each line should feel like one bar and match the BPM's breathing and line length."
        ),
        "max_output_tokens": 220,
    }
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

    generated = _extract_openai_text(data)
    if not generated:
        raise HTTPException(status_code=502, detail="OpenAI returned an empty lyric.")

    lines = [line.strip() for line in generated.splitlines() if line.strip()]
    if lines and lines[0].lower().startswith("[verse"):
        lines = lines[1:]
    return "[Verse]\n" + "\n".join(lines[:8])


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
