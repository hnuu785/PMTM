from fastapi import FastAPI
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
    chorus = (
        f"{payload.theme}을 따라 걷는 밤\n"
        f"{payload.mood}한 숨결이 번지는 light\n"
        "아직 끝나지 않은 마음의 rhyme\n"
        "다시 너를 부르는 line"
    )
    verse = (
        f"{payload.genre} 리듬 위로 천천히 쌓인 장면들\n"
        "흔들리던 문장도 이제는 노래가 되고\n"
        "머뭇대던 진심이 후렴으로 넘어가면\n"
        "우린 한 번 더 선명해진다"
    )

    return LyricGenerateResponse(
        title=f"{payload.theme} Demo",
        lyrics=f"[Verse]\n{verse}\n\n[Chorus]\n{chorus}",
        notes=[
            "현재는 LLM 연결 전 데모 응답입니다.",
            "다음 단계에서 Anthropic 또는 OpenAI API를 연결하면 됩니다.",
        ],
    )
