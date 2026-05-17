from pydantic import BaseModel, Field


class LyricGenerateRequest(BaseModel):
    theme: str = Field(..., min_length=1, max_length=120)
    mood: str = Field(..., min_length=1, max_length=60)
    genre: str = Field(..., min_length=1, max_length=60)


class LyricGenerateResponse(BaseModel):
    title: str
    lyrics: str
    notes: list[str]
