from typing import Literal

from pydantic import BaseModel, Field

LyricModel = Literal["qwen-local", "qwen-exp-001-sft", "qwen-exp-001-grpo", "openai"]


class LyricGenerateRequest(BaseModel):
    bpm: int = Field(..., ge=40, le=220)
    llm: LyricModel = "qwen-local"


class LyricGenerateResponse(BaseModel):
    title: str
    lyrics: str
    bpm: int
    llm: LyricModel
    notes: list[str]
