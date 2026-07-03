from typing import Literal

from pydantic import BaseModel, Field

LyricModel = Literal[
    "qwen-local",
    "qwen-exp-001-sft",
    "qwen-exp-001-grpo",
    "qwen-exp-002-sft",
    "qwen-exp-002-grpo",
    "openai",
]


class LyricGenerateRequest(BaseModel):
    bpm: int = Field(..., ge=40, le=220)
    llm: LyricModel = "qwen-local"


class RhymeAnalyzeRequest(BaseModel):
    lines: list[str] = Field(default_factory=list, max_length=32)


class RhymeLineAnalysis(BaseModel):
    text: str
    rhymeGroup: int | None = None
    score: float = 0.0
    highlightStart: int | None = None
    highlightEnd: int | None = None


class LyricGenerateResponse(BaseModel):
    title: str
    lyrics: str
    bpm: int
    llm: LyricModel
    notes: list[str]
    rhymeAnalysis: list[RhymeLineAnalysis] = Field(default_factory=list)
