from typing import Literal

from pydantic import BaseModel, Field

LyricModel = Literal[
    "qwen-local",
    "qwen-exp-005-sft",
    "qwen-exp-005-grpo",
    "openai",
]


class LyricGenerateRequest(BaseModel):
    bpm: int = Field(..., ge=40, le=220)
    llm: LyricModel = "qwen-local"


class RhymeAnalyzeRequest(BaseModel):
    lines: list[str] = Field(default_factory=list, max_length=32)


class RhymeHighlightRange(BaseModel):
    start: int
    end: int


class RhymeLineAnalysis(BaseModel):
    text: str
    rhymeGroup: int | None = None
    score: float = 0.0
    highlightStart: int | None = None
    highlightEnd: int | None = None
    highlightRanges: list[RhymeHighlightRange] = Field(default_factory=list)


class LyricGenerateResponse(BaseModel):
    title: str
    lyrics: str
    bpm: int
    llm: LyricModel
    notes: list[str]
    rhymeAnalysis: list[RhymeLineAnalysis] = Field(default_factory=list)


DemoStatus = Literal[
    "queued",
    "analyzing",
    "writing",
    "voicing",
    "mixing",
    "succeeded",
    "failed",
]


class DemoGenerateResponse(BaseModel):
    jobId: str
    status: DemoStatus


class DemoStatusResponse(BaseModel):
    jobId: str
    status: DemoStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    workerAvailable: bool = True
    workerCount: int = 0
    bpm: int | None = None
    lyrics: str | None = None
    notes: list[str] = Field(default_factory=list)
    error: str | None = None
    audioUrl: str | None = None
