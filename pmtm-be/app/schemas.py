from typing import Literal
from pydantic import BaseModel, Field

LyricModel = str


class LyricGenerateRequest(BaseModel):
    bpm: int = Field(..., ge=40, le=220)
    topic: str | None = Field(default=None, description="Target lyric topic")
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


class TimedValue(BaseModel):
    time: float
    value: float


class BeatAnalysisResponse(BaseModel):
    fileName: str
    durationSec: float
    sampleRate: int
    sampleCount: int
    tempo: float
    timeSignature: str
    timeSignatureSource: str
    introStartSec: float
    introEndSec: float
    drumEntrySec: float
    firstBeatSec: float
    firstBarStartSec: float
    firstBarEndSec: float
    firstBarBeatTimes: list[float]
    beatTimes: list[float]
    onsetTimes: list[float]
    waveform: list[TimedValue]
    rms: list[TimedValue]
    onsetStrength: list[TimedValue]
    spectral: dict[str, float]
    chroma: list[float]
    mfcc: list[float]


DemoStatus = Literal[
    "queued",
    "analyzing",
    "writing",
    "planning",
    "voicing",
    "rendering",
    "converting_rvc",
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
    vocalUrl: str | None = None
    flowPlanUrl: str | None = None
    voicebank: str | None = None


class VoicebankResponse(BaseModel):
    id: str
    label: str
    available: bool


class RvcModelResponse(BaseModel):
    id: str
    label: str
    model_file: str
    has_index: bool
    available: bool
