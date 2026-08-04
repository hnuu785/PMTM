from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from beat_analysis import analyze_audio

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"  # Optional preview WAVs and legacy JSON lookup.
CACHE_INDEX_PATH = UPLOAD_DIR / ".analysis_cache_index.json"

# Development default. Replace with the deployed frontend origin in production.
CORS_ALLOW_ORIGINS = ["*"]
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
HASH_LENGTH = 64
UPLOAD_CHUNK_SIZE = 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Beat Analysis API",
    description="오디오 파일의 BPM, 박자 grid, onset, 스네어 후보를 분석합니다.",
    version="1.3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Prevent duplicate analysis of identical uploads within this API process.
_analysis_locks: dict[str, asyncio.Lock] = {}


def validate_analysis_id(analysis_id: str) -> None:
    if len(analysis_id) != HASH_LENGTH or any(
        character not in "0123456789abcdef" for character in analysis_id
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="analysis_id must be a lowercase SHA-256 hash.",
        )


def analysis_json_filename(original_filename: str) -> str:
    """Build the established '<original stem>_advanced_analysis.json' name."""
    stem = Path(original_filename).stem
    if not stem or stem in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded filename must contain a valid stem.",
        )
    return f"{stem}_advanced_analysis.json"


def safe_json_filename(json_filename: str) -> str:
    """Allow spaces, parentheses, and Korean names, but never directory traversal."""
    filename_path = Path(json_filename)
    if (
        filename_path.name != json_filename
        or not json_filename.endswith("_advanced_analysis.json")
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="json_filename must be an analysis JSON filename.",
        )
    return json_filename


def read_result(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as result_file:
            result = json.load(result_file)
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The stored analysis result is unreadable.",
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The stored analysis result has an invalid format.",
        )
    return result


def write_result(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as result_file:
        json.dump(result, result_file, ensure_ascii=False, indent=2)


def read_cache_index() -> dict[str, dict[str, str]]:
    """Read the SHA-256-to-filename index; analysis results remain separate JSON files."""
    if not CACHE_INDEX_PATH.is_file():
        return {}
    try:
        with CACHE_INDEX_PATH.open("r", encoding="utf-8") as index_file:
            index = json.load(index_file)
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The analysis cache index is unreadable.",
        ) from error
    if not isinstance(index, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The analysis cache index has an invalid format.",
        )
    return {
        key: value
        for key, value in index.items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def write_cache_index(index: dict[str, dict[str, str]]) -> None:
    with CACHE_INDEX_PATH.open("w", encoding="utf-8") as index_file:
        json.dump(index, index_file, ensure_ascii=False, indent=2)


def cached_entry(index: dict[str, dict[str, str]], analysis_id: str) -> dict[str, str] | None:
    entry = index.get(analysis_id)
    if not entry:
        return None
    json_filename = entry.get("json_filename")
    original_filename = entry.get("original_filename")
    if not isinstance(json_filename, str) or not isinstance(original_filename, str):
        return None
    try:
        safe_json_filename(json_filename)
    except HTTPException:
        return None
    return {"json_filename": json_filename, "original_filename": original_filename}


def find_result_by_filename(json_filename: str) -> tuple[Path, str] | None:
    """Find a filename-based cache, always preferring uploads over legacy outputs."""
    upload_path = UPLOAD_DIR / safe_json_filename(json_filename)
    if upload_path.is_file():
        return upload_path, "uploads"

    legacy_output_path = OUTPUT_DIR / json_filename
    if legacy_output_path.is_file():
        return legacy_output_path, "outputs"
    return None


def find_result_from_entry(entry: dict[str, str]) -> tuple[Path, str] | None:
    """Resolve an indexed result, even when it still lives in legacy outputs."""
    return find_result_by_filename(entry["json_filename"])


def add_preview_url(request: Request, result: dict[str, Any]) -> None:
    """Expose an optional preview WAV. Result JSON files are not statically exposed."""
    output_files = result.get("output_files")
    if not isinstance(output_files, dict) or not output_files.get("preview_wav"):
        return
    try:
        relative_path = Path(str(output_files["preview_wav"])).resolve().relative_to(
            OUTPUT_DIR.resolve()
        )
    except ValueError:
        return
    output_files["preview_url"] = str(
        request.url_for("outputs", path=relative_path.as_posix())
    )


def response_body(
    *,
    analysis_id: str | None,
    original_filename: str | None,
    json_filename: str,
    cached: bool,
    cache_source: str | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "analysis_id": analysis_id,
        "original_filename": original_filename,
        "json_filename": json_filename,
        "cached": cached,
        "cache_source": cache_source,
        "result": result,
    }


async def save_upload_and_hash(upload: UploadFile, temporary_path: Path) -> str:
    """Write an upload in chunks and calculate its SHA-256 without loading it all."""
    digest = hashlib.sha256()
    with temporary_path.open("wb") as temporary_file:
        while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
            digest.update(chunk)
            temporary_file.write(chunk)
    return digest.hexdigest()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Beat Analysis API is running", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/analysis/{analysis_id}", summary="SHA-256 ID로 저장된 분석 결과 조회")
def get_analysis(analysis_id: str, request: Request) -> dict[str, Any]:
    """Look up a cache result by its SHA-256 key through the JSON cache index."""
    validate_analysis_id(analysis_id)
    entry = cached_entry(read_cache_index(), analysis_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis result exists for this analysis_id.",
        )
    located_result = find_result_from_entry(entry)
    if located_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The cached analysis JSON file no longer exists.",
        )
    result_path, cache_source = located_result
    result = read_result(result_path)
    add_preview_url(request, result)
    return response_body(
        analysis_id=analysis_id,
        original_filename=entry["original_filename"],
        json_filename=entry["json_filename"],
        cached=True,
        cache_source=cache_source,
        result=result,
    )


@app.get(
    "/analysis/by-filename/{json_filename}",
    summary="분석 JSON 파일명으로 결과 조회",
)
def get_analysis_by_filename(json_filename: str, request: Request) -> dict[str, Any]:
    """Read uploads first, then support legacy outputs/*_advanced_analysis.json files."""
    json_filename = safe_json_filename(json_filename)
    located_result = find_result_by_filename(json_filename)
    if located_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis result exists for this json_filename.",
        )

    result_path, cache_source = located_result
    index = read_cache_index()
    matched_id = next(
        (
            analysis_id
            for analysis_id, entry in index.items()
            if cached_entry(index, analysis_id)
            and entry.get("json_filename") == json_filename
        ),
        None,
    )
    entry = cached_entry(index, matched_id) if matched_id else None
    result = read_result(result_path)
    add_preview_url(request, result)
    return response_body(
        analysis_id=matched_id,
        original_filename=entry["original_filename"] if entry else None,
        json_filename=json_filename,
        cached=True,
        cache_source=cache_source,
        result=result,
    )


@app.post(
    "/analyze",
    summary="오디오 업로드 및 비트 분석",
    responses={
        400: {"description": "잘못된 업로드 또는 분석 파라미터"},
        409: {"description": "동일한 결과 파일명이 다른 오디오에 이미 사용됨"},
    },
)
async def analyze(
    request: Request,
    file: UploadFile = File(
        ..., description="분석할 MP3, WAV, FLAC, OGG 또는 M4A 오디오 파일"
    ),
    snare_threshold: float = Query(
        0.65, ge=0.0, le=1.0, description="스네어 후보 confidence 임계값 (0.0~1.0)"
    ),
    beats_per_bar: int = Query(4, ge=1, description="한 마디의 박자 수"),
    generate_preview: bool = Query(
        False, description="스네어 click preview WAV 생성 여부 (기본값: false)"
    ),
) -> dict[str, Any]:
    """Save only filename-based JSON results in uploads and remove source audio."""
    original_filename = file.filename or "audio.wav"
    extension = Path(original_filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    json_filename = analysis_json_filename(original_filename)
    temporary_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"
    analysis_input_path: Path | None = None
    try:
        analysis_id = await save_upload_and_hash(file, temporary_path)
        lock = _analysis_locks.setdefault(analysis_id, asyncio.Lock())

        async with lock:
            index = read_cache_index()
            # Filename-based caches are valid even when their JSON has no SHA-256.
            # This check must precede analysis and uses uploads -> outputs precedence.
            filename_cache = find_result_by_filename(json_filename)
            if filename_cache is not None:
                cache_path, cache_source = filename_cache
                result = read_result(cache_path)
                add_preview_url(request, result)
                index[analysis_id] = {
                    "json_filename": json_filename,
                    "original_filename": original_filename,
                }
                write_cache_index(index)
                return response_body(
                    analysis_id=analysis_id,
                    original_filename=original_filename,
                    json_filename=json_filename,
                    cached=True,
                    cache_source=cache_source,
                    result=result,
                )

            entry = cached_entry(index, analysis_id)
            if entry is not None:
                indexed_cache = find_result_from_entry(entry)
                if indexed_cache is not None:
                    cache_path, cache_source = indexed_cache
                    result = read_result(cache_path)
                    add_preview_url(request, result)
                    return response_body(
                        analysis_id=analysis_id,
                        original_filename=entry["original_filename"],
                        json_filename=entry["json_filename"],
                        cached=True,
                        cache_source=cache_source,
                        result=result,
                    )

            result_path = UPLOAD_DIR / json_filename
            if result_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        "An analysis JSON with this original filename already exists for "
                        "different audio content. Rename the source file before uploading."
                    ),
                )

            # Use a hash-based temporary source name, then delete it in finally.
            analysis_input_path = UPLOAD_DIR / f"{analysis_id}{extension}"
            temporary_path.replace(analysis_input_path)
            result = analyze_audio(
                audio_path_value=analysis_input_path,
                output_dir=UPLOAD_DIR,
                preview_dir=OUTPUT_DIR,
                beats_per_bar=beats_per_bar,
                snare_confidence_threshold=snare_threshold,
                save_json=False,
                save_preview=generate_preview,
            )
            # Existing result field structure is retained; only its saved-path value is set.
            result["output_files"]["json"] = str(result_path)
            add_preview_url(request, result)
            write_result(result_path, result)
            index[analysis_id] = {
                "json_filename": json_filename,
                "original_filename": original_filename,
            }
            write_cache_index(index)
            return response_body(
                analysis_id=analysis_id,
                original_filename=original_filename,
                json_filename=json_filename,
                cached=False,
                cache_source=None,
                result=result,
            )

    except HTTPException:
        raise
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save or process the uploaded audio file.",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during audio analysis.",
        ) from error
    finally:
        await file.close()
        # Only source audio is removed. JSON result files and the JSON index are retained.
        temporary_path.unlink(missing_ok=True)
        if analysis_input_path is not None:
            analysis_input_path.unlink(missing_ok=True)
