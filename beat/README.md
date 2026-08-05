# Beat Analysis API

> **Windows 기준 안내**: 이 문서의 설치·실행 명령과 경로 표기는 **Windows PowerShell** 기준입니다. macOS/Linux에서는 가상환경 활성화 명령과 경로 표기가 다릅니다.

FastAPI 기반 오디오 비트 분석 API입니다. 지원 오디오 파일을 업로드하면 BPM, downbeat, 16분음표 grid, kick/snare/hat onset, 스네어 후보 confidence를 분석합니다.

## 주요 기능

- `POST /analyze`: 오디오 업로드 및 분석
- 원본 파일명 기반 JSON 캐시
- SHA-256 기반 내부 캐시 인덱스
- `GET /analysis/{analysis_id}` 및 파일명 기반 조회
- 선택적 스네어 click preview WAV 생성
- Swagger UI 테스트 (`/docs`)

## 요구 사항

- Python 3.10 이상
- FFmpeg 권장: MP3, M4A 등 압축 오디오 분석 시 필요할 수 있습니다.

Windows에서 FFmpeg를 설치했다면 FFmpeg 실행 파일이 `PATH`에 포함되어 있어야 합니다.

## 설치 및 실행 (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn api:app --reload
```

서버 실행 후 다음 주소를 사용합니다.

- Swagger UI: http://127.0.0.1:8000/docs
- 상태 확인: http://127.0.0.1:8000/health

가상환경 활성화가 PowerShell 실행 정책으로 막히면 현재 PowerShell 창에서만 다음을 실행한 뒤 다시 시도합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 저장 및 캐시 규칙

### 파일명 규칙

원본 확장자를 제거한 stem 뒤에 `_advanced_analysis.json`을 붙입니다.

```text
80_boombap(2).mp3
→ uploads/80_boombap(2)_advanced_analysis.json

내 비트 01.wav
→ uploads/내 비트 01_advanced_analysis.json
```

공백, 괄호, 한글 파일명도 지원합니다.

### 캐시 탐색 순서

`POST /analyze`는 새 분석 전에 원본 파일명으로 만든 JSON 파일을 다음 순서로 찾습니다.

1. `uploads/<원본 stem>_advanced_analysis.json`
2. `outputs/<원본 stem>_advanced_analysis.json` (이전 버전 레거시 결과)

둘 중 하나라도 있으면 `beat_analysis.py`를 다시 실행하지 않고 저장된 결과를 반환합니다. 이때 응답은 `cached: true`이며 `cache_source`가 각각 `"uploads"` 또는 `"outputs"`입니다.

매칭되는 파일명 JSON이 없을 때만 SHA-256 내부 인덱스를 확인하고, 그래도 없을 때만 새 분석을 실행합니다. 새 분석 결과는 항상 `uploads/`에 저장됩니다.

> 파일명 기반 캐시가 최우선입니다. 따라서 같은 파일명을 다시 업로드하면 내용이 달라도 기존 파일명 JSON이 캐시로 반환됩니다. 서로 다른 음원은 서로 다른 원본 파일명을 사용하세요.

### 폴더 구조

```text
uploads/
  <원본 stem>_advanced_analysis.json  # 분석 결과 JSON
  .analysis_cache_index.json          # SHA-256 → JSON 파일명 내부 인덱스
  <임시 오디오>                       # 분석 중에만 존재하고 완료 후 삭제
outputs/
  <sha256>_snare_preview.wav          # preview 요청 시에만 생성
  <기존>_advanced_analysis.json        # 레거시 조회 호환용
```

원본 오디오는 `uploads/`에 임시 저장된 뒤, 성공·실패와 관계없이 요청 종료 시 삭제됩니다. JSON 결과와 캐시 인덱스 JSON은 삭제하지 않습니다.

## API

### `POST /analyze`

오디오 파일을 업로드하고 분석하거나 캐시 결과를 반환합니다. Swagger UI에서 `Try it out`으로 바로 테스트할 수 있습니다.

| 파라미터 | 위치 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `file` | multipart form | 필수 | MP3, WAV, FLAC, OGG, M4A 파일 |
| `snare_threshold` | query | `0.65` | 스네어 후보 confidence 임계값 (`0.0`~`1.0`) |
| `beats_per_bar` | query | `4` | 한 마디의 박자 수 |
| `generate_preview` | query | `false` | 스네어 click preview WAV 생성 여부 |

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze?snare_threshold=0.65&beats_per_bar=4&generate_preview=false" `
  -F "file=@C:\music\80_boombap(2).mp3"
```

응답 예시:

```json
{
  "analysis_id": "64자리_sha256_해시",
  "original_filename": "80_boombap(2).mp3",
  "json_filename": "80_boombap(2)_advanced_analysis.json",
  "cached": true,
  "cache_source": "outputs",
  "result": {
    "bpm": {},
    "absolute_grid": [],
    "onsets_quantized": {},
    "snare_detection": {}
  }
}
```

`cache_source` 값은 아래와 같습니다.

| 값 | 의미 |
| --- | --- |
| `"uploads"` | `uploads`의 JSON 캐시 반환 |
| `"outputs"` | 기존 `outputs` 레거시 JSON 반환 |
| `null` | 새 분석 실행 후 결과 저장 |

preview WAV는 `generate_preview=true`일 때만 생성됩니다. 기본값은 `false`입니다.

### `GET /analysis/{analysis_id}`

SHA-256 내부 캐시 ID로 결과를 조회합니다.

```powershell
curl.exe "http://127.0.0.1:8000/analysis/<analysis_id>"
```

### `GET /analysis/by-filename/{json_filename}`

분석 JSON 파일명으로 결과를 조회합니다. `uploads`를 먼저 찾고, 없으면 기존 `outputs` JSON을 찾습니다.

```powershell
curl.exe "http://127.0.0.1:8000/analysis/by-filename/80_boombap%282%29_advanced_analysis.json"
```

| 상태 코드 | 의미 |
| --- | --- |
| `200` | 분석 또는 캐시 결과 반환 |
| `400` | 지원하지 않는 파일 형식 또는 잘못된 분석 파라미터 |
| `404` | 분석 ID 또는 JSON 파일명을 찾을 수 없음 |
| `500` | 저장된 JSON 또는 캐시 인덱스를 읽을 수 없음 |

## CORS

개발 환경에서는 [api.py](api.py)의 다음 상수가 모든 Origin을 허용합니다.

```python
CORS_ALLOW_ORIGINS = ["*"]
```

운영 배포 전에는 프론트엔드 주소로 제한하세요.

```python
CORS_ALLOW_ORIGINS = ["https://app.example.com"]
```

## Git 관리

`.gitignore`는 원본·생성 오디오, 환경 변수, Python 캐시와 로컬 개발 산출물을 제외합니다. `uploads/*.json` 분석 결과와 `.analysis_cache_index.json`은 Git 추적 대상으로 남습니다. 현재 구조에서 사용하지 않는 `outputs/*.json`은 Git에서 제외됩니다.

## 프로젝트 구조

```text
api.py             # FastAPI 라우트, 파일명/해시 캐시, 업로드 정리
beat_analysis.py   # librosa 기반 비트/스네어 분석 로직
requirements.txt   # Python 의존성
uploads/           # 분석 JSON, 캐시 인덱스, 요청 중 임시 오디오
outputs/           # 선택적 preview WAV 및 레거시 JSON
```
