# PMTM

`Next.js + FastAPI + PostgreSQL + Redis` 기반 작사 서비스 초기 세팅입니다.

## Structure

- `pmtm-fe`: Next.js 프론트엔드
- `pmtm-be`: FastAPI 백엔드
- `pmtm-ai`: 가사 생성 AI 학습/추론 코드 영역
- `docker-compose.yml`: 로컬 인프라 실행

## Local Development

### 1. Frontend

```bash
cd pmtm-fe
npm install
npm run dev
```

기본 주소: `http://localhost:3100`

### 2. Backend

백엔드는 로컬에서 실행합니다. 가사 생성은 업로드한 비트 파일을 `librosa`로 분석해 BPM을 추정한 뒤,
`pmtm-ai/venv`의 `Qwen/Qwen2.5-3B-Instruct` 추론 CLI를 호출합니다.
LLM 선택값 `qwen-exp-005-sft`는 `pmtm-ai/models/exp-005/sft_rap_qwen`, `qwen-exp-005-grpo`는 `pmtm-ai/models/exp-005/grpo_rap_qwen` LoRA 어댑터를 적용합니다.
OpenAI 선택지를 사용하려면 `OPENAI_API_KEY`를 설정합니다.

```bash
cd pmtm-be
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8100
```

기본 주소: `http://localhost:8100`

비트 기반 생성 API는 `POST /api/v1/lyrics/generate-from-beat`를 사용합니다.
요청 형식은 `multipart/form-data`이며 `beat` 오디오 파일과 `llm` 값을 보냅니다.
지원 형식은 MP3, WAV, M4A/MP4, AAC, FLAC이고 파일은 임시 분석 후 저장하지 않습니다.

AI 보컬 데모 생성 API는 `POST /api/v1/demos/generate-from-beat`를 사용합니다.
요청 형식은 `multipart/form-data`이며 `beat`, `llm`, `genre`, `mood`, `demoLengthSec`, `voice` 값을 보냅니다.
응답의 `jobId`로 `GET /api/v1/demos/{jobId}`를 polling하고, 완료 후 `GET /api/v1/demos/{jobId}/audio`에서 데모 오디오를 받을 수 있습니다.
데모 생성은 Redis + RQ 비동기 작업이므로 백엔드와 별도로 워커를 실행해야 합니다.

```bash
cd pmtm-be
source .venv/bin/activate
PYTHONPATH=. rq worker demo-generation --worker-class rq.SimpleWorker --url redis://localhost:6380/0
```

로컬 macOS 개발에서는 기본 RQ worker의 fork 방식이 `librosa`/오디오 분석 단계에서 멈출 수 있어 `rq.SimpleWorker`를 사용합니다.

보컬 합성은 OpenAI Speech API를 사용하므로 `OPENAI_API_KEY`가 필요합니다.
기본 TTS 모델은 `OPENAI_TTS_MODEL=gpt-4o-mini-tts`입니다.
MP3/M4A 등 비-WAV 비트 처리와 MP3 데모 출력을 위해 로컬 `ffmpeg` 설치가 필요합니다.

### 3. Local Infrastructure

Docker Compose로 PostgreSQL, Redis만 실행합니다. 프론트엔드와 백엔드는 로컬에서 직접 실행합니다.

```bash
docker compose up
```

서비스:

- postgres: `localhost:5433`
- redis: `localhost:6380`

OpenAI 선택지를 사용하려면 `pmtm-be/.env` 또는 `pmtm-be/.env.local`에 키를 넣습니다.

```bash
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5-mini
```

Qwen 로컬 추론은 `pmtm-ai`의 별도 Python 환경과 모델 캐시가 필요합니다. 로컬 개발에서는 백엔드도 로컬 Python 환경에서 실행합니다.

## AI Workspace

`pmtm-ai`는 모델 학습, 추론, 데이터셋, 체크포인트 자산을 분리해서 다루는 디렉터리입니다.

```bash
cd pmtm-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

현재 로컬 기본 추론 모델은 `Qwen/Qwen2.5-3B-Instruct`입니다. 모델 파일은 Hugging Face 캐시에 있어야 하며, 백엔드 기본 설정은 `pmtm-ai/venv/bin/python`을 호출합니다.
exp-005 시연 모델은 `llm=qwen-exp-005-sft` 또는 `llm=qwen-exp-005-grpo`로 호출하며, 어댑터 파일은 각각 `pmtm-ai/models/exp-005/sft_rap_qwen`, `pmtm-ai/models/exp-005/grpo_rap_qwen`에 있어야 합니다.
OpenAI 선택지의 기본 모델은 `gpt-5-mini`입니다.
