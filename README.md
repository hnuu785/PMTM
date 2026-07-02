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

백엔드는 로컬에서 실행합니다. 가사 생성은 `pmtm-ai/venv`의 순수 `Qwen/Qwen2.5-1.5B` 추론 CLI를 호출합니다.
OpenAI 선택지를 사용하려면 `OPENAI_API_KEY`를 설정합니다.

```bash
cd pmtm-be
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8100
```

기본 주소: `http://localhost:8100`

### 3. Docker Compose

Docker Compose로 백엔드, PostgreSQL, Redis를 실행합니다. 프론트엔드는 로컬에서 직접 실행합니다.

```bash
docker compose up --build
```

서비스:

- backend: `http://localhost:8100`
- postgres: `localhost:5433`
- redis: `localhost:6380`

백엔드 Docker 이미지는 FastAPI 실행에 필요한 최소 의존성만 설치합니다. OpenAI 선택지를 사용하려면 루트 `.env` 또는 `.env.local`에 키를 넣습니다. 둘 다 있으면 `.env.local` 값이 우선합니다.

```bash
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5-mini
```

Qwen 로컬 추론은 `pmtm-ai`의 별도 Python 환경과 모델 캐시가 필요합니다. Docker 백엔드 컨테이너 안에서 Qwen까지 실행하려면 AI 의존성 설치와 Hugging Face 캐시 마운트 구성이 추가로 필요합니다.

## AI Workspace

`pmtm-ai`는 모델 학습, 추론, 데이터셋, 체크포인트 자산을 분리해서 다루는 디렉터리입니다.

```bash
cd pmtm-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

현재 로컬 기본 추론 모델은 `Qwen/Qwen2.5-1.5B`입니다. 모델 파일은 Hugging Face 캐시에 있어야 하며, 백엔드 기본 설정은 `pmtm-ai/venv/bin/python`을 호출합니다.
OpenAI 선택지의 기본 모델은 `gpt-5-mini`입니다.
학습 결과물 선택지는 `pmtm-ai/models/exp-001/sft_rap_qwen`, `pmtm-ai/models/exp-001/grpo_rap_qwen` 어댑터를 사용합니다.
