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

프론트엔드와 백엔드는 Docker로 띄우지 않고 로컬에서 실행합니다. Docker Compose는 PostgreSQL, Redis만 실행합니다.

```bash
docker compose up --build
```

서비스:

- postgres: `localhost:5433`
- redis: `localhost:6380`

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
