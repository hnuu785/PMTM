# PMTM

`Next.js + FastAPI + PostgreSQL + Redis` 기반 작사 서비스 초기 세팅입니다.

## Structure

- `pmtm-fe`: Next.js 프론트엔드
- `pmtm-be`: FastAPI 백엔드
- `docker-compose.yml`: 로컬 개발용 통합 실행

## Local Development

### 1. Frontend

```bash
cd pmtm-fe
npm install
npm run dev
```

기본 주소: `http://localhost:3100`

### 2. Backend

```bash
cd pmtm-be
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8100
```

기본 주소: `http://localhost:8100`

### 3. Docker Compose

```bash
docker compose up --build
```

서비스:

- frontend: `http://localhost:3100`
- backend: `http://localhost:8100`
- docs: `http://localhost:8100/docs`
- postgres: `localhost:5433`
- redis: `localhost:6380`
