# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

화장품 성분 상담 AI 에이전트 — FastAPI + LangChain/LangGraph 기반의 ReAct 에이전트가 식약처 PDF 문서를 Elasticsearch 하이브리드 검색(BM25 + kNN)하여 답변을 생성합니다.

## Commands

```bash
# 의존성 설치
uv sync

# 개발 서버 실행
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 테스트
uv run pytest
uv run pytest tests/test_main.py              # 단일 파일
uv run pytest tests/test_main.py::test_name   # 단일 테스트

# 린트
uv run ruff check .

# 포맷팅
uv run black .

# PDF 데이터 인덱싱 (data/pdfs/에 PDF 파일 필요, ES 연결 필요)
uv run python scripts/ingest.py
```

## Architecture

**요청 흐름:** API Route → `AgentService.process_query()` → LangGraph 에이전트 (SSE 스트리밍 응답)

- `app/main.py` — FastAPI 앱, CORS, 로깅 미들웨어. 모든 라우트는 `settings.API_V1_PREFIX` (기본 `/api/v1`) 하위에 마운트
- `app/api/routes/chat.py` — `POST /api/v1/chat` SSE 스트리밍 엔드포인트
- `app/api/routes/threads.py` — 대화 스레드/즐겨찾기 질문 조회 API
- `app/services/agent_service.py` — LangGraph 에이전트 실행 및 스트리밍 청크 처리. `process_query()`는 async generator로 SSE 이벤트를 yield
- `app/agents/cosmetic_agent.py` — ReAct 에이전트 정의. `create_agent()` + 3개 검색 도구(`search_safety`, `search_risk`, `search_labeling`)로 Elasticsearch 인덱스별 검색
- `app/agents/prompts.py` — 시스템 프롬프트
- `app/core/config.py` — `pydantic-settings` 기반 설정. `.env` 파일에서 로드, `env_nested_delimiter="__"`로 중첩 설정 지원
- `app/data/` — JSON 기반 스레드/즐겨찾기 데이터 저장소
- `scripts/ingest.py` — 식약처 PDF를 청크 분할 후 Elasticsearch에 임베딩 인덱싱

**에이전트 응답 구조:** `ChatResponse` Pydantic 모델 (message_id, content, metadata)이 `ToolStrategy`로 구조화된 출력 강제

## Key Conventions

- 환경 변수: `env.sample`을 `.env`로 복사 후 설정 (`OPENAI_API_KEY`, `OPENAI_MODEL`, `ES_URL/USERNAME/PASSWORD` 등)
- 대화 문맥 유지: `InMemorySaver` 체크포인터 + `thread_id` 기반
- SSE 스트리밍 프로토콜: `step` 필드로 이벤트 타입 구분 (`"model"`, `"tools"`, `"done"`)
- Python 3.11~3.13
