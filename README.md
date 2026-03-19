# 화장품 성분 상담 AI 에이전트 (Cosmetic Ingredient Advisor)

LangChain + LangGraph 기반의 **화장품 성분 전문 AI 에이전트** 서버입니다.
ReAct(Reasoning and Acting) 패턴을 활용하여 사용자의 화장품 성분 관련 질문에 대해 도구(Tool)를 자율적으로 선택하고 실행합니다.

식약처 공공데이터포털 REST API를 통해 성분 기본 정보와 사용제한 여부를 실시간으로 조회하고, Elasticsearch 벡터 검색을 통해 학술 논문/가이드라인 기반의 심화 정보를 제공합니다.

## 기술 스택

| 분류 | 기술 |
|------|------|
| **Backend Framework** | FastAPI 0.104+ |
| **Agent Framework** | LangChain v1.2 (`create_agent`, `middleware`) |
| **LLM** | OpenAI GPT-4.1 (`langchain-openai`) |
| **검색 엔진** | Elasticsearch 9.x (벡터 유사도 검색) |
| **ES 연동** | `langchain-elasticsearch` (`ElasticsearchStore`) |
| **임베딩** | OpenAI `text-embedding-3-small` |
| **외부 API** | 공공데이터포털 REST API (2개 서비스) |
| **PDF 파싱** | pymupdf4llm (2단 컬럼 레이아웃 지원) |
| **HTTP Client** | httpx |
| **설정 관리** | pydantic-settings (`.env` 기반) |
| **패키지 관리** | uv |
| **Python** | 3.11 ~ 3.13 |

## 아키텍처

```
┌─────────────┐     SSE Stream      ┌──────────────────────┐
│  React UI   │ ◄──────────────────► │  FastAPI Server      │
│ (port 5173) │   POST /api/v1/chat  │  (port 8000)         │
└─────────────┘                      │                      │
                                     │  ┌────────────────┐  │
                                     │  │ AgentService    │  │
                                     │  │ (SSE 스트리밍)   │  │
                                     │  └───────┬────────┘  │
                                     │          │           │
                                     │  ┌───────▼────────┐  │
                                     │  │ ReAct Agent     │  │
                                     │  │ (create_agent)  │  │
                                     │  │                 │  │
                                     │  │ ┌─────────────┐ │  │
                                     │  │ │ GPT-4.1 LLM │ │  │
                                     │  │ └──────┬──────┘ │  │
                                     │  │        │        │  │
                                     │  │  ┌─────▼─────┐  │  │
                                     │  │  │ Middleware │  │  │
                                     │  │  │ (에러 처리) │  │  │
                                     │  │  └─────┬─────┘  │  │
                                     │  │        │        │  │
                                     │  │   Tool 선택/실행  │  │
                                     │  └──┬─────┬─────┬─┘  │
                                     └─────┼─────┼─────┼─────┘
                                           │     │     │
                              ┌────────────┘     │     └────────────┐
                              │                  │                  │
                         ┌────▼────┐   ┌────────▼────────┐   ┌────▼──────────┐
                         │ 원료성분 │   │ 사용제한 원료    │   │ PDF 지식 베이스│
                         │ 정보 API │   │ 정보 API        │   │ (ES 벡터 검색) │
                         └─────────┘   └─────────────────┘   └───────────────┘
                            식약처 공공데이터포털                Elasticsearch
```

### 요청-응답 흐름

1. 사용자가 React UI에서 메시지를 전송합니다.
2. `POST /api/v1/chat`으로 `{ thread_id, message }` 형태의 요청이 전달됩니다.
3. `AgentService`가 모듈 수준 싱글턴 에이전트를 사용하여 `astream(stream_mode="updates")`으로 스트리밍을 시작합니다.
4. 에이전트는 LLM이 판단한 도구를 실행하고, 각 단계를 SSE(Server-Sent Events)로 실시간 전달합니다.
5. 도구 실행 중 예외가 발생하면 `@wrap_tool_call` middleware가 잡아서 ToolMessage로 변환하여 대화가 깨지지 않도록 합니다.
6. 최종 응답이 생성되면 `"step": "done"` 이벤트로 전달됩니다.

### SSE 스트림 이벤트 포맷

```jsonc
// 1) 도구 호출 시작
{"step": "model", "tool_calls": ["search_ingredient"]}

// 2) 도구 실행 결과
{"step": "tools", "name": "search_ingredient", "content": "...검색 결과..."}

// 3) 최종 응답
{"step": "done", "message_id": "uuid", "role": "assistant", "content": "답변 내용", "metadata": {}, "created_at": "..."}
```

## Middleware — 도구 에러 처리

`create_agent`의 `middleware` 파라미터를 활용하여 모든 도구의 예외를 일괄 처리합니다.

### 왜 필요한가?

`create_agent` 내부의 ToolNode는 도구 실행 예외를 기본적으로 다시 raise합니다. 예외가 처리되지 않으면:

1. `tool_call`에 대한 `ToolMessage`가 생성되지 않음
2. 다음 LLM 호출 시 OpenAI가 "tool_call_id에 대한 응답이 없다"고 거부
3. 해당 thread의 대화가 완전히 깨짐

`@wrap_tool_call` middleware가 예외를 잡아 `ToolMessage`로 변환하여 대화 흐름을 유지합니다.

### 에러 분류 기준

| 에러 유형 | 사용자 메시지 | 사용자 행동 |
|----------|-------------|-----------|
| `httpx.TimeoutException` | 응답 시간 초과 | 재시도 |
| `httpx.NetworkError` | 서비스 연결 불가 | 대기 |
| 기타 `Exception` | 일반 오류 | - |

## 에이전트 도구 (Tools)

에이전트는 사용자 질문을 분석하여 아래 3개 도구 중 적절한 것을 자동으로 선택합니다.

### Tool 1: `search_ingredient` — 원료성분정보 조회

| 항목 | 내용 |
|------|------|
| **데이터 출처** | 식품의약품안전처_화장품 원료성분정보 API |
| **입력** | `ingredient_name`: 성분명 (예: `"나이아신아마이드"`, `"레티놀"`) |
| **출력** | 표준명, 영문명, CAS번호, 기원 및 정의, 이명 (최대 5건) |
| **용도** | "이 성분이 뭐야?", "영문명이 뭐야?" 등 성분 기본 정보 질문 |

### Tool 2: `search_restricted_ingredient` — 사용제한 원료정보 조회

| 항목 | 내용 |
|------|------|
| **데이터 출처** | 식품의약품안전처_화장품 사용제한 원료정보 API |
| **입력** | `ingredient_name`: 성분명 (예: `"하이드로퀴논"`, `"파라벤"`) |
| **출력** | 규제 구분(금지/제한), 표준명, 영문명, 고시원료명, 제한사항, 단서조항, 배합제한국가 |
| **용도** | "이 성분 써도 돼?", "안전한 성분이야?" 등 안전성 질문 |

### Tool 3: `search_cosmetic_knowledge` — PDF 지식 베이스 검색

| 항목 | 내용 |
|------|------|
| **데이터 출처** | Elasticsearch (`edu-cosmetic` 인덱스, 14개 PDF → 827개 청크) |
| **검색 방식** | 벡터 유사도 검색 (`OpenAI text-embedding-3-small`) |
| **입력** | `query`: 검색 질문 (예: `"나이아신아마이드 미백 작용 원리"`) |
| **출력** | 상위 5개 문서 (출처 파일명, 본문 500자) |
| **용도** | 효능/작용 원리, 성분 간 궁합, 피부 타입별 적합성 등 심화 질문 |
| **문서 출처** | 학술 논문 (대한화장품학회지 등), 식약처 가이드라인, 화장품원료기준 성분사전 |

### 도구 사용 전략

| 질문 유형 | 사용 도구 |
|---|---|
| "이 성분이 뭐야?" (기본 정보) | `search_ingredient` |
| "써도 돼?" (안전성) | `search_ingredient` → `search_restricted_ingredient` |
| "왜 효과가 있어?" (작용 원리) | `search_cosmetic_knowledge` |
| "같이 써도 돼?" (성분 궁합) | `search_cosmetic_knowledge` |
| "민감성 피부에 괜찮아?" (피부 타입) | `search_cosmetic_knowledge` |
| 복합 질문 | 여러 도구 순차 호출 |

## PDF 인덱싱

### 인덱싱 파이프라인

`scripts/store_data.py`를 실행하여 PDF 문서를 Elasticsearch에 인덱싱합니다.

```bash
uv run python scripts/store_data.py
```

파이프라인 흐름:

```
docs/pdfs/*.pdf
    │
    ▼ pymupdf4llm (2단 컬럼 레이아웃 지원)
텍스트 추출 (마크다운)
    │
    ▼ RecursiveCharacterTextSplitter (800자, 200자 overlap)
청크 분할
    │
    ▼ OpenAI text-embedding-3-small
벡터 임베딩 (1536차원)
    │
    ▼ ElasticsearchStore.from_documents()
ES edu-cosmetic 인덱스에 저장
```

### 인덱싱된 문서

| 분류 | PDF | 청크 수 |
|---|---|---|
| 성분 효능 | 미백 효능 화장품 임상적 특성, 펩타이드 항산화/항노화, PDRN 항염증 등 | ~300 |
| 성분 궁합/제형 | 화장품 표시광고 실증 규정, Cosmetic Science 등 | ~200 |
| 피부 타입/안전성 | 보습 효과 비교, 세라마이드 피부장벽, 접촉피부염 첩포검사 등 | ~150 |
| 가이드라인 | 미백/주름개선 유효성평가, 위해평가 가이드라인, 성분사전 | ~180 |

## 환경 준비 및 설치

### 1. 사전 요구사항

- Python 3.11 이상 3.13 이하
- `uv` 패키지 매니저:
  ```bash
  # macOS / Linux / Windows (WSL)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. 의존성 설치

```bash
cd ai-agent
uv sync
```

실행 후 프로젝트 디렉토리에 `.venv` 폴더가 생성됩니다.

### 3. 환경 변수 설정

```bash
cp env.sample .env
```

`.env` 파일을 열고 아래 항목을 설정합니다:

```env
# API 라우트 prefix
API_V1_PREFIX=/api/v1

# CORS 허용 Origin (React UI 주소)
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# =====================================================
# OpenAI 설정
# =====================================================
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1

# =====================================================
# Elasticsearch 설정
# =====================================================
ES_URL=https://your-elasticsearch-url
ES_USERNAME=elastic
ES_PASSWORD=your_password
ES_INDEX=edu-cosmetic

# =====================================================
# 공공데이터포털 API 키
# https://www.data.go.kr 에서 아래 2개 서비스 활용 신청 후 발급
# - 식품의약품안전처_화장품 원료성분정보
# - 식품의약품안전처_화장품 사용제한 원료정보
# =====================================================
PUBLIC_DATA_API_KEY=your_api_key

# DeepAgents 설정
DEEPAGENT_RECURSION_LIMIT=20
```

### 4. 공공데이터포털 API 키 발급 방법

1. [공공데이터포털](https://www.data.go.kr) 회원가입 및 로그인
2. 아래 2개 API 서비스를 각각 검색하여 **활용 신청**
   - [식품의약품안전처_화장품 원료성분정보](https://www.data.go.kr/data/15111774/openapi.do) (성분 기본 정보)
   - [식품의약품안전처_화장품 사용제한 원료정보](https://www.data.go.kr/data/15111772/openapi.do) (사용 제한/금지 여부)
3. 승인 후 마이페이지에서 **일반 인증키 (Decoding)** 를 복사
4. `.env` 파일의 `PUBLIC_DATA_API_KEY`에 설정 (2개 서비스 모두 동일 키 사용)

> **참고**: 신규 발급 후 API가 실제로 활성화되기까지 최대 1~2시간이 소요될 수 있습니다.

### 5. PDF 인덱싱 (최초 1회)

```bash
uv run python scripts/store_data.py
```

`docs/pdfs/` 디렉토리의 PDF 파일을 Elasticsearch `edu-cosmetic` 인덱스에 임베딩하여 저장합니다.

### 6. 서버 실행

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API 문서 (Swagger UI): http://localhost:8000/docs
- 헬스 체크: http://localhost:8000/health

### 7. UI 연동 (선택)

별도의 React UI 프로젝트와 함께 사용합니다.

```bash
cd ui
npm install
npm run dev    # http://localhost:5173
```

## API 엔드포인트

| Method | Path | 설명 | 요청 Body |
|--------|------|------|-----------|
| `GET` | `/` | API 정보 | - |
| `GET` | `/health` | 헬스 체크 | - |
| `POST` | `/api/v1/chat` | 채팅 (SSE 스트리밍) | `{ "thread_id": "uuid", "message": "질문" }` |
| `GET` | `/api/v1/favorites/questions` | 즐겨찾기 질문 목록 | - |
| `GET` | `/api/v1/threads` | 대화 목록 조회 | - |
| `GET` | `/api/v1/threads/{thread_id}` | 대화 상세 조회 | - |

### 채팅 API 사용 예시

```bash
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6", "message": "나이아신아마이드가 뭐야?"}'
```

## 대화 메모리

`MemorySaver` (인메모리) 기반 체크포인터를 사용하여 동일한 `thread_id`의 대화 문맥을 유지합니다.

- 같은 `thread_id`로 요청하면 이전 대화를 기억하고 맥락에 맞는 답변을 생성합니다.
- 서버 재시작 시 대화 기록이 초기화됩니다.
- 향후 `SqliteSaver` 또는 `PostgresSaver`로 교체하여 영구 저장이 가능합니다.

## 프로젝트 구조

```
ai-agent/
├── app/
│   ├── main.py                         # FastAPI 앱 진입점, CORS, 미들웨어 설정
│   ├── agents/
│   │   ├── cosmetic_agent.py           # 에이전트 팩토리 (create_agent + middleware)
│   │   ├── middleware.py               # @wrap_tool_call 에러 처리 middleware
│   │   ├── prompts.py                  # 시스템 프롬프트 (도구 설명, 답변 규칙)
│   │   └── tools.py                    # 3개 도구 정의 (API 2개 + ES 벡터 검색 1개)
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py                 # POST /api/v1/chat (SSE 스트리밍 응답)
│   │       └── threads.py              # GET /api/v1/threads (대화 목록/상세 조회)
│   ├── core/
│   │   └── config.py                   # 환경 설정 (pydantic-settings, .env 로드)
│   ├── models/
│   │   ├── chat.py                     # ChatRequest / ChatResponse 모델
│   │   └── threads.py                  # 대화 스레드 모델
│   ├── services/
│   │   ├── agent_service.py            # 에이전트 싱글턴 + SSE 스트리밍 + Opik 트레이싱
│   │   ├── conversation_service.py     # 대화 세션 관리
│   │   └── threads_service.py          # 대화 목록 JSON 조회
│   ├── data/                           # JSON 기반 스레드/즐겨찾기 데이터 저장소
│   └── utils/
│       ├── logger.py                   # 커스텀 로거 + @log_execution 데코레이터
│       └── read_json.py                # JSON 파일 읽기 유틸리티
├── scripts/
│   └── store_data.py                   # PDF → ES 인덱싱 파이프라인
├── docs/
│   └── pdfs/                           # 인덱싱할 PDF 문서 (학술 논문, 가이드라인)
├── tests/                              # pytest 테스트 (8개 시나리오)
├── env.sample                          # 환경 변수 샘플
├── pyproject.toml                      # 프로젝트 설정 및 의존성 (uv)
└── README.md
```

### 주요 모듈 설명

| 모듈 | 역할 |
|------|------|
| `agents/cosmetic_agent.py` | `create_agent()`로 ReAct 에이전트 생성. `ChatOpenAI(temperature=0, streaming=True)` + 도구 3개 + 시스템 프롬프트 + `response_format=ChatResponse` + 체크포인터 + middleware 조합 |
| `agents/middleware.py` | `@wrap_tool_call` 데코레이터로 도구 실행 예외를 일괄 처리. 타임아웃/네트워크/기타로 분류하여 ToolMessage 반환 |
| `agents/tools.py` | 3개 `@tool` 함수 정의. 공공데이터 API 2개 + ES 벡터 검색 1개. ES 클라이언트/벡터스토어는 모듈 수준 싱글턴 |
| `agents/prompts.py` | 화장품 성분 상담사 페르소나, 도구 사용 전략, 답변 규칙을 포함한 시스템 프롬프트 |
| `services/agent_service.py` | 모듈 수준 싱글턴 에이전트. `MemorySaver` 체크포인터로 대화 기록 유지. `recursion_limit` 적용. Opik 트레이싱 연동. `astream(stream_mode="updates")`으로 SSE 이벤트 변환 |
| `scripts/store_data.py` | PDF 인덱싱 파이프라인. pymupdf4llm → 청크 분할 → OpenAI 임베딩 → ES 저장. 인덱스 중복 체크 포함 |
| `api/routes/chat.py` | SSE `StreamingResponse` 생성. 초기 "Planning" 이벤트 전송 후 에이전트 스트림 연결 |

## 질문 예시

### 성분 기본 정보
- "나이아신아마이드가 뭐야?"
- "레티놀이 뭔지 알려줘"

### 성분 안전성
- "레티놀 써도 괜찮아?"
- "파라벤 안전한 성분이야?"
- "하이드로퀴논 사용 제한 있어?"

### 효능/작용 원리 (ES 검색)
- "나이아신아마이드가 미백에 좋다는데 왜 그런 거야?"
- "히알루론산이 어떤 효과가 있어?"

### 성분 궁합 (ES 검색)
- "비타민C 세럼이랑 레티놀 같이 쓰고 있는데 괜찮을까?"
- "AHA랑 레티놀 같이 써도 돼?"

### 피부 타입 (ES 검색)
- "민감성 피부에 레티놀 괜찮아?"
- "지성 피부에 좋은 보습 성분 뭐야?"

### 복합 질문
- "하이드로퀴논이 뭔지, 써도 되는 성분인지 알려줘"

### 대화 문맥 유지 (연속 질문)
- 1차: "나이아신아마이드에 대해 알려줘"
- 2차: "그 성분 사용 제한은 없어?"

## 주의사항

- 모든 답변은 **일반적인 정보 제공**을 목적으로 하며, 전문 의료 상담을 대체하지 않습니다.
- 피부 이상 반응이 있을 경우 **전문의 상담**을 권장합니다.
- 공공데이터포털 API는 **일일 10,000건** 호출 제한이 있습니다.
- `PUBLIC_DATA_API_KEY`, `ES_URL`, `ES_USERNAME`, `ES_PASSWORD`, `ES_INDEX`는 필수 설정입니다. 미설정 시 서버가 시작되지 않습니다.
