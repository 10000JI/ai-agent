# 화장품 성분 상담 AI 에이전트 설계서

> 작성일: 2026-03-12
> 프로젝트: /Users/n-mjkim/IdeaProjects/ai-agent

---

## 1. 개요

식약처 공개 PDF 문서를 Elasticsearch에 인덱싱하고, LangChain ReAct 에이전트가 하이브리드 검색(BM25 + kNN + RRF)으로 관련 정보를 찾아 답변하는 화장품 성분 상담 도우미.

### 타겟 사용자
- 20-30대 여성, 화장품 성분에 관심 있는 일반 소비자

### 주요 질문 시나리오
- 성분 간 궁합/조합: "레티놀이랑 나이아신아마이드 같이 써도 돼?"
- 피부 타입별 추천: "건성 피부에 좋은 성분 추천해줘"
- 성분표 분석: "이 성분표 분석해줘: 정제수, 글리세린, ..."
- 안전성 질문: "이 성분 위험해?"

### 답변 톤
- 기본: 친근하고 쉬운 말투
- 근거/출처는 정확히 표시 (문서명, 페이지 번호)

---

## 2. 전체 아키텍처

```
┌──────────────┐     SSE      ┌──────────────────────────────────┐
│   UI (React) │ ──────────→  │         FastAPI 서버              │
│  기존 ui 재사용│ ← ────────  │                                  │
└──────────────┘              │  ┌────────────────────────────┐  │
                              │  │   LangChain ReAct Agent     │  │
                              │  │                            │  │
                              │  │  Tools:                    │  │
                              │  │   ├─ search_safety()       │  │
                              │  │   ├─ search_risk()         │  │
                              │  │   └─ search_labeling()     │  │
                              │  └─────────┬──────────────────┘  │
                              └────────────┼─────────────────────┘
                                           ↓
                              ┌─────────────────────────────────┐
                              │  Elasticsearch (교육용 서버)      │
                              │  https://elasticsearch-edu.      │
                              │  didim365.app                   │
                              │                                 │
                              │  인덱스:                         │
                              │   ├─ cosmetic-safety            │
                              │   ├─ cosmetic-risk              │
                              │   └─ cosmetic-labeling          │
                              └─────────────────────────────────┘
```

### 실행 흐름
1. 사용자가 UI에서 질문 입력
2. FastAPI → ReAct 에이전트가 질문을 분석하여 적절한 도구(인덱스) 선택
3. ES 하이브리드 검색(BM25 + kNN)으로 관련 문단 검색
4. 검색 결과 기반으로 LLM이 친근한 톤 + 출처 포함 답변 생성
5. SSE 스트리밍으로 UI에 실시간 전달

---

## 3. PDF 인덱싱 파이프라인

### 데이터 소스 (식약처 PDF)

| PDF | 내용 | 인덱스 |
|-----|------|--------|
| 화장품 안전기준 등에 관한 규정 해설서 | 배합금지/사용제한 원료, 한도 | cosmetic-safety |
| 화장품 위해평가 가이드라인 | 성분 위해성 평가 방법, 기준 | cosmetic-risk |
| 화장품 표시·광고 관리 지침 | 성분 표시 규정, 광고 기준 | cosmetic-labeling |

### 처리 흐름

```
[PDF 파일] (data/pdfs/)
    ↓
[PyPDFLoader] — PDF를 페이지 단위로 로드
    ↓
[RecursiveCharacterTextSplitter] — 500~1000자 청크로 분할
    ↓
[메타데이터 태깅]
    {
      "source": "화장품_안전기준_해설서.pdf",
      "page": 12,
      "doc_type": "safety"
    }
    ↓
[OpenAI Embedding] — 각 청크를 벡터로 변환
    ↓
[ElasticsearchStore] — 문서 유형별 인덱스에 저장
```

### ES 인덱스 매핑 (공통)

| 필드 | 타입 | 설명 |
|------|------|------|
| text | text | 청크 원문 (BM25 검색용) |
| vector | dense_vector | 임베딩 벡터 (kNN 검색용) |
| metadata.source | keyword | 원본 PDF 파일명 |
| metadata.page | integer | 페이지 번호 |

### 프로젝트 내 위치

```
ai-agent/
├── app/           ← 기존 서버 코드
├── scripts/
│   └── ingest.py  ← PDF 인덱싱 스크립트 (1회 실행)
└── data/
    └── pdfs/      ← PDF 파일 저장 위치
```

---

## 4. ReAct 에이전트 구성

### 도구 (Tools)

| 도구명 | 인덱스 | 사용 시점 |
|--------|--------|----------|
| search_safety | cosmetic-safety | 성분 안전성, 배합 금지, 사용 제한, 성분 궁합 |
| search_risk | cosmetic-risk | 성분 위험도, 피부 자극, 위해성 평가 |
| search_labeling | cosmetic-labeling | 성분표 읽는 법, 표시 순서, 광고 규정 |

### 하이브리드 검색

각 도구 내부에서 동일한 검색 로직 사용:
- BM25: 키워드 매칭 ("레티놀" 같은 정확한 성분명)
- kNN: 의미 검색 ("주름에 좋은 성분" 같은 자연어)
- RRF: 두 결과를 융합하여 최종 랭킹

### 시스템 프롬프트

```
역할: 화장품 성분 전문 상담사
톤: 친근하고 쉬운 말투, 하지만 근거는 정확히 표시
규칙:
  - 검색 결과에 기반해서만 답변 (할루시네이션 방지)
  - 답변 끝에 출처(문서명, 페이지) 표시
  - 모르면 솔직히 "관련 자료를 찾지 못했어요" 답변
```

---

## 5. 파일 변경 계획

| 파일 | 변경 내용 |
|------|----------|
| `app/agents/dummy.py` → `app/agents/cosmetic_agent.py` | ReAct 에이전트 + 3개 검색 도구 |
| `app/agents/prompts.py` | 화장품 상담사 시스템 프롬프트 |
| `app/services/agent_service.py` | `_create_agent()`에서 새 에이전트 로드 |
| `app/core/config.py` | ES 연결 설정(ES_URL, ES_USERNAME, ES_PASSWORD) 추가 |
| `.env` | ES 연결 정보 추가 |
| `scripts/ingest.py` (신규) | PDF 파싱 → ES 인덱싱 스크립트 |
| `data/pdfs/` (신규) | 식약처 PDF 파일 저장 디렉토리 |

---

## 6. 구현 단계

### Phase 1: ES 인프라 + 인덱싱
- `.env`에 ES 연결 정보 설정
- `scripts/ingest.py` 작성
- `data/pdfs/`에 식약처 PDF 다운로드
- 인덱싱 실행 후 ES에서 데이터 확인

### Phase 2: 에이전트 구현
- `app/agents/cosmetic_agent.py` 작성
- `app/agents/prompts.py` 교체
- `app/core/config.py`에 ES 설정 추가
- `app/services/agent_service.py`에서 새 에이전트 연결

### Phase 3: 검색 고도화
- 하이브리드 검색(BM25 + kNN + RRF) 적용
- 검색 결과 품질 테스트 및 청크 사이즈 튜닝

### Phase 4: 통합 테스트
- UI 연결하여 전체 플로우 테스트
- 시나리오별 테스트
- GitHub push
