"""
화장품 성분 상담 도구 — 식약처 공공데이터 API + Elasticsearch PDF 지식 베이스.
"""

import httpx
from elasticsearch import Elasticsearch
from langchain_core.tools import tool
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


# ============================================================
# 공공데이터 API 공통 헬퍼
# ============================================================

def _call_api(url: str, params: dict) -> dict:
    """공공데이터포털 API 공통 호출"""
    params["serviceKey"] = settings.PUBLIC_DATA_API_KEY
    params["type"] = "json"
    resp = httpx.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ============================================================
# Elasticsearch 클라이언트 및 벡터 스토어 (모듈 수준 싱글턴)
# ============================================================

_es_client = Elasticsearch(
    settings.ES_URL,
    basic_auth=(settings.ES_USERNAME, settings.ES_PASSWORD),
    verify_certs=False,
)

_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.OPENAI_API_KEY,
)

_vector_store = ElasticsearchStore(
    index_name=settings.ES_INDEX,
    embedding=_embeddings,
    client=_es_client,
)


# ============================================================
# Tool 1: 성분 기본 정보 조회 (공공데이터 API)
# ============================================================

@tool
def search_ingredient(ingredient_name: str) -> str:
    """화장품 성분의 기본 정보를 조회합니다. 성분명(한글)으로 검색하면 표준명, 영문명, CAS번호, 기원 및 정의, 이명을 확인할 수 있습니다.

    Args:
        ingredient_name: 검색할 화장품 성분명 (예: '나이아신아마이드', '레티놀', '히알루론산')
    """
    data = _call_api(
        settings.INGREDIENT_API_URL,
        {"INGR_KOR_NAME": ingredient_name, "pageNo": "1", "numOfRows": "5"},
    )
    items = data.get("body", {}).get("items", [])
    if not items:
        return f"'{ingredient_name}'에 대한 성분 정보를 찾지 못했습니다."

    results = []
    for item in items:
        results.append(
            f"- 표준명: {item.get('INGR_KOR_NAME', '-')}\n"
            f"  영문명: {item.get('INGR_ENG_NAME', '-')}\n"
            f"  CAS번호: {item.get('CAS_NO', '-')}\n"
            f"  기원 및 정의: {item.get('ORIGIN_MAJOR_KOR_NAME', '-')}\n"
            f"  이명: {item.get('INGR_SYNONYM', '-')}"
        )
    return "\n\n".join(results)


# ============================================================
# Tool 2: 사용제한 원료정보 조회 (공공데이터 API)
# ============================================================

@tool
def search_restricted_ingredient(ingredient_name: str) -> str:
    """화장품 성분의 사용 제한/금지 여부를 조회합니다. 성분명(한글)으로 검색하면 규제 구분(금지/제한), 제한사항, 단서조항, 배합제한국가 등을 확인할 수 있습니다.

    Args:
        ingredient_name: 검색할 화장품 성분명 (예: '레티놀', '하이드로퀴논', '파라벤')
    """
    data = _call_api(
        settings.RESTRICTED_INGREDIENT_API_URL,
        {"pageNo": "1", "numOfRows": "100"},
    )
    items = data.get("body", {}).get("items", [])
    if not items:
        return f"'{ingredient_name}'에 대한 사용제한 정보를 찾지 못했습니다."

    matched = [
        item for item in items
        if ingredient_name in (item.get("INGR_STD_NAME") or "")
        or ingredient_name in (item.get("INGR_SYNONYM") or "")
        or ingredient_name in (item.get("NOTICE_INGR_NAME") or "")
    ]

    if not matched:
        return f"'{ingredient_name}'은(는) 현재 사용제한 원료 목록에 없습니다. (안전하게 사용 가능한 성분일 수 있습니다)"

    results = []
    for item in matched:
        results.append(
            f"- 구분: {item.get('REGULATE_TYPE', '-')}\n"
            f"  표준명: {item.get('INGR_STD_NAME', '-')}\n"
            f"  영문명: {item.get('INGR_ENG_NAME', '-')}\n"
            f"  고시원료명: {item.get('NOTICE_INGR_NAME', '-')}\n"
            f"  제한사항: {item.get('LIMIT_COND', '-')}\n"
            f"  단서조항: {item.get('PROVIS_ATRCL', '-')}\n"
            f"  배합제한국가: {item.get('COUNTRY_NAME', '-')}"
        )
    return "\n\n".join(results)


# ============================================================
# Tool 3: 화장품 성분 지식 검색 (Elasticsearch PDF 지식 베이스)
# ============================================================

@tool
def search_cosmetic_knowledge(query: str) -> str:
    """화장품 성분의 효능, 작용 원리, 성분 간 궁합, 피부 타입별 적합성 등 심화 정보를 검색합니다. 학술 논문과 전문 가이드라인 기반의 정보를 제공합니다.

    Args:
        query: 검색할 질문 (예: '나이아신아마이드 미백 작용 원리', '레티놀 비타민C 함께 사용', '민감성 피부 세라마이드')
    """
    docs = _vector_store.similarity_search(query, k=5)

    if not docs:
        return "관련 문서를 찾지 못했습니다."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "알 수 없음")
        content = doc.page_content[:500]
        results.append(f"[문서 {i}] (출처: {source})\n{content}")

    return "\n\n---\n\n".join(results)
