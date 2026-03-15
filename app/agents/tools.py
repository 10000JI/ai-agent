"""
화장품 성분 상담 도구 — 식약처 공공데이터 API 기반.
"""

import httpx
from langchain_core.tools import tool

from app.core.config import settings


def _call_api(url: str, params: dict) -> dict:
    """공공데이터포털 API 공통 호출"""
    params["serviceKey"] = settings.PUBLIC_DATA_API_KEY
    params["type"] = "json"
    resp = httpx.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


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
