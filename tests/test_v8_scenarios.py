"""화장품 성분 상담 AI 에이전트 시나리오 테스트.

실제 OpenAI API, 공공데이터 API, Elasticsearch를 호출하여 SSE 스트리밍 파이프라인의 다양한 시나리오를 검증한다.
- Case 1: 도구 호출 없이 직접 응답 (일반 인사)
- Case 2: 단일 도구 호출 (성분 정보 조회)
- Case 3: 단일 도구 호출 (성분 안전성 조회)
- Case 4: 두 도구 모두 호출 (성분 정보 + 안전성)
- Case 5: 멀티턴 대화 (문맥 유지)
- Case 6: 존재하지 않는 성분 질문
- Case 7: ES 검색 — 성분 효능/작용 원리
- Case 8: API + ES 도구 조합 (기본 정보 + 심화 정보)
"""
import pytest
import json
import uuid
from fastapi.testclient import TestClient
from typing import List, Dict, Any


def parse_sse_response(response_text: str) -> List[Dict[str, Any]]:
    """SSE 응답을 파싱하는 헬퍼 함수"""
    events = []
    for line in response_text.strip().split('\n'):
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                events.append(json.loads(data_str))
            except json.JSONDecodeError:
                pass
    return events


def get_tool_calls(events: List[Dict[str, Any]]) -> List[str]:
    """SSE 이벤트에서 도구 호출 목록을 추출하는 헬퍼"""
    tool_calls = []
    for event in events:
        if event.get("step") == "model" and "tool_calls" in event:
            tool_calls.extend(event["tool_calls"])
    return tool_calls


def get_done_event(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """SSE 이벤트에서 최종 응답(done)을 추출하는 헬퍼"""
    done_events = [e for e in events if e.get("step") == "done"]
    assert len(done_events) >= 1, "No 'done' event found"
    return done_events[-1]


@pytest.mark.order(1)
def test_case1_no_tool_greeting(client: TestClient):
    """
    Case 1: 도구 호출 없이 직접 응답 — 일반 인사
    사용자 질문: "안녕하세요"
    기대: 도구 호출 없이 done 이벤트만 반환
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "안녕하세요"
        }
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = parse_sse_response(response.text)

    # Planning 이벤트 제외하고 실제 도구 호출이 없어야 함
    tool_calls = get_tool_calls(events)
    actual_tools = [t for t in tool_calls if t != "Planning"]
    assert len(actual_tools) == 0, f"도구 호출 없이 응답해야 하는데 {actual_tools} 호출됨"

    done = get_done_event(events)
    assert done["role"] == "assistant"
    assert len(done["content"]) > 0


@pytest.mark.order(2)
def test_case2_search_ingredient(client: TestClient):
    """
    Case 2: 단일 도구 호출 — 성분 기본 정보 조회
    사용자 질문: "나이아신아마이드가 뭐야?"
    기대: search_ingredient 호출, done 이벤트에 성분 설명 포함
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "나이아신아마이드가 뭐야?"
        }
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)
    tool_calls = get_tool_calls(events)
    done = get_done_event(events)

    assert "search_ingredient" in tool_calls, f"search_ingredient not found in {tool_calls}"
    assert done["role"] == "assistant"
    assert len(done["content"]) > 0


@pytest.mark.order(3)
def test_case3_search_restricted(client: TestClient):
    """
    Case 3: 단일 도구 호출 — 성분 안전성 조회
    사용자 질문: "레티놀 써도 괜찮아?"
    기대: search_ingredient 또는 search_restricted_ingredient 호출
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "레티놀 써도 괜찮아?"
        }
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)
    tool_calls = get_tool_calls(events)
    done = get_done_event(events)

    has_tool = "search_restricted_ingredient" in tool_calls or "search_ingredient" in tool_calls
    assert has_tool, f"성분 관련 도구 호출 없음. 호출된 도구: {tool_calls}"
    assert len(done["content"]) > 0


@pytest.mark.order(4)
def test_case4_both_tools(client: TestClient):
    """
    Case 4: 두 도구 모두 호출 — 성분 정보 + 안전성
    사용자 질문: "하이드로퀴논이 뭔지, 써도 되는 성분인지 알려줘"
    기대: search_ingredient, search_restricted_ingredient 모두 호출
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "하이드로퀴논이 뭔지, 써도 되는 성분인지 알려줘"
        }
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)
    tool_calls = get_tool_calls(events)
    done = get_done_event(events)

    assert "search_ingredient" in tool_calls, f"search_ingredient not found in {tool_calls}"
    assert "search_restricted_ingredient" in tool_calls, f"search_restricted_ingredient not found in {tool_calls}"
    assert len(done["content"]) > 0


@pytest.mark.order(5)
def test_case5_multiturn_context(client: TestClient):
    """
    Case 5: 멀티턴 대화 (동일 thread_id로 문맥 유지)
    1차: "나이아신아마이드에 대해 알려줘"
    2차: "그 성분 사용 제한은 없어?"
    기대: 2차에서 "그 성분"을 나이아신아마이드로 이해하고 도구 호출
    """
    thread_id = str(uuid.uuid4())

    # 1차 요청
    response1 = client.post(
        "/api/v1/chat",
        json={"thread_id": thread_id, "message": "나이아신아마이드에 대해 알려줘"}
    )
    assert response1.status_code == 200
    events1 = parse_sse_response(response1.text)
    done1 = get_done_event(events1)
    assert len(done1["content"]) > 0

    # 2차 요청 — 문맥 유지
    response2 = client.post(
        "/api/v1/chat",
        json={"thread_id": thread_id, "message": "그 성분 사용 제한은 없어?"}
    )
    assert response2.status_code == 200
    events2 = parse_sse_response(response2.text)
    tool_calls2 = get_tool_calls(events2)
    done2 = get_done_event(events2)

    has_tool = "search_restricted_ingredient" in tool_calls2 or "search_ingredient" in tool_calls2
    assert has_tool, f"문맥 유지 실패: 도구 호출 없음. 호출된 도구: {tool_calls2}"
    assert len(done2["content"]) > 0


@pytest.mark.order(6)
def test_case6_unknown_ingredient(client: TestClient):
    """
    Case 6: 존재하지 않는 성분 질문
    사용자 질문: "ㅁㄴㅇㄹ 성분 안전해?"
    기대: 도구 호출 후 "찾을 수 없다" 또는 "등록된 성분이 아니다" 류의 응답
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "ㅁㄴㅇㄹ 성분 안전해?"
        }
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)
    tool_calls = get_tool_calls(events)
    done = get_done_event(events)

    # 도구를 호출해서 확인 시도는 해야 함
    assert len(tool_calls) > 0, "존재 여부 확인을 위해 도구 호출이 필요합니다"
    assert len(done["content"]) > 0


@pytest.mark.order(7)
def test_case7_cosmetic_knowledge_efficacy(client: TestClient):
    """
    Case 7: ES 검색 — 성분 효능/작용 원리
    사용자 질문: "나이아신아마이드가 미백에 좋다는데 왜 그런 거야?"
    기대: search_cosmetic_knowledge 호출, 학술 근거 기반 답변
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "나이아신아마이드가 미백에 좋다는데 왜 그런 거야?"
        }
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)
    tool_calls = get_tool_calls(events)
    done = get_done_event(events)

    assert "search_cosmetic_knowledge" in tool_calls, f"search_cosmetic_knowledge not found in {tool_calls}"
    assert done["role"] == "assistant"
    assert len(done["content"]) > 0


@pytest.mark.order(8)
def test_case8_api_and_es_combined(client: TestClient):
    """
    Case 8: API + ES 도구 조합 — 기본 정보와 심화 정보를 동시에 요구
    사용자 질문: "나이아신아마이드가 뭔지, 그리고 왜 미백에 효과가 있는지 알려줘"
    기대: search_ingredient (API) + search_cosmetic_knowledge (ES) 모두 호출
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "나이아신아마이드가 뭔지, 그리고 왜 미백에 효과가 있는지 알려줘"
        }
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)
    tool_calls = get_tool_calls(events)
    done = get_done_event(events)

    assert "search_ingredient" in tool_calls, f"search_ingredient not found in {tool_calls}"
    assert "search_cosmetic_knowledge" in tool_calls, f"search_cosmetic_knowledge not found in {tool_calls}"
    assert done["role"] == "assistant"
    assert len(done["content"]) > 0
