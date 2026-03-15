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


def assert_sse_done_event(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """SSE 이벤트 목록에서 done 이벤트를 찾아 반환"""
    done_events = [e for e in events if e.get("step") == "done"]
    assert len(done_events) > 0, f"done 이벤트가 없습니다. 전체 이벤트: {events}"
    return done_events[0]


@pytest.mark.order(3)
def test_case1_greeting(client: TestClient, thread_id: str):
    """
    Case 1: 일반 인사 (도구 호출 없이 응답)
    사용자 질문: "안녕하세요"
    """
    response = client.post(
        "/api/v1/chat",
        json={"thread_id": thread_id, "message": "안녕하세요"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    events = parse_sse_response(response.text)
    done = assert_sse_done_event(events)

    assert done["role"] == "assistant"
    assert len(done["content"]) > 0

    # 도구 호출 없이 바로 응답해야 함
    tool_events = [e for e in events if e.get("step") == "tools"]
    assert len(tool_events) == 0


@pytest.mark.order(4)
def test_case2_search_ingredient(client: TestClient, thread_id: str):
    """
    Case 2: 성분 기본 정보 조회 (search_ingredient 도구 호출)
    사용자 질문: "나이아신아마이드가 뭐야?"
    """
    response = client.post(
        "/api/v1/chat",
        json={"thread_id": thread_id, "message": "나이아신아마이드가 뭐야?"}
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)

    # search_ingredient 도구가 호출되었는지 확인
    model_events = [e for e in events if e.get("step") == "model"]
    tool_names = []
    for e in model_events:
        tool_names.extend(e.get("tool_calls", []))
    assert "search_ingredient" in tool_names, f"search_ingredient 호출 없음. 호출된 도구: {tool_names}"

    # 최종 응답 확인
    done = assert_sse_done_event(events)
    assert done["role"] == "assistant"
    assert len(done["content"]) > 0


@pytest.mark.order(5)
def test_case3_search_restricted(client: TestClient, thread_id: str):
    """
    Case 3: 성분 안전성 조회 (search_restricted_ingredient 도구 호출)
    사용자 질문: "레티놀 써도 괜찮아?"
    """
    response = client.post(
        "/api/v1/chat",
        json={"thread_id": thread_id, "message": "레티놀 써도 괜찮아?"}
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)

    # 안전성 질문이므로 search_ingredient 또는 search_restricted_ingredient 호출
    model_events = [e for e in events if e.get("step") == "model"]
    tool_names = []
    for e in model_events:
        tool_names.extend(e.get("tool_calls", []))
    has_tool = "search_restricted_ingredient" in tool_names or "search_ingredient" in tool_names
    assert has_tool, f"성분 관련 도구 호출 없음. 호출된 도구: {tool_names}"

    done = assert_sse_done_event(events)
    assert len(done["content"]) > 0


@pytest.mark.order(6)
def test_case4_multiturn_context(client: TestClient):
    """
    Case 4: 대화 문맥 유지 (멀티턴)
    1차: "나이아신아마이드에 대해 알려줘"
    2차: "그 성분 사용 제한은 없어?"
    """
    thread_id = str(uuid.uuid4())

    # 1차 요청
    response1 = client.post(
        "/api/v1/chat",
        json={"thread_id": thread_id, "message": "나이아신아마이드에 대해 알려줘"}
    )
    assert response1.status_code == 200
    events1 = parse_sse_response(response1.text)
    done1 = assert_sse_done_event(events1)
    assert len(done1["content"]) > 0

    # 2차 요청 — "그 성분"이 나이아신아마이드를 지칭해야 함
    response2 = client.post(
        "/api/v1/chat",
        json={"thread_id": thread_id, "message": "그 성분 사용 제한은 없어?"}
    )
    assert response2.status_code == 200
    events2 = parse_sse_response(response2.text)

    # 문맥을 이해했다면 도구 호출 시도
    model_events = [e for e in events2 if e.get("step") == "model"]
    tool_names = []
    for e in model_events:
        tool_names.extend(e.get("tool_calls", []))
    has_tool = "search_restricted_ingredient" in tool_names or "search_ingredient" in tool_names
    assert has_tool, f"문맥 유지 실패: 도구 호출 없음. 호출된 도구: {tool_names}"

    done2 = assert_sse_done_event(events2)
    assert len(done2["content"]) > 0
