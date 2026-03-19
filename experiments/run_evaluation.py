"""Opik Experiment 평가 스크립트 — 6대 평가지표 기반 에이전트 성능 진단.

6대 평가지표 (Six Core Metrics):
┌─────────────────────────────────────────────────────────────┐
│ L1: 에이전트 동작 검증 (Heuristic, LLM 호출 없음)           │
│   1. Tool Correctness   — 도구 선택 정확도                  │
│   2. Keyword Coverage    — 응답 핵심 키워드 포함률           │
│   3. Safety Compliance   — 화장품 안전성 면책 준수           │
│                                                             │
│ L2: 응답 품질 평가 (Opik 내장 LLM-as-a-Judge)              │
│   4. Answer Relevance    — 답변 관련성                      │
│   5. Hallucination       — 환각 탐지                        │
│   6. Usefulness          — 답변 유용성                      │
└─────────────────────────────────────────────────────────────┘

실행:
    uv run python experiments/run_evaluation.py            # 전체 (L1+L2, 6개 메트릭)
    uv run python experiments/run_evaluation.py --level 1  # L1만 (3개 메트릭)
    uv run python experiments/run_evaluation.py --level 2  # L1+L2 (6개 메트릭)
"""

import argparse
import asyncio
import json
import re
import uuid

from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import (
    AnswerRelevance,
    BaseMetric,
    Hallucination,
    Usefulness,
)
from opik.evaluation.metrics.score_result import ScoreResult

from app.agents.cosmetic_agent import create_cosmetic_agent
from app.core.config import settings
from app.services.agent_service import _configure_opik

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Opik 환경변수 설정
_configure_opik()

# 에이전트 생성 + Opik 트레이싱 래핑
_checkpointer = MemorySaver()
_agent = create_cosmetic_agent(checkpointer=_checkpointer)

if settings.OPIK:
    from opik.integrations.langchain import OpikTracer, track_langgraph
    _opik_tracer = OpikTracer(project_name=settings.OPIK.PROJECT)
    _agent = track_langgraph(_agent, _opik_tracer)


# ============================================================
# L1-① Tool Correctness (도구 선택 정확도)
#
# ReAct 에이전트가 질문 의도에 맞는 올바른 도구를 선택했는지 평가합니다.
# 기대 도구 집합(expected)과 실제 호출 도구 집합(actual)의 교집합 비율로 산출합니다.
# expected_tools가 비어있으면 도구를 호출하지 않아야 정답입니다.
# ============================================================
class ToolCorrectness(BaseMetric):
    name = "tool_correctness"

    def score(self, tool_calls: str, expected_tools: str, **kwargs) -> ScoreResult:
        if not expected_tools or expected_tools == "none":
            is_correct = (tool_calls == "none")
            return ScoreResult(
                name=self.name,
                value=1.0 if is_correct else 0.0,
                reason="도구 미호출 기대" if is_correct else f"불필요한 도구 호출: {tool_calls}",
            )

        expected_set = set(expected_tools.split("|"))
        actual_set = set(tool_calls.split(",")) - {"none"}
        matched = expected_set & actual_set
        score = len(matched) / len(expected_set) if expected_set else 0.0

        return ScoreResult(
            name=self.name,
            value=score,
            reason=f"기대: {expected_set}, 실제: {actual_set}, 일치: {matched}",
        )


# ============================================================
# L1-② Keyword Coverage (응답 핵심 키워드 포함률)
#
# 응답에 기대하는 핵심 키워드가 포함되어 있는지 검사합니다.
# expected_keywords에 '|'로 구분된 키워드 중 몇 개가 응답에 등장하는지
# 비율로 산출합니다. RAG Grounding의 코드 레벨 근사치입니다.
# ============================================================
class KeywordCoverage(BaseMetric):
    name = "keyword_coverage"

    def score(self, output: str, expected_keywords: str, **kwargs) -> ScoreResult:
        if not expected_keywords:
            return ScoreResult(name=self.name, value=1.0, reason="키워드 검증 불필요")

        keywords = [k.strip() for k in expected_keywords.split("|")]
        found = [k for k in keywords if k in output]
        score = len(found) / len(keywords) if keywords else 0.0

        return ScoreResult(
            name=self.name,
            value=score,
            reason=f"기대: {keywords}, 포함: {found} ({len(found)}/{len(keywords)})",
        )


# ============================================================
# L1-③ Safety Compliance (화장품 안전성 면책 준수)
#
# 화장품 성분 상담에서 안전 관련 응답에 면책 문구가 포함되어 있는지 검사합니다.
# - 안전성/제한 관련 질문: 전문의 상담 권유 문구 필요
# - 일반 정보/인사: 면제
# ============================================================
class SafetyCompliance(BaseMetric):
    name = "safety_compliance"

    SAFETY_PATTERNS = [
        r"전문의", r"피부과", r"상담", r"의사",
        r"피부.*이상", r"테스트.*후", r"패치.*테스트",
        r"주의", r"자극", r"전문가",
    ]

    # 안전 문구가 필요한 도구
    SAFETY_REQUIRED_TOOLS = {
        "search_restricted_ingredient",
        "search_cosmetic_knowledge",
    }

    def score(self, output: str, expected_tools: str, **kwargs) -> ScoreResult:
        if not expected_tools or expected_tools == "none":
            return ScoreResult(name=self.name, value=1.0, reason="일반 대화 — 안전 문구 불필요")

        expected_set = set(expected_tools.split("|"))
        needs_safety = expected_set & self.SAFETY_REQUIRED_TOOLS

        if not needs_safety:
            return ScoreResult(name=self.name, value=1.0, reason="기본 정보 조회 — 안전 문구 불필요")

        found = [p for p in self.SAFETY_PATTERNS if re.search(p, output)]
        passed = len(found) > 0

        return ScoreResult(
            name=self.name,
            value=1.0 if passed else 0.0,
            reason=f"안전 문구 {'포함' if passed else '미포함'}: {found[:3]}",
        )


# ============================================================
# 레벨별 메트릭 구성
# ============================================================
def get_metrics(level: int):
    """레벨에 따라 사용할 메트릭 목록을 반환합니다.

    L1 (3개): 코드 기반 에이전트 동작 검증 — LLM 호출 없이 빠르게 실행
    L2 (6개): L1 + Opik 내장 LLM-as-a-Judge 3개 추가
    """
    l1 = [
        ToolCorrectness(),
        KeywordCoverage(),
        SafetyCompliance(),
    ]

    l2 = [
        AnswerRelevance(require_context=False),
        Hallucination(),
        Usefulness(),
    ]

    if level == 1:
        return l1
    else:
        return l1 + l2


# ============================================================
# 에이전트 실행
# ============================================================
def run_agent(query: str) -> dict:
    """에이전트를 실행하고 응답 + 호출된 도구 목록 + context를 반환합니다."""
    thread_id = str(uuid.uuid4())

    result = asyncio.run(
        _agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": settings.DEEPAGENT_RECURSION_LIMIT,
            },
        )
    )

    messages = result.get("messages", [])

    # 호출된 도구 이름 수집 + 도구 결과(context) 수집
    tool_calls = []
    tool_results = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls.extend([tc["name"] for tc in msg.tool_calls])
        if hasattr(msg, "tool_call_id") and msg.content:
            tool_results.append(msg.content[:500])

    # 최종 응답 (마지막 AI 메시지)
    final_content = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_call_id"):
            try:
                parsed = json.loads(msg.content)
                final_content = parsed.get("content") or msg.content
            except (json.JSONDecodeError, TypeError):
                final_content = msg.content
            break

    if not final_content:
        final_content = "(응답 없음)"

    return {
        "output": final_content,
        "tool_calls": ",".join(tool_calls) if tool_calls else "none",
        "context": tool_results,
    }


def evaluation_task(x: dict) -> dict:
    """Opik evaluation task — 데이터셋 항목을 받아 에이전트 실행 결과를 반환합니다."""
    result = run_agent(x["input"])
    return {
        "output": result["output"],
        "context": result["context"],
        "tool_calls": result["tool_calls"],
        "expected_tools": x.get("expected_tools", ""),
        "expected_keywords": x.get("expected_keywords", ""),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="화장품 성분 상담 에이전트 6대 평가지표 성능 진단")
    parser.add_argument(
        "--level", type=int, default=2, choices=[1, 2],
        help="평가 레벨 (1: Heuristic 3개, 2: +LLM Judge 총 6개)",
    )
    args = parser.parse_args()

    client = Opik()
    dataset = client.get_dataset(name="cosmetic-agent-eval")
    metrics = get_metrics(args.level)

    metric_names = [m.name for m in metrics]
    print(f"\n[Level {args.level}] 6대 평가지표 중 {len(metrics)}개 적용: {metric_names}\n")

    evaluation = evaluate(
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=metrics,
        experiment_name=f"cosmetic-agent-eval-L{args.level}",
        project_name=settings.OPIK.PROJECT if settings.OPIK else None,
    )

    print(f"\nExperiment ID: {evaluation.experiment_id}")
    print("Opik 대시보드에서 결과를 확인하세요.")
