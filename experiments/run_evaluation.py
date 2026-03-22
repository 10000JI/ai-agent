"""Opik Experiment 평가 스크립트 — 6대 평가지표 기반 에이전트 성능 진단.

6대 평가지표 (Six Core Metrics):
┌─────────────────────────────────────────────────────────────┐
│ L1: 에이전트 동작 검증 (Heuristic, LLM 호출 없음)           │
│   1. Tool Correctness    — 도구 선택 정확도                 │
│   2. Keyword Coverage    — 응답 핵심 키워드 포함률           │
│   3. Safety Compliance   — 화장품 안전성 면책 준수           │
│                                                             │
│ L2: 응답 품질 평가 (Opik 내장 LLM-as-a-Judge + G-Eval)     │
│   4. Answer Relevance    — 답변 관련성 (Opik 내장)          │
│   5. Hallucination       — 환각 탐지 (Opik 내장)            │
│   6. Cosmetic Accuracy   — 화장품 성분 정확성 (G-Eval)      │
└─────────────────────────────────────────────────────────────┘

참고:
  - Opik LangGraph: https://www.comet.com/docs/opik/integrations/langgraph
  - DeepEval G-Eval: https://deepeval.com/docs/metrics-llm-evals
  - DeepEval Agent Metrics: https://deepeval.com/guides/guides-ai-agent-evaluation-metrics

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
)
from opik.evaluation.metrics.score_result import ScoreResult

from deepeval.metrics import GEval as DeepEvalGEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.agents.cosmetic_agent import create_cosmetic_agent
from app.core.config import settings
from app.services.agent_service import _configure_opik

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Opik 환경변수 설정
_configure_opik()

# 에이전트 생성
_checkpointer = MemorySaver()
_agent = create_cosmetic_agent(checkpointer=_checkpointer)


# ============================================================
# L1-① Tool Correctness (도구 선택 정확도)
#
# DeepEval의 Tool Correctness 메트릭에 대응합니다.
# ReAct 에이전트가 질문 의도에 맞는 올바른 도구를 선택했는지 평가합니다.
# Precision(불필요한 도구 미호출)과 Recall(필요한 도구 호출)의
# F1 스코어로 산출하여 과잉 호출도 감점합니다.
#
# 참고: https://deepeval.com/guides/guides-ai-agent-evaluation-metrics
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

        recall = len(matched) / len(expected_set) if expected_set else 0.0
        precision = len(matched) / len(actual_set) if actual_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return ScoreResult(
            name=self.name,
            value=f1,
            reason=f"기대: {expected_set}, 실제: {actual_set}, 일치: {matched} (P={precision:.2f}, R={recall:.2f}, F1={f1:.2f})",
        )


# ============================================================
# L1-② Keyword Coverage (응답 핵심 키워드 포함률)
#
# RAG 파이프라인에서 Faithfulness/Grounding에 해당하는 코드 레벨 메트릭입니다.
# expected_keywords에 '|'로 구분된 키워드 중 응답에 포함된 비율로 산출합니다.
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
# 도구별로 안전 문구 기준을 분리하여 평가합니다.
#
# - search_restricted_ingredient: 성분 제한/금지 → 전문의/패치테스트 권유
# - search_cosmetic_knowledge: 효능/궁합/피부타입 → 피부 이상 시 전문가 상담 권유
# ============================================================
class SafetyCompliance(BaseMetric):
    name = "safety_compliance"

    # 강한 안전 문구: 구체적인 행동 지시 (전문의 방문, 패치테스트 등)
    # 약한 안전 문구: "주의", "자극" 등 일반적 경고
    # 강한 패턴 1개 이상 매칭 시 1.0, 약한 패턴만 매칭 시 0.5, 없으면 0.0
    TOOL_SAFETY_CRITERIA = {
        "search_restricted_ingredient": {
            "label": "성분 안전성",
            "strong_patterns": [
                r"전문의", r"피부과", r"의사",
                r"패치.*테스트", r"상담.*후.*사용",
            ],
            "weak_patterns": [
                r"주의", r"자극", r"상담",
            ],
        },
        "search_cosmetic_knowledge": {
            "label": "성분 효능/궁합",
            "strong_patterns": [
                r"전문의", r"피부과", r"전문가",
                r"패치.*테스트", r"소량.*테스트",
            ],
            "weak_patterns": [
                r"주의", r"자극", r"피부.*이상",
            ],
        },
    }

    def score(self, output: str, expected_tools: str, **kwargs) -> ScoreResult:
        if not expected_tools or expected_tools == "none":
            return ScoreResult(name=self.name, value=1.0, reason="일반 대화 — 안전 문구 불필요")

        expected_set = set(expected_tools.split("|"))
        required_tools = expected_set & set(self.TOOL_SAFETY_CRITERIA.keys())

        if not required_tools:
            return ScoreResult(name=self.name, value=1.0, reason="기본 정보 조회 — 안전 문구 불필요")

        results = []
        for tool in required_tools:
            criteria = self.TOOL_SAFETY_CRITERIA[tool]
            strong_found = [p for p in criteria["strong_patterns"] if re.search(p, output)]
            weak_found = [p for p in criteria["weak_patterns"] if re.search(p, output)]

            if strong_found:
                tool_score = 1.0
                status = "✓ 강한 안전 문구"
            elif weak_found:
                tool_score = 0.5
                status = "△ 약한 안전 문구만"
            else:
                tool_score = 0.0
                status = "✗ 안전 문구 없음"

            results.append({
                "tool": tool,
                "label": criteria["label"],
                "score": tool_score,
                "status": status,
                "matched": strong_found[:2] + weak_found[:2],
            })

        score = sum(r["score"] for r in results) / len(results)

        details = []
        for r in results:
            details.append(f"{r['label']}({r['tool']}): {r['status']} {r['matched']}")

        return ScoreResult(
            name=self.name,
            value=score,
            reason=" | ".join(details),
        )


# ============================================================
# L2-⑥ Cosmetic Accuracy (화장품 성분 정확성) — DeepEval G-Eval
#
# Opik 내장 메트릭은 범용적이라 화장품 도메인의 전문성을 판단하지 못합니다.
# DeepEval의 G-Eval(LLM-as-a-Judge + CoT)로 화장품 성분 정확성을 평가하고,
# 결과를 Opik BaseMetric 포맷으로 래핑하여 대시보드에 통합합니다.
#
# 참고: https://deepeval.com/docs/metrics-llm-evals
# ============================================================
_deepeval_cosmetic_accuracy = DeepEvalGEval(
    name="cosmetic_accuracy",
    criteria="화장품 성분 상담 AI 에이전트의 응답이 성분학적으로 정확하고 완결한지 평가합니다.",
    evaluation_steps=[
        "1. 성분 정보 정확성: 성분의 정의, 영문명, 효능 설명이 사실과 부합하는지 확인하세요. "
        "예: 나이아신아마이드 → 멜라닌 전이 억제로 미백 효과 (정확), 레티놀 → 콜라겐 합성 촉진으로 주름 개선 (정확)",
        "2. 안전성 판단: 성분의 사용 제한/금지 여부가 정확한지 확인하세요. "
        "허위 안전성 주장이나 알려진 제한사항 누락, 등록되지 않은 성분을 안전하다고 판단하는 것은 감점하세요.",
        "3. 성분 궁합: 함께 사용 시 주의사항이 정확한지 확인하세요. "
        "예: 레티놀 + AHA → 이중 자극 가능성 안내 (정확). 근거 없이 단정하는 것은 감점하세요.",
        "4. 설명 품질: 전문 용어를 쉽게 풀어 설명했는지, 학술 근거 기반 답변에서 출처를 인용했는지 확인하세요.",
        "5. 일반 인사 등 화장품과 무관한 응답은 1.0점을 부여하세요.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4.1-nano",
    threshold=0.5,
)


class CosmeticAccuracy(BaseMetric):
    """DeepEval G-Eval 결과를 Opik BaseMetric 포맷으로 래핑합니다."""
    name = "cosmetic_accuracy"

    def score(self, output: str, **kwargs) -> ScoreResult:
        test_case = LLMTestCase(
            input=kwargs.get("input", ""),
            actual_output=output,
        )
        _deepeval_cosmetic_accuracy.measure(test_case)

        return ScoreResult(
            name=self.name,
            value=_deepeval_cosmetic_accuracy.score,
            reason=_deepeval_cosmetic_accuracy.reason,
        )


# ============================================================
# 레벨별 메트릭 구성
# ============================================================
def get_metrics(level: int):
    """레벨에 따라 사용할 메트릭 목록을 반환합니다.

    L1 (3개): 코드 기반 에이전트 동작 검증 — LLM 호출 없이 빠르게 실행
    L2 (6개): L1 + Opik 내장 LLM-as-a-Judge 2개 + DeepEval G-Eval 1개
    """
    # L1: 에이전트 동작 검증 (Heuristic)
    l1 = [
        ToolCorrectness(),       # DeepEval Tool Correctness
        KeywordCoverage(),       # RAG Faithfulness (code-level)
        SafetyCompliance(),      # 화장품 도메인 안전성
    ]

    # L2: Opik 내장 + DeepEval G-Eval (응답 품질)
    l2 = [
        AnswerRelevance(require_context=False),
        Hallucination(),
        CosmeticAccuracy(),      # DeepEval G-Eval 래핑
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

    tool_calls = []
    tool_results = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls.extend([tc["name"] for tc in msg.tool_calls])
        if hasattr(msg, "tool_call_id") and msg.content:
            tool_results.append(msg.content[:500])

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
