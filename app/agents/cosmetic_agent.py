"""
화장품 성분 상담 ReAct 에이전트.
식약처 공공데이터 API를 통해 성분 정보와 사용제한 여부를 조회하여 답변합니다.
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from app.agents.middleware import handle_tool_errors
from app.agents.prompts import system_prompt
from app.agents.tools import (
    search_ingredient, search_restricted_ingredient, search_cosmetic_knowledge,
)
from app.core.config import settings
from app.models.chat import ChatResponse
from app.utils.logger import custom_logger

COSMETIC_TOOLS = [search_ingredient, search_restricted_ingredient, search_cosmetic_knowledge]


def create_cosmetic_agent(checkpointer: MemorySaver):
    """화장품 성분 상담 에이전트를 생성합니다.

    Args:
        checkpointer: 대화 기록 체크포인터 (MemorySaver)

    Returns:
        LangGraph 에이전트 인스턴스
    """
    custom_logger.info("화장품 성분 상담 에이전트 생성 중...")

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        streaming=True,
    )

    agent = create_agent(
        model=llm,
        tools=COSMETIC_TOOLS,
        system_prompt=system_prompt,
        response_format=ChatResponse,
        checkpointer=checkpointer,
        middleware=[handle_tool_errors],
    )

    custom_logger.info(
        f"에이전트 생성 완료 (도구: {[t.name for t in COSMETIC_TOOLS]})"
    )
    return agent
