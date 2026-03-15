import asyncio
import contextlib
from datetime import datetime
import json
import uuid

from app.utils.logger import log_execution, custom_logger
from app.agents.cosmetic_agent import create_cosmetic_agent
from app.core.config import settings

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError


# 체크포인터 & 에이전트를 모듈 수준에서 한 번만 생성
# (checkpointer가 thread_id로 대화를 분리하므로 싱글턴으로 충분)
_checkpointer = MemorySaver()
_agent = create_cosmetic_agent(checkpointer=_checkpointer)


class AgentService:
    def __init__(self):
        self.agent = _agent
        self.progress_queue: asyncio.Queue = asyncio.Queue()

    @log_execution
    async def process_query(self, user_messages: str, thread_id: uuid.UUID):
        """사용자 메시지를 처리하고 스트리밍 응답을 생성합니다."""
        progress_task = None
        try:
            custom_logger.info(f"사용자 메시지: {user_messages}")

            agent_stream = self.agent.astream(
                {"messages": [HumanMessage(content=user_messages)]},
                config={
                    "configurable": {"thread_id": str(thread_id)},
                    "recursion_limit": settings.DEEPAGENT_RECURSION_LIMIT,
                },
                stream_mode="updates",
            )

            agent_iterator = agent_stream.__aiter__()
            agent_task = asyncio.create_task(agent_iterator.__anext__())
            progress_task = asyncio.create_task(self.progress_queue.get())

            while True:
                pending = {agent_task}
                if progress_task is not None:
                    pending.add(progress_task)

                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                # Progress 이벤트 처리
                if progress_task in done:
                    try:
                        progress_event = progress_task.result()
                        yield json.dumps(progress_event, ensure_ascii=False)
                        progress_task = asyncio.create_task(self.progress_queue.get())
                    except asyncio.CancelledError:
                        progress_task = None
                    except Exception as e:
                        custom_logger.error(f"Error in progress_task: {e}")
                        progress_task = None

                # 에이전트 스트림 처리
                if agent_task in done:
                    try:
                        chunk = agent_task.result()
                    except StopAsyncIteration:
                        agent_task = None
                        break
                    except Exception as e:
                        custom_logger.error(f"Error in agent_task: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())
                        agent_task = None
                        yield json.dumps(self._error_response(str(e)), ensure_ascii=False)
                        break

                    custom_logger.info(f"에이전트 청크: {chunk}")
                    try:
                        for step, event in chunk.items():
                            if not event or step not in ("model", "tools"):
                                continue
                            messages = event.get("messages", [])
                            if not messages:
                                continue
                            message = messages[0]

                            if step == "model":
                                tool_calls = message.tool_calls
                                if tool_calls:
                                    tool_names = [t["name"] for t in tool_calls]
                                    yield json.dumps({"step": "model", "tool_calls": tool_names}, ensure_ascii=False)
                                elif message.content:
                                    try:
                                        args = json.loads(message.content)
                                    except (json.JSONDecodeError, TypeError):
                                        args = {"content": message.content}
                                    metadata = args.get("metadata")
                                    yield json.dumps({
                                        "step": "done",
                                        "message_id": args.get("message_id", str(uuid.uuid4())),
                                        "role": "assistant",
                                        "content": args.get("content", message.content),
                                        "metadata": self._handle_metadata(metadata),
                                        "created_at": datetime.utcnow().isoformat(),
                                    }, ensure_ascii=False)

                            elif step == "tools":
                                try:
                                    content_val = json.loads(message.content)
                                except (json.JSONDecodeError, TypeError):
                                    content_val = message.content
                                yield json.dumps({
                                    "step": "tools",
                                    "name": message.name,
                                    "content": content_val,
                                }, ensure_ascii=False)

                    except Exception as e:
                        custom_logger.error(f"Error processing chunk: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())
                        yield json.dumps(self._error_response(str(e)), ensure_ascii=False)
                        break

                    agent_task = asyncio.create_task(agent_iterator.__anext__())

            # 남은 progress 이벤트 정리
            if progress_task is not None:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

            while not self.progress_queue.empty():
                try:
                    remaining = self.progress_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                yield json.dumps(remaining, ensure_ascii=False)

        except Exception as e:
            import traceback
            custom_logger.error(f"Error in process_query: {e}")
            custom_logger.error(traceback.format_exc())

            if progress_task is not None:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

            yield json.dumps(
                self._error_response(str(e) if not isinstance(e, GraphRecursionError) else None),
                ensure_ascii=False,
            )

    @staticmethod
    def _handle_metadata(metadata) -> dict:
        """metadata 객체를 dict로 변환"""
        result = {}
        if metadata:
            for k, v in metadata.items():
                result[k] = v
        return result

    @staticmethod
    def _error_response(error: str = None) -> dict:
        """에러 응답 포맷 생성"""
        if error:
            custom_logger.error(f"에러 상세: {error}")
        return {
            "step": "done",
            "message_id": str(uuid.uuid4()),
            "role": "assistant",
            "content": "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
            "metadata": {},
            "created_at": datetime.utcnow().isoformat(),
        }
