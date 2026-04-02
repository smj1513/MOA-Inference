from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from app.core.config import Settings
from app.core.time import current_date_in_timezone
from app.core.database import build_thread_id, validate_user_id
from app.core.logging import LangfuseTraceLogger
from app.domain.chatbot.agents.graph import ChatbotToolBundle, build_chatbot_graph
from app.domain.chatbot.repository.conversation_memory_store import (
    ConversationMemoryStore,
)
from app.domain.chatbot.service.conversation_summary_memory import (
    ConversationTurnSummarizer,
    LangChainConversationSummaryMemorySummarizer,
)


class ChatbotConfigurationError(RuntimeError):
    """Raised when chatbot runtime configuration is missing."""


SUPERVISOR_NODE_NAME = "supervisor"

class ChatbotService:
    def __init__(
        self,
        *,
        settings: Settings,
        graph_factory: Callable[[int], Any] | None = None,
        tracer: LangfuseTraceLogger | None = None,
        conversation_id_factory: Callable[[], str] | None = None,
        conversation_store: ConversationMemoryStore | None = None,
        conversation_summarizer: ConversationTurnSummarizer | None = None,
        closeables: Sequence[Any] | None = None,
    ) -> None:
        self._settings = settings
        self._tracer = tracer or LangfuseTraceLogger.from_settings(settings)
        self._conversation_id_factory = (
            conversation_id_factory or (lambda: str(uuid4()))
        )
        self._conversation_store = conversation_store or ConversationMemoryStore(
            ttl=timedelta(days=3),
        )
        self._conversation_summarizer = conversation_summarizer
        self._graph_factory = graph_factory or _build_default_graph_factory(
            settings=settings,
            tracer=self._tracer,
        )
        self._graph_cache: dict[int, Any] = {}
        self._closeables = list(closeables or [])

    async def reply(
        self,
        *,
        user_id: int,
        message: str,
        conversation_id: str | None,
    ) -> dict[str, object]:
        validated_user_id = validate_user_id(user_id)
        resolved_conversation_id = self._resolve_conversation_id(
            user_id=validated_user_id,
            conversation_id=conversation_id,
        )
        thread_id = build_thread_id(
            user_id=validated_user_id,
            conversation_id=resolved_conversation_id,
        )
        self._conversation_store.append_user_message(
            user_id=validated_user_id,
            conversation_id=resolved_conversation_id,
            content=message,
        )
        with self._tracer.start_as_current_span(
            name="chatbot.reply",
            input={
                "user_id": user_id,
                "message": message,
                "conversation_id": conversation_id,
            },
            metadata={
                "thread_id": thread_id,
            },
        ) as span:
            result = await self._get_graph(validated_user_id).ainvoke(
                {
                    "messages": self._build_graph_input_messages(
                        user_id=validated_user_id,
                        conversation_id=resolved_conversation_id,
                    ),
                    "user_id": validated_user_id,
                    "session_id": resolved_conversation_id,
                    "user_query": message,
                },
                config={"configurable": {"thread_id": thread_id}},
            )
            answer = self._extract_answer(result)
            warnings = list(result.get("warnings") or [])
            tool_summaries = list(result.get("tool_summaries") or [])
            if answer.strip():
                self._conversation_store.append_assistant_message(
                    user_id=validated_user_id,
                    conversation_id=resolved_conversation_id,
                    content=answer,
                )
                await self._update_conversation_summary(
                    user_id=validated_user_id,
                    conversation_id=resolved_conversation_id,
                )
            response = {
                "conversation_id": resolved_conversation_id,
                "answer": answer,
                "warnings": warnings,
                "tool_summaries": tool_summaries,
            }
            span.update(
                output=response,
                metadata={
                    "warning_count": len(warnings),
                    "tool_summary_count": len(tool_summaries),
                },
            )
            return response

    async def stream_reply(
        self,
        *,
        user_id: int,
        message: str,
        conversation_id: str | None,
    ) -> AsyncIterator[dict[str, object]]:
        validated_user_id = validate_user_id(user_id)
        resolved_conversation_id = self._resolve_conversation_id(
            user_id=validated_user_id,
            conversation_id=conversation_id,
        )
        thread_id = build_thread_id(
            user_id=validated_user_id,
            conversation_id=resolved_conversation_id,
        )
        self._conversation_store.append_user_message(
            user_id=validated_user_id,
            conversation_id=resolved_conversation_id,
            content=message,
        )
        graph = self._get_graph(validated_user_id)
        await _ensure_graph_ready(graph)
        async def _event_stream() -> AsyncIterator[dict[str, object]]:
            with self._tracer.start_as_current_span(
                name="chatbot.stream_reply",
                input={
                    "user_id": user_id,
                    "message": message,
                    "conversation_id": conversation_id,
                },
                metadata={
                    "thread_id": thread_id,
                },
            ) as span:
                yielded_text: list[str] = []
                yield {"event": "status", "data": {"message": "routing"}}

                async for event in graph.astream(
                    {
                        "messages": self._build_graph_input_messages(
                            user_id=validated_user_id,
                            conversation_id=resolved_conversation_id,
                        ),
                        "user_id": validated_user_id,
                        "session_id": resolved_conversation_id,
                        "user_query": message,
                    },
                    config={"configurable": {"thread_id": thread_id}},
                    stream_mode=["updates", "messages"],
                    subgraphs=True,
                ):
                    namespace, mode, chunk = self._unpack_stream_event(event)
                    if mode == "updates":
                        continue

                    text = self._extract_stream_text(
                        chunk,
                        namespace=namespace,
                    )
                    if not text:
                        continue

                    yielded_text.append(text)
                    yield {"event": "token", "data": {"text": text}}

                answer = "".join(yielded_text)
                if answer.strip():
                    self._conversation_store.append_assistant_message(
                        user_id=validated_user_id,
                        conversation_id=resolved_conversation_id,
                        content=answer,
                    )
                    await self._update_conversation_summary(
                        user_id=validated_user_id,
                        conversation_id=resolved_conversation_id,
                    )

                span.update(
                    output={
                        "conversation_id": resolved_conversation_id,
                        "answer": answer,
                    },
                    metadata={
                        "token_count": len(yielded_text),
                    },
                )
                yield {
                    "event": "done",
                    "data": {"conversation_id": resolved_conversation_id},
                }

        return _event_stream()

    async def aclose(self) -> None:
        for resource in self._closeables:
            await _close_resource(resource)

    def get_conversation_history(
        self,
        *,
        user_id: int,
        conversation_id: str,
    ) -> dict[str, object]:
        validated_user_id = validate_user_id(user_id)
        return self._conversation_store.get_conversation(
            user_id=validated_user_id,
            conversation_id=conversation_id,
        )

    def get_latest_conversation(
        self,
        *,
        user_id: int,
    ) -> dict[str, object]:
        validated_user_id = validate_user_id(user_id)
        return self._conversation_store.get_or_create_latest_conversation(
            user_id=validated_user_id,
            conversation_id_factory=self._conversation_id_factory,
        )

    def _get_graph(self, user_id: int) -> Any:
        if user_id not in self._graph_cache:
            self._graph_cache[user_id] = self._graph_factory(user_id)
        return self._graph_cache[user_id]

    def _resolve_conversation_id(
        self,
        *,
        user_id: int,
        conversation_id: str | None,
    ) -> str:
        normalized_conversation_id = (conversation_id or "").strip()
        if normalized_conversation_id:
            return normalized_conversation_id

        latest_conversation = self._conversation_store.get_or_create_latest_conversation(
            user_id=user_id,
            conversation_id_factory=self._conversation_id_factory,
        )
        return str(latest_conversation["conversation_id"])

    def _build_graph_input_messages(
        self,
        *,
        user_id: int,
        conversation_id: str,
    ) -> list[BaseMessage]:
        if self._conversation_summarizer is not None:
            prompt_context = self._conversation_store.get_prompt_context(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            summary_text = str(prompt_context.get("summary_text") or "").strip()
            messages = list(prompt_context.get("messages") or [])
            graph_messages: list[BaseMessage] = []
            if summary_text:
                graph_messages.append(
                    AIMessage(
                        content=f"Previous conversation summary:\n{summary_text}"
                    )
                )
            for message in messages:
                role = str(message.get("role") or "").strip()
                content = str(message.get("content") or "")
                if role == "user":
                    graph_messages.append(HumanMessage(content=content))
                    continue
                if role == "assistant":
                    graph_messages.append(AIMessage(content=content))
            return graph_messages

        conversation = self._conversation_store.get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        graph_messages: list[BaseMessage] = []
        for message in conversation["messages"]:
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "")
            if role == "user":
                graph_messages.append(HumanMessage(content=content))
                continue
            if role == "assistant":
                graph_messages.append(AIMessage(content=content))
        return graph_messages

    @staticmethod
    def _extract_answer(result: dict[str, Any]) -> str:
        final_response = result.get("final_response")
        if isinstance(final_response, str) and final_response.strip():
            return final_response

        answer = result.get("final_answer")
        if isinstance(answer, str) and answer.strip():
            return answer

        messages = result.get("messages") or []
        for message in reversed(messages):
            if ChatbotService._message_role(message) == "user":
                break
            if not ChatbotService._is_user_facing_message(message):
                continue
            text = ChatbotService._message_content(message)
            if text.strip():
                return text
        return ""

    async def _update_conversation_summary(
        self,
        *,
        user_id: int,
        conversation_id: str,
    ) -> None:
        if self._conversation_summarizer is None:
            return

        prompt_context = self._conversation_store.get_prompt_context(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        existing_summary = str(prompt_context.get("summary_text") or "")
        raw_messages = list(prompt_context.get("messages") or [])
        if len(raw_messages) < 2:
            return

        summary_messages: list[BaseMessage] = []
        for message in raw_messages:
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "")
            if role == "user":
                summary_messages.append(HumanMessage(content=content))
                continue
            if role == "assistant":
                summary_messages.append(AIMessage(content=content))

        if len(summary_messages) < 2:
            return

        updated_summary = await self._conversation_summarizer.summarize_turns(
            existing_summary=existing_summary,
            messages=summary_messages,
        )
        summarized_message_count = int(prompt_context.get("summarized_message_count") or 0)
        self._conversation_store.save_summary_state(
            user_id=user_id,
            conversation_id=conversation_id,
            summary_text=updated_summary,
            summarized_message_count=summarized_message_count + len(summary_messages),
        )

    @staticmethod
    def _unpack_stream_event(
        event: Any,
    ) -> tuple[tuple[str, ...], str, Any]:
        if (
            isinstance(event, tuple)
            and len(event) == 3
            and isinstance(event[0], tuple)
            and isinstance(event[1], str)
        ):
            return event[0], event[1], event[2]
        if (
            isinstance(event, tuple)
            and len(event) == 2
            and isinstance(event[0], str)
        ):
            return (), event[0], event[1]
        return (), "", event

    @staticmethod
    def _extract_stream_text(
        chunk: Any,
        *,
        namespace: tuple[str, ...] = (),
    ) -> str:
        message, metadata = ChatbotService._extract_stream_message_and_metadata(chunk)
        if message is None:
            return ""
        if not ChatbotService._is_user_facing_stream_chunk(
            message=message,
            metadata=metadata,
            namespace=namespace,
        ):
            return ""
        return ChatbotService._message_content(message)

    @staticmethod
    def _extract_stream_message_and_metadata(
        chunk: Any,
    ) -> tuple[Any | None, dict[str, Any]]:
        if isinstance(chunk, tuple):
            if not chunk:
                return None, {}
            message = chunk[0]
            metadata = chunk[1] if len(chunk) > 1 and isinstance(chunk[1], dict) else {}
            return message, metadata
        return chunk, {}

    @staticmethod
    def _is_user_facing_stream_chunk(
        *,
        message: Any,
        metadata: dict[str, Any],
        namespace: tuple[str, ...] = (),
    ) -> bool:
        source_name = ChatbotService._stream_source_name(
            namespace=namespace,
            metadata=metadata,
            message=message,
        )
        if source_name and source_name not in {
            SUPERVISOR_NODE_NAME,
            "agent",
            "model",
        }:
            return False
        return not ChatbotService._message_has_tool_calls(message)

    @staticmethod
    def _stream_source_name(
        *,
        namespace: tuple[str, ...],
        metadata: dict[str, Any],
        message: Any,
    ) -> str | None:
        namespace_name = ChatbotService._root_stream_namespace_name(namespace)
        if namespace_name is not None:
            return namespace_name

        checkpoint_name = ChatbotService._root_stream_namespace_name_from_checkpoint(
            metadata.get("checkpoint_ns") or metadata.get("langgraph_checkpoint_ns")
        )
        if checkpoint_name is not None:
            return checkpoint_name

        node_name = str(metadata.get("langgraph_node") or "").strip()
        if node_name and node_name not in {"agent", "model"}:
            return node_name

        return ChatbotService._message_name(message)

    @staticmethod
    def _root_stream_namespace_name(namespace: tuple[str, ...]) -> str | None:
        if not namespace:
            return None
        return ChatbotService._root_stream_namespace_name_from_checkpoint(namespace[0])

    @staticmethod
    def _root_stream_namespace_name_from_checkpoint(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        checkpoint = value.strip()
        if not checkpoint:
            return None
        root_segment = checkpoint.split("|", 1)[0].strip()
        if not root_segment:
            return None
        source_name = root_segment.split(":", 1)[0].strip()
        return source_name or None

    @staticmethod
    def _is_user_facing_message(message: Any) -> bool:
        if ChatbotService._message_has_tool_calls(message):
            return False
        if ChatbotService._message_role(message) != "assistant":
            return False
        return ChatbotService._message_name(message) in {
            None,
            SUPERVISOR_NODE_NAME,
        }

    @staticmethod
    def _message_content(message: Any) -> str:
        if isinstance(message, AIMessage):
            content = message.content
        elif isinstance(message, BaseMessage):
            content = message.content
        elif isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", message)

        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    @staticmethod
    def _message_name(message: Any) -> str | None:
        if isinstance(message, dict):
            name = message.get("name")
        else:
            name = getattr(message, "name", None)

        if isinstance(name, str):
            normalized_name = name.strip()
            if normalized_name:
                return normalized_name
        return None

    @staticmethod
    def _message_role(message: Any) -> str | None:
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, AIMessage):
            return "assistant"
        if isinstance(message, dict):
            role = str(message.get("role") or "").strip().casefold()
            return role if role in {"user", "assistant"} else None
        if isinstance(message, BaseMessage):
            message_type = str(getattr(message, "type", "") or "").strip().casefold()
            if message_type in {"human", "user"}:
                return "user"
            if message_type in {"ai", "assistant"}:
                return "assistant"
        role = str(getattr(message, "role", "") or "").strip().casefold()
        return role if role in {"user", "assistant"} else None

    @staticmethod
    def _message_has_tool_calls(message: Any) -> bool:
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls")
            tool_call_chunks = message.get("tool_call_chunks")
        else:
            tool_calls = getattr(message, "tool_calls", None)
            tool_call_chunks = getattr(message, "tool_call_chunks", None)
        return bool(tool_calls or tool_call_chunks)


def build_chatbot_service(
    *,
    settings: Settings,
    tracer: LangfuseTraceLogger | None = None,
) -> ChatbotService | None:
    if not settings.more_finance_mcp_url.strip():
        return None

    if not _has_chatbot_llm_config(settings):
        return None

    tracer = tracer or LangfuseTraceLogger.from_settings(settings)
    model_factory = _build_cached_model_factory(
        settings=settings,
        tracer=tracer,
    )
    return ChatbotService(
        settings=settings,
        graph_factory=_build_mcp_graph_factory(
            settings=settings,
            tracer=tracer,
            model_factory=model_factory,
        ),
        tracer=tracer,
        conversation_summarizer=LangChainConversationSummaryMemorySummarizer(
            llm=model_factory("chatbot_summary")
        ),
    )


def _build_default_graph_factory(
    *,
    settings: Settings,
    tracer: LangfuseTraceLogger,
) -> Callable[[int], Any]:
    if not settings.more_finance_mcp_url.strip():
        raise ChatbotConfigurationError(
            "MORE_FINANCE_MCP_URL, AI_API_KEY, and OPENAI_BASE_URL "
            "must be configured for the chatbot service"
        )
    return _build_mcp_graph_factory(
        settings=settings,
        tracer=tracer,
    )


def _build_mcp_graph_factory(
    *,
    settings: Settings,
    tracer: LangfuseTraceLogger,
    model_factory: Callable[[str], Any] | None = None,
) -> Callable[[int], Any]:
    resolved_model_factory = model_factory or _build_cached_model_factory(
        settings=settings,
        tracer=tracer,
    )

    def _factory(user_id: int) -> Any:
        del user_id
        today_factory = _build_chatbot_today_factory(settings=settings)
        return build_chatbot_graph(
            settings=settings,
            model_factory=resolved_model_factory,
            tracer=tracer,
            tools=ChatbotToolBundle.empty(),
            checkpointer=None,
            today=today_factory(),
        )

    return _factory


def _build_cached_model_factory(
    *,
    settings: Settings,
    tracer: LangfuseTraceLogger,
) -> Callable[[str], Any]:
    models: dict[str, Any] = {}

    def _factory(role: str) -> Any:
        if role not in models:
            from app.core.llm import build_chat_model

            models[role] = build_chat_model(settings=settings, tracer=tracer)
        return models[role]

    return _factory


def _build_chatbot_today_factory(
    *,
    settings: Settings,
    now_factory: Callable[[], datetime] | None = None,
) -> Callable[[], date]:
    return lambda: current_date_in_timezone(
        settings.chatbot_timezone,
        now_factory=now_factory or (lambda: datetime.now(UTC)),
    )
def _has_chatbot_llm_config(settings: Settings) -> bool:
    return all(
        [
            bool(settings.ai_api_key.strip()),
            bool(settings.openai_base_url.strip()),
        ]
    )


async def _ensure_graph_ready(graph: Any) -> None:
    ensure_ready = getattr(graph, "aensure_ready", None)
    if callable(ensure_ready):
        await ensure_ready()


async def _close_resource(resource: Any) -> None:
    if resource is None:
        return

    close = getattr(resource, "aclose", None)
    if callable(close):
        result = close()
        if hasattr(result, "__await__"):
            await result
        return

    close = getattr(resource, "close", None)
    if callable(close):
        result = close()
        if hasattr(result, "__await__"):
            await result
        return

    dispose = getattr(resource, "dispose", None)
    if callable(dispose):
        result = dispose()
        if hasattr(result, "__await__"):
            await result
        return
