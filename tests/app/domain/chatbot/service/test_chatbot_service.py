from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import anyio
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.core.config import Settings
from app.domain.chatbot.errors import ChatbotUpstreamUnavailableError
from app.domain.chatbot.repository.conversation_memory_store import (
    ConversationExpiredError,
    ConversationMemoryStore,
    ConversationNotFoundError,
)
from app.domain.chatbot.service.chatbot_service import (
    ChatbotService,
    build_chatbot_service,
)


class _FakeSpan:
    def __init__(self, record: dict[str, object]) -> None:
        self.record = record

    def __enter__(self) -> "_FakeSpan":
        self.record["entered"] = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.record["exited"] = True
        return False

    def update(self, **kwargs: object) -> None:
        self.record.setdefault("updates", []).append(kwargs)


class _FakeTracer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @property
    def enabled(self) -> bool:
        return True

    def start_as_current_span(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeSpan(kwargs)


class _FakeGraph:
    def __init__(
        self,
        *,
        reply_messages: list[object] | None = None,
        reply_final_answer: str | None = None,
        reply_final_response: str | None = None,
        stream_chunks: list[tuple[str, object]] | None = None,
    ) -> None:
        self.reply_messages = reply_messages or [
            HumanMessage(content="How is my budget?"),
            AIMessage(content="Weekly budget is healthy.", name="supervisor"),
        ]
        self.reply_final_answer = reply_final_answer
        self.reply_final_response = reply_final_response
        self.stream_chunks = stream_chunks or [
            (
                "messages",
                (
                    AIMessage(content="Internal specialist note", name="spending_specialist"),
                    {"langgraph_node": "spending_specialist"},
                ),
            ),
            (
                "messages",
                (
                    AIMessage(content="Budget", name="supervisor"),
                    {"langgraph_node": "supervisor"},
                ),
            ),
            (
                "messages",
                (
                    AIMessage(content=" is healthy.", name="supervisor"),
                    {"langgraph_node": "supervisor"},
                ),
            ),
        ]
        self.reply_calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []

    async def ainvoke(
        self,
        input: dict[str, object],
        config: dict[str, object] | None = None,
    ):
        self.reply_calls.append({"input": input, "config": config})
        return {
            "messages": self.reply_messages,
            "warnings": ["watch budget"],
            "tool_summaries": [
                {
                    "tool_name": "sql_team",
                    "status": "completed",
                }
            ],
            "final_answer": self.reply_final_answer,
            "final_response": self.reply_final_response,
        }

    async def astream(
        self,
        input: dict[str, object],
        config: dict[str, object] | None = None,
        stream_mode: object | None = None,
        subgraphs: bool = False,
    ):
        self.stream_calls.append(
            {
                "input": input,
                "config": config,
                "stream_mode": stream_mode,
                "subgraphs": subgraphs,
            }
        )
        for chunk in self.stream_chunks:
            yield chunk


class _FakeGraphFactory:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.graphs: dict[int, _FakeGraph] = {}

    def __call__(self, user_id: int) -> _FakeGraph:
        self.calls.append(user_id)
        graph = self.graphs.get(user_id)
        if graph is None:
            graph = _FakeGraph()
            self.graphs[user_id] = graph
        return graph


class _FakeConversationSummarizer:
    def __init__(self, result: str = "updated summary") -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def summarize_turns(
        self,
        *,
        existing_summary: str,
        messages: list[object],
    ) -> str:
        self.calls.append(
            {
                "existing_summary": existing_summary,
                "messages": messages,
            }
        )
        return self.result


def _build_settings() -> Settings:
    return Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        OPENAI_BASE_URL="https://example.com/v1",
        CHAT_MODEL="gpt-4.1-mini",
    )


def test_build_chatbot_service_uses_mcp_url_to_activate_the_chatbot_runtime() -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        OPENAI_BASE_URL="https://example.com/v1",
        CHAT_MODEL="gpt-4.1-mini",
        MORE_FINANCE_MCP_URL="http://backend:8080/api/mcp",
    )

    service = build_chatbot_service(settings=settings, tracer=_FakeTracer())

    assert service is not None


def test_build_chatbot_service_passes_tracer_into_graph_runtime(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        OPENAI_BASE_URL="https://example.com/v1",
        CHAT_MODEL="gpt-4.1-mini",
        MORE_FINANCE_MCP_URL="http://backend:8080/api/mcp",
    )
    tracer = _FakeTracer()
    captured: dict[str, object] = {}

    class _StubGraph:
        pass

    def fake_build_chatbot_graph(**kwargs):
        captured.update(kwargs)
        return _StubGraph()

    monkeypatch.setattr(
        "app.domain.chatbot.service.chatbot_service.build_chatbot_graph",
        fake_build_chatbot_graph,
    )

    service = build_chatbot_service(settings=settings, tracer=tracer)

    assert service is not None
    graph = service._get_graph(7)
    assert isinstance(graph, _StubGraph)
    assert captured["tracer"] is tracer


def test_build_chatbot_service_returns_none_without_mcp_url_even_with_llm_settings() -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        OPENAI_BASE_URL="https://example.com/v1",
        CHAT_MODEL="gpt-4.1-mini",
    )

    assert build_chatbot_service(settings=settings, tracer=_FakeTracer()) is None


def test_build_chatbot_service_returns_none_without_mcp_url_even_with_database_settings() -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        OPENAI_BASE_URL="https://example.com/v1",
        CHAT_MODEL="gpt-4.1-mini",
        DB_URL="postgresql://db:5432/moa",
        DB_USERNAME="readonly",
        DB_PASSWORD="secret",
    )

    assert build_chatbot_service(settings=settings, tracer=_FakeTracer()) is None


def test_stream_reply_raises_runtime_unavailable_before_opening_stream() -> None:
    class _UnavailableGraph:
        async def aensure_ready(self) -> None:
            raise ChatbotUpstreamUnavailableError(
                "Chatbot coaching backend is temporarily unavailable. Upstream MCP responded with HTTP 401.",
                upstream_status_code=401,
            )

    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=lambda _user_id: _UnavailableGraph(),
        tracer=_FakeTracer(),
    )

    async def run() -> None:
        await service.stream_reply(
            user_id=7,
            message="How is my budget?",
            conversation_id="conv-1",
        )

    with pytest.raises(
        ChatbotUpstreamUnavailableError,
        match="temporarily unavailable.*401",
    ):
        anyio.run(run)


def test_reply_uses_user_scoped_thread_id_and_caches_graph_per_user() -> None:
    factory = _FakeGraphFactory()
    tracer = _FakeTracer()
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=factory,
        tracer=tracer,
    )

    async def run() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        first = await service.reply(
            user_id=7,
            message="How is my budget?",
            conversation_id="conv-1",
        )
        second = await service.reply(
            user_id=7,
            message="How is my budget now?",
            conversation_id="conv-2",
        )
        third = await service.reply(
            user_id=8,
            message="How is my budget?",
            conversation_id="conv-3",
        )
        return first, second, third

    first, second, third = anyio.run(run)

    assert factory.calls == [7, 8]
    assert factory.graphs[7].reply_calls[0]["config"] == {
        "configurable": {"thread_id": "moa:7:conv-1"}
    }
    assert factory.graphs[7].reply_calls[1]["config"] == {
        "configurable": {"thread_id": "moa:7:conv-2"}
    }
    assert factory.graphs[8].reply_calls[0]["config"] == {
        "configurable": {"thread_id": "moa:8:conv-3"}
    }
    assert first["conversation_id"] == "conv-1"
    assert second["conversation_id"] == "conv-2"
    assert third["conversation_id"] == "conv-3"
    assert first["answer"] == "Weekly budget is healthy."
    assert first["warnings"] == ["watch budget"]
    assert first["tool_summaries"] == [
        {
            "tool_name": "sql_team",
            "status": "completed",
        }
    ]
    assert tracer.calls[0]["name"] == "chatbot.reply"

def test_reply_generates_conversation_id_when_missing() -> None:
    factory = _FakeGraphFactory()
    tracer = _FakeTracer()
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=factory,
        tracer=tracer,
    )

    async def run() -> dict[str, object]:
        return await service.reply(
            user_id=7,
            message="How is my budget?",
            conversation_id=None,
        )

    result = anyio.run(run)

    generated_conversation_id = result["conversation_id"]
    assert isinstance(generated_conversation_id, str)
    UUID(generated_conversation_id)
    assert factory.graphs[7].reply_calls[0]["config"] == {
        "configurable": {
            "thread_id": f"moa:7:{generated_conversation_id}",
        }
    }


def test_reply_without_conversation_id_reuses_latest_active_conversation() -> None:
    current = {"now": datetime(2026, 3, 26, 9, 0, tzinfo=UTC)}
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: current["now"],
    )
    store.append_user_message(user_id=7, conversation_id="conv-1", content="older")
    current["now"] = datetime(2026, 3, 26, 10, 0, tzinfo=UTC)

    factory = _FakeGraphFactory()
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=factory,
        tracer=_FakeTracer(),
        conversation_store=store,
        conversation_id_factory=lambda: "generated-conv",
    )

    async def run() -> dict[str, object]:
        return await service.reply(
            user_id=7,
            message="How is my budget?",
            conversation_id=None,
        )

    result = anyio.run(run)

    assert result["conversation_id"] == "conv-1"
    assert factory.graphs[7].reply_calls[0]["config"] == {
        "configurable": {"thread_id": "moa:7:conv-1"}
    }


def test_get_latest_conversation_creates_conversation_when_missing() -> None:
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=_FakeGraphFactory(),
        tracer=_FakeTracer(),
        conversation_store=ConversationMemoryStore(
            ttl=timedelta(days=3),
            now_factory=lambda: datetime(2026, 3, 26, 9, 0, tzinfo=UTC),
        ),
        conversation_id_factory=lambda: "generated-conv",
    )

    latest = service.get_latest_conversation(user_id=7)

    assert latest == {
        "conversation_id": "generated-conv",
        "created_at": "2026-03-26T09:00:00+00:00",
        "last_activity_at": "2026-03-26T09:00:00+00:00",
        "expires_at": "2026-03-29T09:00:00+00:00",
    }


def test_reply_ignores_non_supervisor_messages_when_final_answer_missing() -> None:
    graph = _FakeGraph(
        reply_messages=[
            HumanMessage(content="How is my budget?"),
            AIMessage(content="Weekly budget is healthy.", name="supervisor"),
            AIMessage(content="Internal specialist note", name="spending_specialist"),
        ]
    )
    tracer = _FakeTracer()
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=lambda _user_id: graph,
        tracer=tracer,
    )

    async def run() -> dict[str, object]:
        return await service.reply(
            user_id=7,
            message="How is my budget?",
            conversation_id="conv-1",
        )

    result = anyio.run(run)

    assert result["answer"] == "Weekly budget is healthy."


def test_reply_does_not_echo_user_message_or_summary_when_final_answer_is_blank() -> None:
    graph = _FakeGraph(
        reply_messages=[
            AIMessage(content="Previous conversation summary:\nprior coaching recap"),
            HumanMessage(content="How is my budget?"),
        ],
        reply_final_answer="",
        reply_final_response="",
    )
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=lambda _user_id: graph,
        tracer=_FakeTracer(),
    )

    async def run() -> dict[str, object]:
        return await service.reply(
            user_id=7,
            message="How is my budget?",
            conversation_id="conv-1",
        )

    result = anyio.run(run)

    assert result["answer"] == ""


def test_reply_builds_graph_input_from_sanitized_conversation_history() -> None:
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: datetime(2026, 3, 26, 9, 0, tzinfo=UTC),
    )
    store.append_user_message(user_id=7, conversation_id="conv-1", content="older user")
    store.append_assistant_message(
        user_id=7,
        conversation_id="conv-1",
        content="older assistant",
    )
    graph = _FakeGraph(reply_final_answer="latest answer")
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=lambda _user_id: graph,
        tracer=_FakeTracer(),
        conversation_store=store,
    )

    async def run() -> None:
        await service.reply(
            user_id=7,
            message="latest user",
            conversation_id="conv-1",
        )

    anyio.run(run)

    input_messages = graph.reply_calls[0]["input"]["messages"]
    assert [message.content for message in input_messages] == [
        "older user",
        "older assistant",
        "latest user",
    ]
    assert [type(message).__name__ for message in input_messages] == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
    ]
    assert graph.reply_calls[0]["input"]["user_id"] == 7
    assert graph.reply_calls[0]["input"]["session_id"] == "conv-1"
    assert graph.reply_calls[0]["input"]["user_query"] == "latest user"


def test_reply_prefers_final_response_when_present() -> None:
    graph = _FakeGraph(
        reply_messages=[
            HumanMessage(content="How is my budget?"),
            AIMessage(content="legacy final answer", name="supervisor"),
        ],
        reply_final_answer=None,
        reply_final_response="# 이번 소비 요약\n- 새 최종 응답",
    )
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=lambda _user_id: graph,
        tracer=_FakeTracer(),
    )

    async def run() -> dict[str, object]:
        return await service.reply(
            user_id=7,
            message="How is my budget?",
            conversation_id="conv-1",
        )

    result = anyio.run(run)

    assert result["answer"] == "# 이번 소비 요약\n- 새 최종 응답"


def test_reply_uses_summary_plus_unsummarized_tail_when_summarizer_is_configured() -> None:
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: datetime(2026, 3, 26, 9, 0, tzinfo=UTC),
    )
    store.append_user_message(user_id=7, conversation_id="conv-1", content="older user")
    store.append_assistant_message(
        user_id=7,
        conversation_id="conv-1",
        content="older assistant",
    )
    store.save_summary_state(
        user_id=7,
        conversation_id="conv-1",
        summary_text="summary of prior turns",
        summarized_message_count=2,
    )
    graph = _FakeGraph(reply_final_answer="latest answer")
    summarizer = _FakeConversationSummarizer(result="summary after latest turn")
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=lambda _user_id: graph,
        tracer=_FakeTracer(),
        conversation_store=store,
        conversation_summarizer=summarizer,
    )

    async def run() -> None:
        await service.reply(
            user_id=7,
            message="latest user",
            conversation_id="conv-1",
        )

    anyio.run(run)

    input_messages = graph.reply_calls[0]["input"]["messages"]
    assert [message.content for message in input_messages] == [
        "Previous conversation summary:\nsummary of prior turns",
        "latest user",
    ]
    assert [type(message).__name__ for message in input_messages] == [
        "AIMessage",
        "HumanMessage",
    ]
    assert summarizer.calls[0]["existing_summary"] == "summary of prior turns"
    assert [message.content for message in summarizer.calls[0]["messages"]] == [
        "latest user",
        "latest answer",
    ]
    prompt_context = store.get_prompt_context(user_id=7, conversation_id="conv-1")
    assert prompt_context["summary_text"] == "summary after latest turn"
    assert prompt_context["summarized_message_count"] == 4
    assert prompt_context["messages"] == []


def test_stream_reply_updates_summary_after_final_assistant_output() -> None:
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: datetime(2026, 3, 26, 9, 0, tzinfo=UTC),
    )
    summarizer = _FakeConversationSummarizer(result="summary after stream turn")
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=lambda _user_id: _FakeGraph(),
        tracer=_FakeTracer(),
        conversation_store=store,
        conversation_summarizer=summarizer,
    )

    async def run() -> None:
        stream = await service.stream_reply(
            user_id=7,
            message="hello",
            conversation_id="conv-1",
        )
        async for _ in stream:
            pass

    anyio.run(run)

    assert [message.content for message in summarizer.calls[0]["messages"]] == [
        "hello",
        "Budget is healthy.",
    ]
    prompt_context = store.get_prompt_context(user_id=7, conversation_id="conv-1")
    assert prompt_context["summary_text"] == "summary after stream turn"
    assert prompt_context["summarized_message_count"] == 2
    assert prompt_context["messages"] == []


def test_reply_rejects_non_positive_user_id() -> None:
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=_FakeGraphFactory(),
        tracer=_FakeTracer(),
    )

    async def run() -> None:
        await service.reply(
            user_id=0,
            message="How is my budget?",
            conversation_id="conv-1",
        )

    with pytest.raises(ValueError, match="positive integer"):
        anyio.run(run)


def test_stream_reply_emits_only_supervisor_tokens_and_done() -> None:
    factory = _FakeGraphFactory()
    tracer = _FakeTracer()
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=factory,
        tracer=tracer,
    )

    async def run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        stream = await service.stream_reply(
            user_id=7,
            message="How is my budget?",
            conversation_id="conv-1",
        )
        async for event in stream:
            events.append(event)
        return events

    events = anyio.run(run)

    assert events[0] == {"event": "status", "data": {"message": "routing"}}
    assert events[1] == {"event": "token", "data": {"text": "Budget"}}
    assert events[2] == {"event": "token", "data": {"text": " is healthy."}}
    assert events[-1] == {
        "event": "done",
        "data": {"conversation_id": "conv-1"},
    }
    assert len(events) == 4
    assert factory.graphs[7].stream_calls[0]["config"] == {
        "configurable": {"thread_id": "moa:7:conv-1"}
    }
    assert factory.graphs[7].stream_calls[0]["subgraphs"] is True
    assert tracer.calls[-1]["name"] == "chatbot.stream_reply"


def test_stream_reply_filters_subgraph_chunks_to_final_supervisor_tokens() -> None:
    graph = _FakeGraph(
        stream_chunks=[
            (
                ("supervisor:step-1",),
                "messages",
                (
                    AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": "transfer_to_budget_specialist",
                                "args": "",
                                "id": "call-1",
                                "index": 0,
                                "type": "tool_call_chunk",
                            }
                        ],
                    ),
                    {
                        "checkpoint_ns": "supervisor:step-1",
                        "langgraph_node": "agent",
                    },
                ),
            ),
            (
                ("budget_specialist:step-1",),
                "messages",
                (
                    AIMessageChunk(content="Internal"),
                    {
                        "checkpoint_ns": "budget_specialist:step-1",
                        "langgraph_node": "model",
                    },
                ),
            ),
            (
                ("supervisor:step-2",),
                "messages",
                (
                    AIMessageChunk(content="Hel"),
                    {
                        "checkpoint_ns": "supervisor:step-2",
                        "langgraph_node": "agent",
                    },
                ),
            ),
            (
                ("supervisor:step-2",),
                "messages",
                (
                    AIMessageChunk(content="lo"),
                    {
                        "checkpoint_ns": "supervisor:step-2",
                        "langgraph_node": "agent",
                    },
                ),
            ),
            (
                ("supervisor:step-2",),
                "messages",
                (
                    AIMessageChunk(content="", chunk_position="last"),
                    {
                        "checkpoint_ns": "supervisor:step-2",
                        "langgraph_node": "agent",
                    },
                ),
            ),
        ]
    )
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=lambda _user_id: graph,
        tracer=_FakeTracer(),
    )

    async def run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        stream = await service.stream_reply(
            user_id=7,
            message="Reply with exactly Hello.",
            conversation_id="conv-1",
        )
        async for event in stream:
            events.append(event)
        return events

    events = anyio.run(run)

    assert events == [
        {"event": "status", "data": {"message": "routing"}},
        {"event": "token", "data": {"text": "Hel"}},
        {"event": "token", "data": {"text": "lo"}},
        {"event": "done", "data": {"conversation_id": "conv-1"}},
    ]
    assert graph.stream_calls[0]["subgraphs"] is True


def test_stream_reply_without_conversation_id_reuses_latest_active_conversation() -> None:
    current = {"now": datetime(2026, 3, 26, 9, 0, tzinfo=UTC)}
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: current["now"],
    )
    store.append_user_message(user_id=7, conversation_id="conv-1", content="older")
    current["now"] = datetime(2026, 3, 26, 10, 0, tzinfo=UTC)

    factory = _FakeGraphFactory()
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=factory,
        tracer=_FakeTracer(),
        conversation_store=store,
        conversation_id_factory=lambda: "generated-conv",
    )

    async def run() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        stream = await service.stream_reply(
            user_id=7,
            message="hello",
            conversation_id=None,
        )
        async for event in stream:
            events.append(event)
        return events

    events = anyio.run(run)

    assert events[-1] == {
        "event": "done",
        "data": {"conversation_id": "conv-1"},
    }
    assert factory.graphs[7].stream_calls[0]["config"] == {
        "configurable": {"thread_id": "moa:7:conv-1"}
    }


def test_reply_records_user_and_assistant_messages() -> None:
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: datetime(2026, 3, 26, 9, 0, tzinfo=UTC),
    )
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=lambda _user_id: _FakeGraph(reply_final_answer="answer"),
        tracer=_FakeTracer(),
        conversation_store=store,
    )

    async def run() -> None:
        await service.reply(
            user_id=7,
            message="hello",
            conversation_id="conv-1",
        )

    anyio.run(run)
    conversation = service.get_conversation_history(user_id=7, conversation_id="conv-1")

    assert [message["role"] for message in conversation["messages"]] == [
        "user",
        "assistant",
    ]
    assert [message["content"] for message in conversation["messages"]] == [
        "hello",
        "answer",
    ]


def test_stream_reply_records_final_assistant_message_only() -> None:
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: datetime(2026, 3, 26, 9, 0, tzinfo=UTC),
    )
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=lambda _user_id: _FakeGraph(),
        tracer=_FakeTracer(),
        conversation_store=store,
    )

    async def run() -> None:
        stream = await service.stream_reply(
            user_id=7,
            message="hello",
            conversation_id="conv-1",
        )
        async for _ in stream:
            pass

    anyio.run(run)
    conversation = service.get_conversation_history(user_id=7, conversation_id="conv-1")

    assert [message["content"] for message in conversation["messages"]] == [
        "hello",
        "Budget is healthy.",
    ]


def test_reply_rejects_expired_conversation_id() -> None:
    current = {"now": datetime(2026, 3, 26, 9, 0, tzinfo=UTC)}
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: current["now"],
    )
    store.append_user_message(user_id=7, conversation_id="conv-1", content="old")
    current["now"] = datetime(2026, 3, 29, 9, 0, 1, tzinfo=UTC)

    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=lambda _user_id: _FakeGraph(),
        tracer=_FakeTracer(),
        conversation_store=store,
    )

    async def run() -> None:
        await service.reply(user_id=7, message="new", conversation_id="conv-1")

    with pytest.raises(ConversationExpiredError):
        anyio.run(run)


def test_get_conversation_history_raises_for_unknown_id() -> None:
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=_FakeGraphFactory(),
        tracer=_FakeTracer(),
        conversation_store=ConversationMemoryStore(
            ttl=timedelta(days=3),
            now_factory=lambda: datetime(2026, 3, 26, 9, 0, tzinfo=UTC),
        ),
    )

    with pytest.raises(ConversationNotFoundError):
        service.get_conversation_history(user_id=7, conversation_id="missing")


def test_get_latest_conversation_returns_latest_metadata() -> None:
    current = {"now": datetime(2026, 3, 26, 9, 0, tzinfo=UTC)}
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: current["now"],
    )
    service = ChatbotService(
        settings=_build_settings(),
        graph_factory=_FakeGraphFactory(),
        tracer=_FakeTracer(),
        conversation_store=store,
    )

    store.append_user_message(user_id=7, conversation_id="conv-1", content="first")
    current["now"] = datetime(2026, 3, 26, 10, 0, tzinfo=UTC)
    store.append_user_message(user_id=7, conversation_id="conv-2", content="second")

    latest = service.get_latest_conversation(user_id=7)

    assert latest == {
        "conversation_id": "conv-2",
        "created_at": "2026-03-26T10:00:00+00:00",
        "last_activity_at": "2026-03-26T10:00:00+00:00",
        "expires_at": "2026-03-29T10:00:00+00:00",
    }
