from __future__ import annotations

from fastapi.testclient import TestClient

from app.domain.chatbot.errors import ChatbotUpstreamUnavailableError
from app.domain.chatbot.repository.conversation_memory_store import (
    ConversationExpiredError,
    ConversationNotFoundError,
)
from app.main import create_app


class _FakeChatbotService:
    def __init__(self) -> None:
        self.reply_calls: list[dict[str, object]] = []
        self.conversation_calls: list[dict[str, object]] = []
        self.latest_conversation_calls: list[dict[str, object]] = []
        self.history_error: str | None = None
        self.latest_error: str | None = None
        self.reply_error: str | None = None
        self.conversation_result = {
            "conversation_id": "conv-1",
            "created_at": "2026-03-26T09:00:00+00:00",
            "last_activity_at": "2026-03-26T09:01:00+00:00",
            "expires_at": "2026-03-29T09:01:00+00:00",
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                    "created_at": "2026-03-26T09:00:00+00:00",
                }
            ],
        }
        self.latest_conversation_result = {
            "conversation_id": "conv-1",
            "created_at": "2026-03-26T09:00:00+00:00",
            "last_activity_at": "2026-03-26T09:01:00+00:00",
            "expires_at": "2026-03-29T09:01:00+00:00",
        }

    async def reply(
        self,
        *,
        user_id: int,
        message: str,
        conversation_id: str | None,
    ) -> dict[str, object]:
        if self.reply_error == "upstream_unavailable":
            raise ChatbotUpstreamUnavailableError(
                "Chatbot coaching backend is temporarily unavailable. Upstream MCP responded with HTTP 401.",
                upstream_status_code=401,
            )
        self.reply_calls.append(
            {
                "user_id": user_id,
                "message": message,
                "conversation_id": conversation_id,
            }
        )
        return {
            "conversation_id": conversation_id or "generated-id",
            "answer": "Weekly budget is healthy.",
            "warnings": [],
            "tool_summaries": [],
        }

    def get_conversation_history(
        self,
        *,
        user_id: int,
        conversation_id: str,
    ) -> dict[str, object]:
        self.conversation_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
            }
        )
        if self.history_error == "not_found":
            raise ConversationNotFoundError("missing")
        if self.history_error == "expired":
            raise ConversationExpiredError("expired")
        return self.conversation_result

    def get_latest_conversation(
        self,
        *,
        user_id: int,
    ) -> dict[str, object]:
        self.latest_conversation_calls.append({"user_id": user_id})
        if self.latest_error == "expired":
            raise ConversationExpiredError("expired")
        return self.latest_conversation_result


class _FakeStreamingChatbotService(_FakeChatbotService):
    def __init__(self) -> None:
        super().__init__()
        self.stream_calls: list[dict[str, object]] = []
        self.stream_error: str | None = None

    async def stream_reply(
        self,
        *,
        user_id: int,
        message: str,
        conversation_id: str | None,
    ):
        if self.stream_error == "expired":
            raise ConversationExpiredError("expired")
        if self.stream_error == "upstream_unavailable":
            raise ChatbotUpstreamUnavailableError(
                "Chatbot coaching backend is temporarily unavailable. Upstream MCP responded with HTTP 401.",
                upstream_status_code=401,
            )

        self.stream_calls.append(
            {
                "user_id": user_id,
                "message": message,
                "conversation_id": conversation_id,
            }
        )

        async def _events():
            yield {"event": "status", "data": {"message": "routing"}}
            yield {"event": "token", "data": {"text": "Budget"}}
            yield {"event": "token", "data": {"text": " is on track."}}
            yield {
                "event": "done",
                "data": {"conversation_id": conversation_id or "generated-id"},
            }

        return _events()


def test_chatbot_message_endpoint_returns_json_response() -> None:
    service = _FakeChatbotService()

    def load_chatbot_service() -> _FakeChatbotService:
        return service

    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        load_chatbot_service=load_chatbot_service,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chatbot/messages",
            headers={"X-User-Id": "7"},
            json={"message": "Is my budget okay this week?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": "generated-id",
        "answer": "Weekly budget is healthy.",
        "warnings": [],
        "tool_summaries": [],
    }
    assert service.reply_calls == [
        {
            "user_id": 7,
            "message": "Is my budget okay this week?",
            "conversation_id": None,
        }
    ]


def test_chatbot_message_endpoint_returns_503_when_chatbot_backend_is_unavailable() -> None:
    service = _FakeChatbotService()
    service.reply_error = "upstream_unavailable"
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=service,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chatbot/messages",
            headers={"X-User-Id": "7"},
            json={"message": "Is my budget okay this week?"},
        )

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]


def test_chatbot_message_endpoint_requires_user_id_header() -> None:
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=_FakeChatbotService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chatbot/messages",
            json={"message": "Is my budget okay this week?"},
        )

    assert response.status_code == 422


def test_chatbot_message_endpoint_rejects_non_positive_user_id_header() -> None:
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=_FakeChatbotService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chatbot/messages",
            headers={"X-User-Id": "0"},
            json={"message": "Is my budget okay this week?"},
        )

    assert response.status_code == 422


def test_chatbot_message_endpoint_rejects_blank_message() -> None:
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=_FakeChatbotService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chatbot/messages",
            headers={"X-User-Id": "7"},
            json={"message": "   "},
        )

    assert response.status_code == 422


def test_chatbot_message_endpoint_preserves_provided_conversation_id() -> None:
    service = _FakeChatbotService()
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=service,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chatbot/messages",
            headers={"X-User-Id": "7"},
            json={
                "message": "Is my budget okay this week?",
                "conversation_id": "conv-1",
            },
        )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "conv-1"
    assert service.reply_calls == [
        {
            "user_id": 7,
            "message": "Is my budget okay this week?",
            "conversation_id": "conv-1",
        }
    ]


def test_chatbot_message_endpoint_accepts_status_based_tool_summaries() -> None:
    class _StatusToolSummaryService(_FakeChatbotService):
        async def reply(
            self,
            *,
            user_id: int,
            message: str,
            conversation_id: str | None,
        ) -> dict[str, object]:
            self.reply_calls.append(
                {
                    "user_id": user_id,
                    "message": message,
                    "conversation_id": conversation_id,
                }
            )
            return {
                "conversation_id": conversation_id or "generated-id",
                "answer": "이번 주 소비를 분석했어요.",
                "warnings": [],
                "tool_summaries": [
                    {
                        "tool_name": "get_spending_summary",
                        "status": "completed",
                    },
                    {
                        "tool_name": "get_spending_by_category",
                        "status": "completed",
                    },
                ],
            }

    service = _StatusToolSummaryService()
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=service,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chatbot/messages",
            headers={"X-User-Id": "7"},
            json={"message": "이번 주 소비 피드백 해줘"},
        )

    assert response.status_code == 200
    assert response.json()["tool_summaries"] == [
        {
            "tool_name": "get_spending_summary",
            "status": "completed",
        },
        {
            "tool_name": "get_spending_by_category",
            "status": "completed",
        },
    ]


def test_chatbot_message_endpoint_rejects_blank_conversation_id() -> None:
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=_FakeChatbotService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chatbot/messages",
            headers={"X-User-Id": "7"},
            json={"message": "Hello", "conversation_id": "   "},
        )

    assert response.status_code == 422


def test_chatbot_service_aclose_is_called_when_app_shuts_down() -> None:
    class _ClosableChatbotService(_FakeChatbotService):
        def __init__(self) -> None:
            super().__init__()
            self.aclose_calls = 0

        async def aclose(self) -> None:
            self.aclose_calls += 1

    service = _ClosableChatbotService()
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=service,
    )

    with TestClient(app):
        pass

    assert service.aclose_calls == 1


def test_chatbot_service_close_is_called_when_app_shuts_down() -> None:
    class _ClosableChatbotService(_FakeChatbotService):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    service = _ClosableChatbotService()
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=service,
    )

    with TestClient(app):
        pass

    assert service.close_calls == 1


def test_chatbot_stream_endpoint_returns_sse_response() -> None:
    service = _FakeStreamingChatbotService()

    def load_chatbot_service() -> _FakeStreamingChatbotService:
        return service

    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        load_chatbot_service=load_chatbot_service,
    )

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chatbot/messages/stream",
            headers={"X-User-Id": "7"},
            json={
                "message": "Is my budget okay this week?",
                "conversation_id": "conv-1",
            },
        ) as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: status" in body
    assert "event: token" in body
    assert "event: done" in body
    assert '"conversation_id": "conv-1"' in body
    assert service.stream_calls == [
        {
            "user_id": 7,
            "message": "Is my budget okay this week?",
            "conversation_id": "conv-1",
        }
    ]


def test_chatbot_stream_endpoint_rejects_non_positive_user_id_header() -> None:
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=_FakeStreamingChatbotService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chatbot/messages/stream",
            headers={"X-User-Id": "-1"},
            json={"message": "Is my budget okay this week?"},
        )

    assert response.status_code == 422


def test_chatbot_conversation_history_endpoint_returns_conversation() -> None:
    service = _FakeChatbotService()
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=service,
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/chatbot/conversations/conv-1",
            headers={"X-User-Id": "7"},
        )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "conv-1"
    assert service.conversation_calls == [
        {
            "user_id": 7,
            "conversation_id": "conv-1",
        }
    ]


def test_chatbot_latest_conversation_endpoint_returns_latest_conversation() -> None:
    service = _FakeChatbotService()
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=service,
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/chatbot/conversations/latest",
            headers={"X-User-Id": "7"},
        )

    assert response.status_code == 200
    assert response.json() == service.latest_conversation_result
    assert service.latest_conversation_calls == [{"user_id": 7}]


def test_chatbot_conversation_history_endpoint_returns_404() -> None:
    service = _FakeChatbotService()
    service.history_error = "not_found"
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=service,
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/chatbot/conversations/missing",
            headers={"X-User-Id": "7"},
        )

    assert response.status_code == 404


def test_chatbot_conversation_history_endpoint_returns_410() -> None:
    service = _FakeChatbotService()
    service.history_error = "expired"
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=service,
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/chatbot/conversations/expired",
            headers={"X-User-Id": "7"},
        )

    assert response.status_code == 410


def test_chatbot_latest_conversation_endpoint_creates_conversation_when_missing() -> None:
    service = _FakeChatbotService()
    service.latest_conversation_result = {
        "conversation_id": "generated-id",
        "created_at": "2026-03-26T09:00:00+00:00",
        "last_activity_at": "2026-03-26T09:00:00+00:00",
        "expires_at": "2026-03-29T09:00:00+00:00",
    }
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=service,
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/chatbot/conversations/latest",
            headers={"X-User-Id": "7"},
        )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "generated-id"


def test_chatbot_stream_endpoint_returns_410_for_expired_conversation() -> None:
    service = _FakeStreamingChatbotService()
    service.stream_error = "expired"
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=service,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chatbot/messages/stream",
            headers={"X-User-Id": "7"},
            json={"message": "hello", "conversation_id": "conv-1"},
        )

    assert response.status_code == 410


def test_chatbot_stream_endpoint_returns_503_when_chatbot_backend_is_unavailable() -> None:
    service = _FakeStreamingChatbotService()
    service.stream_error = "upstream_unavailable"
    app = create_app(
        classifier=object(),
        image_generation_service=object(),
        chatbot_service=service,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chatbot/messages/stream",
            headers={"X-User-Id": "7"},
            json={"message": "hello", "conversation_id": "conv-1"},
        )

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]
