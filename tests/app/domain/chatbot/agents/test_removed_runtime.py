from __future__ import annotations

from datetime import date
from typing import Any, Sequence

import anyio
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from app.core.config import Settings
from app.domain.chatbot.agents.graph import ChatbotToolBundle, build_chatbot_graph


class _BindableFakeChatModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Sequence[BaseTool | dict | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "_BindableFakeChatModel":
        return self


def test_removed_runtime_placeholder_is_replaced_by_a_create_agent_runtime() -> None:
    model = _BindableFakeChatModel(
        responses=[AIMessage(content="Try to pause impulse buys this week.")],
    )
    graph = build_chatbot_graph(
        settings=Settings(
            _env_file=None,
            AI_API_KEY="relay-key",
            OPENAI_BASE_URL="https://example.com/v1",
            CHAT_MODEL="gpt-4.1-mini",
        ),
        model_factory=lambda _role: model,
        tools=ChatbotToolBundle.empty(),
        today=date(2026, 3, 27),
    )

    result = anyio.run(
        lambda: graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="Coach my spending habits."),
                ],
                "user_id": 7,
                "session_id": "conv-1",
                "user_query": "Coach my spending habits.",
            },
            config={"configurable": {"thread_id": "moa:7:conv-1"}},
        )
    )

    assert result["final_response"].strip()
    assert result["final_answer"].strip()
    assert "removed" not in result["final_response"].casefold()
    assert "placeholder" not in result["final_response"].casefold()
