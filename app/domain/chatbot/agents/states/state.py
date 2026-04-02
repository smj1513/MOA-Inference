from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import BaseMessage


class ToolSummary(TypedDict):
    tool_name: str
    status: str


class ChatbotGraphState(TypedDict, total=False):
    messages: list[BaseMessage]
    user_id: int
    session_id: str
    user_query: str
    warnings: list[str]
    tool_summaries: list[ToolSummary]
    final_response: str | None
    final_answer: str | None
