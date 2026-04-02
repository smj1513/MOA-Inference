from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatbotMessageRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError("message must not be blank")
        return value

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value.strip() == "":
            raise ValueError("conversation_id must not be blank")
        return value


class ChatbotToolSummary(BaseModel):
    tool_name: str
    as_of: str | None = None
    status: str | None = None


class ChatbotConversationMessage(BaseModel):
    role: str
    content: str
    created_at: str


class ChatbotConversationResponse(BaseModel):
    conversation_id: str
    created_at: str
    last_activity_at: str
    expires_at: str
    messages: list[ChatbotConversationMessage]


class ChatbotConversationSummaryResponse(BaseModel):
    conversation_id: str
    created_at: str
    last_activity_at: str
    expires_at: str


class ChatbotMessageResponse(BaseModel):
    conversation_id: str
    answer: str
    warnings: list[str]
    tool_summaries: list[ChatbotToolSummary]
