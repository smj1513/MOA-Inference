from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chatbot_service
from app.domain.chatbot.api.dto.detail_dto import (
    ChatbotConversationSummaryResponse,
    ChatbotConversationResponse,
    ChatbotMessageRequest,
    ChatbotMessageResponse,
)
from app.domain.chatbot.errors import ChatbotUpstreamUnavailableError
from app.domain.chatbot.repository.conversation_memory_store import (
    ConversationExpiredError,
    ConversationNotFoundError,
)

router = APIRouter(prefix="/v1/chatbot", tags=["chatbot"])
TrustedUserId = Annotated[int, Header(alias="X-User-Id", ge=1)]


@router.post(
    "/messages",
    response_model=ChatbotMessageResponse,
    response_model_exclude_none=True,
)
async def create_chatbot_message(
    request: ChatbotMessageRequest,
    user_id: TrustedUserId,
    service=Depends(get_chatbot_service),
) -> dict[str, object]:
    try:
        return await service.reply(
            user_id=user_id,
            message=request.message,
            conversation_id=request.conversation_id,
        )
    except ChatbotUpstreamUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/messages/stream")
async def stream_chatbot_message(
    request: ChatbotMessageRequest,
    user_id: TrustedUserId,
    service=Depends(get_chatbot_service),
) -> StreamingResponse:
    try:
        stream = await service.stream_reply(
            user_id=user_id,
            message=request.message,
            conversation_id=request.conversation_id,
        )
    except ConversationExpiredError as error:
        raise HTTPException(status_code=410, detail=str(error)) from error
    except ChatbotUpstreamUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    async def event_stream() -> AsyncIterator[str]:
        async for event in stream:
            event_name = str(event["event"])
            payload = json.dumps(event["data"], ensure_ascii=False)
            yield f"event: {event_name}\ndata: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get(
    "/conversations/latest",
    response_model=ChatbotConversationSummaryResponse,
)
async def get_latest_chatbot_conversation(
    user_id: TrustedUserId,
    service=Depends(get_chatbot_service),
) -> dict[str, object]:
    try:
        return service.get_latest_conversation(user_id=user_id)
    except ConversationExpiredError as error:
        raise HTTPException(status_code=410, detail=str(error)) from error


@router.get(
    "/conversations/{conversation_id}",
    response_model=ChatbotConversationResponse,
)
async def get_chatbot_conversation(
    conversation_id: str,
    user_id: TrustedUserId,
    service=Depends(get_chatbot_service),
) -> dict[str, object]:
    try:
        return service.get_conversation_history(
            user_id=user_id,
            conversation_id=conversation_id,
        )
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ConversationExpiredError as error:
        raise HTTPException(status_code=410, detail=str(error)) from error
