from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.core.logging import LangfuseTraceLogger

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
except ImportError:  # pragma: no cover - dependency is expected but optional at runtime.
    LangfuseCallbackHandler = None


def build_chat_model(
    *,
    settings: Settings | None = None,
    tracer: LangfuseTraceLogger | None = None,
) -> ChatOpenAI:
    settings = settings or Settings()
    tracer = tracer or LangfuseTraceLogger.from_settings(settings)
    callbacks = _build_chat_model_callbacks(settings=settings, tracer=tracer)

    kwargs: dict[str, Any] = {
        "model": settings.chat_model,
        "api_key": settings.ai_api_key,
        "base_url": settings.openai_base_url,
        "streaming": True,
    }
    if callbacks:
        kwargs["callbacks"] = callbacks

    return ChatOpenAI(**kwargs)


def _build_chat_model_callbacks(
    *,
    settings: Settings,
    tracer: LangfuseTraceLogger | None,
) -> list[Any]:
    if LangfuseCallbackHandler is None:
        return []
    if not settings.langfuse_tracing_enabled:
        return []

    public_key = settings.langfuse_public_key or ""
    if not public_key.strip():
        return []

    if tracer is None or not tracer.enabled:
        return []

    return [LangfuseCallbackHandler(public_key=public_key)]
