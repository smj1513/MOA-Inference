from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.core.llm import build_chat_model


class _FakeChatOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeEnabledTracer:
    enabled = True


class _FakeDisabledTracer:
    enabled = False


def test_build_chat_model_attaches_langfuse_callbacks_when_configured(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}
    callback_calls: list[dict[str, Any]] = []

    def fake_chat_openai(**kwargs: Any) -> _FakeChatOpenAI:
        captured.update(kwargs)
        return _FakeChatOpenAI(**kwargs)

    def fake_callback_handler(*, public_key: str) -> object:
        callback_calls.append({"public_key": public_key})
        return object()

    monkeypatch.setattr(
        "app.core.llm.ChatOpenAI",
        fake_chat_openai,
    )
    monkeypatch.setattr(
        "app.core.llm.LangfuseTraceLogger.from_settings",
        lambda settings: _FakeEnabledTracer(),
    )
    monkeypatch.setattr(
        "app.core.llm.LangfuseCallbackHandler",
        fake_callback_handler,
    )

    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        OPENAI_BASE_URL="https://example.com/v1",
        CHAT_MODEL="gpt-4.1-mini",
        LANGFUSE_PUBLIC_KEY="public-key",
        LANGFUSE_SECRET_KEY="secret-key",
        LANGFUSE_BASE_URL="https://langfuse.example.com",
        LANGFUSE_TRACING_ENABLED=True,
    )

    model = build_chat_model(settings=settings)

    assert isinstance(model, _FakeChatOpenAI)
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["api_key"] == "relay-key"
    assert captured["base_url"] == "https://example.com/v1"
    assert captured["streaming"] is True
    assert "disable_streaming" not in captured
    assert "callbacks" in captured
    assert len(captured["callbacks"]) == 1
    assert callback_calls == [{"public_key": "public-key"}]


def test_build_chat_model_omits_langfuse_callbacks_when_unconfigured(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_chat_openai(**kwargs: Any) -> _FakeChatOpenAI:
        captured.update(kwargs)
        return _FakeChatOpenAI(**kwargs)

    monkeypatch.setattr(
        "app.core.llm.ChatOpenAI",
        fake_chat_openai,
    )
    monkeypatch.setattr(
        "app.core.llm.LangfuseTraceLogger.from_settings",
        lambda settings: _FakeDisabledTracer(),
    )

    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        OPENAI_BASE_URL="https://example.com/v1",
        CHAT_MODEL="gpt-4.1-mini",
        LANGFUSE_PUBLIC_KEY="",
        LANGFUSE_SECRET_KEY="",
        LANGFUSE_BASE_URL="",
        LANGFUSE_TRACING_ENABLED=True,
    )

    model = build_chat_model(settings=settings)

    assert isinstance(model, _FakeChatOpenAI)
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["streaming"] is True
    assert "disable_streaming" not in captured
    assert "callbacks" not in captured
