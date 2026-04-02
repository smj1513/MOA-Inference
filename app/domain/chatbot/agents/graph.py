from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Sequence
from uuid import uuid4

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.core.llm import build_chat_model
from app.core.logging import LangfuseTraceLogger
from app.domain.chatbot.errors import ChatbotUpstreamUnavailableError
from app.domain.chatbot.agents.prompts.chatbot_prompts import (
    build_queryguard_prompt,
    build_querysmith_prompt,
    build_querysmith_schema_digest,
    build_response_guard_prompt,
    build_spendwise_coach_prompt,
    build_sql_executor_prompt,
    build_supervisor_prompt,
)

_DEFAULT_CHECKPOINTER = object()
_FORBIDDEN_RESPONSE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"internal context",
        r"system prompts?",
        r"tool names?",
        r"reasoning traces?",
        r"hidden orchestration details?",
        r"\bmcp\s+(server|tool|tools|adapter|client|call|calls|supervisor)\b",
        r"\bsupervisor\s+(node|agent|prompt|graph)\b",
        r"\b(i|we)\s+(used|called|asked)\b.*\bmcp\b",
        r"\b(i|we)\s+(used|called|asked)\b.*\bsupervisor\b",
        r"analysisincluded\s*:\s*(true|false)",
        r"\bnon_essential\b",
        r"\bincome\b\s+(label|reason|code|status|사유|처리)",
        r"\bout\b\s*(\(|label|reason|code|status|사유|처리)",
        r"\b(raw|backend|internal)\s+(field|label|status|flag|enum)\b",
    )
]
_BUDGET_STATUS_TOOL_NAMES = frozenset({"get_budget_home", "get_budget_dashboard"})
_WEEKLY_SPENDING_FALLBACK_TOOLS = (
    "get_weekly_spending_analysis",
    "get_weekly_expense_graph",
    "get_transaction_list",
)


@dataclass(slots=True)
class ChatbotAgentContext:
    user_id: int
    session_id: str
    user_query: str


@dataclass(slots=True)
class ChatbotToolBundle:
    schema: Sequence[BaseTool] = ()
    sql: Sequence[BaseTool] = ()
    sql_middleware: Sequence[AgentMiddleware] = ()

    @classmethod
    def empty(cls) -> "ChatbotToolBundle":
        return cls()


class _TrustedUserContextMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        return handler(self._inject_user_id(request))

    async def awrap_tool_call(self, request, handler):
        return await handler(self._inject_user_id(request))

    def _inject_user_id(self, request):
        tool_call = _normalized_tool_call(getattr(request, "tool_call", None))
        if tool_call is None:
            return request

        tool = getattr(request, "tool", None)
        if _tool_accepts_user_id(tool):
            runtime = getattr(request, "runtime", None)
            context = getattr(runtime, "context", None)
            user_id = getattr(context, "user_id", None)
            if isinstance(user_id, int):
                args = dict(tool_call.get("args") or {})
                args["user_id"] = user_id
                tool_call["args"] = args
        return request.override(tool_call=tool_call)


class _ModelMessageSanitizingMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        return handler(self._sanitize_request_messages(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._sanitize_request_messages(request))

    def _sanitize_request_messages(self, request):
        normalized_messages = _normalized_model_messages(getattr(request, "messages", []))
        if normalized_messages is request.messages:
            return request
        return request.override(messages=normalized_messages)


class _ToolTraceMiddleware(AgentMiddleware):
    def __init__(self, tracer: LangfuseTraceLogger | None = None) -> None:
        self._tracer = tracer or LangfuseTraceLogger()

    def wrap_tool_call(self, request, handler):
        if not self._tracer.enabled:
            try:
                return handler(request)
            except Exception as error:
                recovered_message = _build_recoverable_tool_error_message(request, error)
                if recovered_message is not None:
                    return recovered_message
                _raise_translated_tool_error(error)

        with self._tracer.start_as_current_span(**self._span_kwargs(request)) as span:
            try:
                response = handler(request)
            except Exception as error:
                recovered_message = _build_recoverable_tool_error_message(request, error)
                if recovered_message is not None:
                    span.update(
                        metadata={
                            "status": "recovered_no_data",
                            "result_type": type(recovered_message).__name__,
                            "error_type": error.__class__.__name__,
                            "error_message": str(error),
                        }
                    )
                    return recovered_message
                translated_error = _translate_tool_runtime_error(error)
                span.update(
                    metadata={
                        "status": "error",
                        "error_type": translated_error.__class__.__name__,
                        "error_message": str(translated_error),
                        "upstream_error_message": str(error),
                    }
                )
                if translated_error is error:
                    raise
                raise translated_error from error
            span.update(
                metadata={
                    "status": "completed",
                    "result_type": type(response).__name__,
                }
            )
            return response

    async def awrap_tool_call(self, request, handler):
        if not self._tracer.enabled:
            try:
                return await handler(request)
            except Exception as error:
                recovered_message = _build_recoverable_tool_error_message(request, error)
                if recovered_message is not None:
                    return recovered_message
                _raise_translated_tool_error(error)

        with self._tracer.start_as_current_span(**self._span_kwargs(request)) as span:
            try:
                response = await handler(request)
            except Exception as error:
                recovered_message = _build_recoverable_tool_error_message(request, error)
                if recovered_message is not None:
                    span.update(
                        metadata={
                            "status": "recovered_no_data",
                            "result_type": type(recovered_message).__name__,
                            "error_type": error.__class__.__name__,
                            "error_message": str(error),
                        }
                    )
                    return recovered_message
                translated_error = _translate_tool_runtime_error(error)
                span.update(
                    metadata={
                        "status": "error",
                        "error_type": translated_error.__class__.__name__,
                        "error_message": str(translated_error),
                        "upstream_error_message": str(error),
                    }
                )
                if translated_error is error:
                    raise
                raise translated_error from error
            span.update(
                metadata={
                    "status": "completed",
                    "result_type": type(response).__name__,
                }
            )
            return response

    def _span_kwargs(self, request) -> dict[str, Any]:
        tool_name = _request_tool_name(request)
        return {
            "name": "chatbot.tool_call",
            "input": {
                "tool_name": tool_name,
                "arguments": _request_tool_arguments(request),
            },
            "metadata": _request_tool_metadata(request, tool_name=tool_name),
        }


class _ChatbotGraphViewState(dict):
    pass


class _ChatbotAgentRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        autoload_mcp_tools: bool,
        model_factory: Callable[[str], BaseChatModel],
        tracer: LangfuseTraceLogger | None,
        tools: ChatbotToolBundle,
        checkpointer: Any,
        today: date | None,
    ) -> None:
        self._settings = settings
        self._autoload_mcp_tools = autoload_mcp_tools
        self._model_factory = model_factory
        self._tracer = tracer or LangfuseTraceLogger()
        self._tool_bundle = tools
        self._checkpointer = None if checkpointer is _DEFAULT_CHECKPOINTER else checkpointer
        self._today = today
        self._agent: Any | None = None
        self._agent_lock = asyncio.Lock()
        self._graph_view: Any | None = None

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        agent = await self._ensure_agent()
        messages = _coerce_messages(input.get("messages"))
        context = _build_context(input)
        result = await agent.ainvoke(
            {"messages": messages},
            config=config,
            context=context,
        )
        runtime_messages = list(result.get("messages") or [])
        warnings = _extract_warnings(runtime_messages)
        tool_summaries = _extract_tool_summaries(runtime_messages)
        final_text = _extract_final_text(runtime_messages)
        return {
            "messages": runtime_messages,
            "user_id": context.user_id,
            "session_id": context.session_id,
            "user_query": context.user_query,
            "warnings": warnings,
            "tool_summaries": tool_summaries,
            "final_response": final_text,
            "final_answer": final_text,
        }

    async def astream(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        *,
        stream_mode: Any = None,
        subgraphs: bool = False,
    ):
        agent = await self._ensure_agent()
        messages = _coerce_messages(input.get("messages"))
        context = _build_context(input)
        async for event in agent.astream(
            {"messages": messages},
            config=config,
            context=context,
            stream_mode=stream_mode,
            subgraphs=subgraphs,
        ):
            yield event

    async def aensure_ready(self) -> None:
        await self._ensure_agent()

    def get_graph(self, xray: bool = False):
        if self._agent is not None:
            return self._agent.get_graph(xray=xray)

        if self._uses_local_tools() or not self._autoload_mcp_tools:
            self._agent = self._build_agent(list(self._local_tools()))
            return self._agent.get_graph(xray=xray)

        if self._graph_view is None:
            self._graph_view = _build_static_graph().get_graph(xray=xray)
        return self._graph_view

    async def _ensure_agent(self):
        if self._agent is not None:
            return self._agent

        async with self._agent_lock:
            if self._agent is not None:
                return self._agent

            if self._uses_local_tools():
                resolved_tools = list(self._local_tools())
            elif self._autoload_mcp_tools:
                resolved_tools = await self._load_mcp_tools()
            else:
                resolved_tools = []

            self._agent = self._build_agent(resolved_tools)
            return self._agent

    def _build_agent(self, resolved_tools: list[BaseTool]):
        return create_agent(
            model=self._model_factory("chatbot"),
            tools=resolved_tools,
            system_prompt=_build_system_prompt(today=self._today),
            middleware=[
                _ModelMessageSanitizingMiddleware(),
                _TrustedUserContextMiddleware(),
                _ToolTraceMiddleware(self._tracer),
                *list(self._tool_bundle.sql_middleware),
            ],
            context_schema=ChatbotAgentContext,
            checkpointer=self._checkpointer,
            name="chatbot_coach",
        )

    async def _load_mcp_tools(self) -> list[BaseTool]:
        client = MultiServerMCPClient(
            {
                "more_finance": {
                    "transport": "http",
                    "url": self._settings.more_finance_mcp_url,
                }
            }
        )
        try:
            return list(await client.get_tools())
        except Exception as error:
            raise _translate_mcp_error(error) from error

    def _local_tools(self) -> Sequence[BaseTool]:
        return [*self._tool_bundle.schema, *self._tool_bundle.sql]

    def _uses_local_tools(self) -> bool:
        return bool(self._tool_bundle.schema or self._tool_bundle.sql)


def build_chatbot_graph(
    *,
    settings: Settings | None = None,
    model_factory: Callable[[str], BaseChatModel] | None = None,
    tracer: LangfuseTraceLogger | None = None,
    tools: ChatbotToolBundle | None = None,
    checkpointer: Any = _DEFAULT_CHECKPOINTER,
    today: date | None = None,
):
    resolved_settings = settings or Settings()
    autoload_mcp_tools = bool(resolved_settings.more_finance_mcp_url.strip())
    resolved_tools = tools or ChatbotToolBundle.empty()
    resolved_model_factory = model_factory or (
        lambda _role: build_chat_model(settings=resolved_settings)
    )
    return _ChatbotAgentRuntime(
        settings=resolved_settings,
        autoload_mcp_tools=autoload_mcp_tools,
        model_factory=resolved_model_factory,
        tracer=tracer,
        tools=resolved_tools,
        checkpointer=checkpointer,
        today=today,
    )


def _build_context(input: dict[str, Any]) -> ChatbotAgentContext:
    user_id = int(input.get("user_id") or 0)
    session_id = str(input.get("session_id") or "")
    user_query = str(input.get("user_query") or _latest_human_text(input.get("messages")))
    return ChatbotAgentContext(
        user_id=user_id,
        session_id=session_id,
        user_query=user_query,
    )


def _build_system_prompt(*, today: date | None) -> str:
    return "\n\n".join(
        part
        for part in (
            build_supervisor_prompt(today=today),
            build_querysmith_schema_digest(),
            build_querysmith_prompt(),
            build_queryguard_prompt(),
            build_sql_executor_prompt(),
            build_spendwise_coach_prompt(today=today),
            build_response_guard_prompt(),
        )
        if part.strip()
    )


def _coerce_messages(value: Any) -> list[BaseMessage]:
    if not isinstance(value, list):
        return []

    normalized_messages: list[BaseMessage] = []
    for item in value:
        if isinstance(item, BaseMessage):
            normalized_messages.append(item)
            continue

        if not isinstance(item, dict):
            continue

        role = str(item.get("role") or "").strip().casefold()
        content = str(item.get("content") or "")
        if role == "assistant":
            normalized_messages.append(AIMessage(content=content))
        else:
            normalized_messages.append(HumanMessage(content=content))
    return normalized_messages


def _latest_human_text(messages: Any) -> str:
    for message in reversed(_coerce_messages(messages)):
        if isinstance(message, HumanMessage):
            text = _message_text(message)
            if text:
                return text
    return ""


def _extract_final_text(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if not isinstance(message, (AIMessage, AIMessageChunk)):
            continue
        if isinstance(message, ToolMessage):
            continue
        if _message_has_tool_calls(message):
            continue

        raw_text = _sanitize_user_facing_text(_message_text(message))
        if _looks_like_internal_response_leak(raw_text):
            return ""
        text = raw_text
        if text:
            return text
    return ""


def _extract_tool_summaries(messages: Sequence[AnyMessage]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        payload = _parse_message_payload(message.content)
        tool_name = ""
        status = ""
        if isinstance(payload, dict):
            tool_name = str(payload.get("tool_name") or "").strip()
            status = str(payload.get("status") or "").strip()

        resolved_name = tool_name or str(getattr(message, "name", "") or "").strip()
        if not resolved_name:
            continue

        resolved_status = status or _infer_tool_status(message.content)
        summaries.append(
            {
                "tool_name": resolved_name,
                "status": resolved_status,
            }
        )
    return summaries


def _extract_warnings(messages: Sequence[AnyMessage]) -> list[str]:
    warnings: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        payload = _parse_message_payload(message.content)
        if not isinstance(payload, dict):
            continue

        raw_warnings = payload.get("warnings")
        if not isinstance(raw_warnings, list):
            continue

        for item in raw_warnings:
            warning = str(item).strip()
            if warning:
                warnings.append(warning)
    return warnings


def _parse_message_payload(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _infer_tool_status(content: Any) -> str:
    if isinstance(content, str) and "reject" in content.casefold():
        return "rejected"
    return "completed"


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    texts.append(text)
        return "".join(texts).strip()
    return str(content).strip()


def _message_has_tool_calls(message: Any) -> bool:
    if isinstance(message, (AIMessage, AIMessageChunk)):
        return bool(message.tool_calls or getattr(message, "tool_call_chunks", None))
    return bool(getattr(message, "tool_calls", None) or getattr(message, "tool_call_chunks", None))


def _sanitize_user_facing_text(text: str) -> str:
    return text.strip()


def _looks_like_internal_response_leak(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    return any(pattern.search(cleaned) for pattern in _FORBIDDEN_RESPONSE_PATTERNS)


def _tool_accepts_user_id(tool: BaseTool | None) -> bool:
    if tool is None:
        return False
    args_schema = getattr(tool, "args_schema", None)
    if isinstance(args_schema, dict):
        properties = args_schema.get("properties")
        return isinstance(properties, dict) and "user_id" in properties
    model_fields = getattr(args_schema, "model_fields", None)
    return isinstance(model_fields, dict) and "user_id" in model_fields


def _normalized_model_messages(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    normalized_messages: list[AnyMessage] = []
    changed = False
    pending_tool_call_ids: set[str] = set()
    for message in messages:
        normalized_message = _normalized_model_message(message)
        if normalized_message is None:
            changed = True
            continue

        if isinstance(normalized_message, ToolMessage):
            tool_call_id = _message_tool_call_id(normalized_message)
            if not tool_call_id or tool_call_id not in pending_tool_call_ids:
                changed = True
                continue
            pending_tool_call_ids.discard(tool_call_id)
        else:
            pending_tool_call_ids = _message_tool_call_ids_set(normalized_message)

        normalized_messages.append(normalized_message)
        if normalized_message is not message:
            changed = True
    return normalized_messages if changed else list(messages)


def _normalized_model_message(message: AnyMessage) -> AnyMessage | None:
    if not isinstance(message, (AIMessage, AIMessageChunk)):
        return message

    updates: dict[str, Any] = {}
    tool_calls, tool_calls_changed = _normalized_tool_call_entries(
        getattr(message, "tool_calls", None),
        openai_format=False,
    )
    if tool_calls_changed:
        updates["tool_calls"] = tool_calls

    invalid_tool_calls, invalid_tool_calls_changed = _normalized_tool_call_entries(
        getattr(message, "invalid_tool_calls", None),
        openai_format=False,
    )
    if invalid_tool_calls_changed:
        updates["invalid_tool_calls"] = invalid_tool_calls

    additional_kwargs = getattr(message, "additional_kwargs", None)
    normalized_additional_kwargs: dict[str, Any] | None = None
    normalized_raw_tool_calls: Any = None
    if isinstance(additional_kwargs, dict):
        raw_tool_calls = additional_kwargs.get("tool_calls")
        normalized_raw_tool_calls, raw_tool_calls_changed = _normalized_tool_call_entries(
            raw_tool_calls,
            openai_format=True,
        )
        if raw_tool_calls_changed:
            normalized_additional_kwargs = dict(additional_kwargs)
            if normalized_raw_tool_calls:
                normalized_additional_kwargs["tool_calls"] = normalized_raw_tool_calls
            else:
                normalized_additional_kwargs.pop("tool_calls", None)
            updates["additional_kwargs"] = normalized_additional_kwargs

    resolved_tool_calls = updates.get("tool_calls", getattr(message, "tool_calls", None))
    resolved_invalid_tool_calls = updates.get(
        "invalid_tool_calls",
        getattr(message, "invalid_tool_calls", None),
    )
    if normalized_additional_kwargs is not None:
        resolved_raw_tool_calls = normalized_additional_kwargs.get("tool_calls")
    elif isinstance(additional_kwargs, dict):
        resolved_raw_tool_calls = additional_kwargs.get("tool_calls")
    else:
        resolved_raw_tool_calls = normalized_raw_tool_calls

    had_tool_call_metadata = bool(
        getattr(message, "tool_calls", None)
        or getattr(message, "invalid_tool_calls", None)
        or (
            isinstance(additional_kwargs, dict)
            and "tool_calls" in additional_kwargs
        )
    )
    if (
        had_tool_call_metadata
        and not _message_text(message)
        and not resolved_tool_calls
        and not resolved_invalid_tool_calls
        and not resolved_raw_tool_calls
    ):
        return None

    if not updates:
        return message
    return message.model_copy(update=updates)


def _normalized_tool_call_entries(
    tool_calls: Any,
    *,
    openai_format: bool,
) -> tuple[Any, bool]:
    if not isinstance(tool_calls, list):
        return tool_calls, False

    normalized_entries: list[Any] = []
    changed = False
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            normalized_entries.append(tool_call)
            continue

        if not _tool_call_name_is_present(tool_call, openai_format=openai_format):
            changed = True
            continue

        normalized_tool_call = dict(tool_call)
        normalized_id = _normalized_tool_call_id(normalized_tool_call.get("id"))
        if normalized_tool_call.get("id") != normalized_id:
            changed = True
        normalized_tool_call["id"] = normalized_id
        normalized_entries.append(normalized_tool_call)

    return normalized_entries, changed


def _normalized_tool_call(tool_call: Any) -> dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None

    normalized = dict(tool_call)
    normalized["id"] = _normalized_tool_call_id(normalized.get("id"))
    return normalized


def _normalized_tool_call_id(tool_call_id: Any) -> str:
    normalized_tool_call_id = str(tool_call_id or "").strip()
    if normalized_tool_call_id:
        return normalized_tool_call_id
    return f"server-tool-call-{uuid4().hex}"


def _tool_call_name_is_present(tool_call: dict[str, Any], *, openai_format: bool) -> bool:
    if openai_format:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            return False
        tool_name = function.get("name")
    else:
        tool_name = tool_call.get("name")

    return isinstance(tool_name, str) and bool(tool_name.strip())


def _message_tool_call_id(message: ToolMessage) -> str | None:
    tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
    return tool_call_id or None


def _message_tool_call_ids_set(message: AnyMessage) -> set[str]:
    if not isinstance(message, (AIMessage, AIMessageChunk)):
        return set()

    tool_call_ids = {
        tool_call_id
        for tool_call_id in (
            _tool_call_entry_id(tool_call)
            for tool_call in getattr(message, "tool_calls", []) or []
        )
        if tool_call_id is not None
    }
    tool_call_ids.update(
        tool_call_id
        for tool_call_id in (
            _tool_call_entry_id(tool_call)
            for tool_call in getattr(message, "invalid_tool_calls", []) or []
        )
        if tool_call_id is not None
    )

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        raw_tool_calls = additional_kwargs.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            tool_call_ids.update(
                tool_call_id
                for tool_call_id in (
                    _tool_call_entry_id(tool_call)
                    for tool_call in raw_tool_calls
                )
                if tool_call_id is not None
            )

    return tool_call_ids


def _tool_call_entry_id(tool_call: Any) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    tool_call_id = str(tool_call.get("id") or "").strip()
    return tool_call_id or None


def _translate_mcp_error(error: Exception) -> ChatbotUpstreamUnavailableError:
    status_error = _find_http_status_error(error)
    if status_error is None:
        return ChatbotUpstreamUnavailableError(
            "Chatbot coaching backend is temporarily unavailable."
        )

    return ChatbotUpstreamUnavailableError(
        "Chatbot coaching backend is temporarily unavailable. "
        f"Upstream MCP responded with HTTP {status_error.response.status_code}.",
        upstream_status_code=status_error.response.status_code,
    )


def _translate_tool_runtime_error(error: Exception) -> Exception:
    if isinstance(error, ChatbotUpstreamUnavailableError):
        return error

    if isinstance(error, ToolException):
        error_message = str(error).strip()
        return ChatbotUpstreamUnavailableError(
            "Chatbot coaching backend tool execution failed. "
            f"Upstream tool reported: {error_message}"
        )

    return error


def _build_recoverable_tool_error_message(
    request: Any,
    error: Exception,
) -> ToolMessage | None:
    if not isinstance(error, ToolException):
        return None

    error_message = str(error).strip()
    if "user not found" not in error_message.casefold():
        return None

    tool_name = _request_tool_name(request) or "unknown_tool"
    tool_call = getattr(request, "tool_call", {}) or {}
    tool_call_id = str(tool_call.get("id") or "").strip()
    suggested_next_tools = _suggested_no_data_fallback_tools(
        request=request,
        tool_name=tool_name,
    )
    payload = {
        "tool_name": tool_name,
        "status": "no_data",
        "warnings": [
            "Current finance records were not available for this user context."
        ],
        "observation": _build_no_data_observation(
            request=request,
            tool_name=tool_name,
            suggested_next_tools=suggested_next_tools,
        ),
    }
    if suggested_next_tools:
        payload["suggested_next_tools"] = list(suggested_next_tools)
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        name=tool_name,
        tool_call_id=tool_call_id,
        status="error",
    )


def _raise_translated_tool_error(error: Exception) -> None:
    translated_error = _translate_tool_runtime_error(error)
    if translated_error is error:
        raise error
    raise translated_error from error


def _build_no_data_observation(
    *,
    request: Any,
    tool_name: str,
    suggested_next_tools: Sequence[str],
) -> str:
    if suggested_next_tools:
        primary_tool = suggested_next_tools[0]
        secondary_tools = ", ".join(suggested_next_tools[1:])
        return (
            "No current finance records were available from the budget-status snapshot "
            "for this user context. This user asked for weekly spending feedback. "
            f"Before giving general coaching, try {primary_tool}. "
            f"If that is unavailable, try {secondary_tools} for this week's spending evidence. "
            "Only continue with general coaching without tool evidence if those tools are also unavailable. "
            "Tell the user that current financial records could not be fully loaded, "
            "avoid blaming the user, and avoid unsupported numeric claims."
        )

    return (
        "No current finance records were available for this user context. "
        "Continue with general coaching without tool evidence. "
        "Tell the user that current financial records could not be loaded, "
        "avoid blaming the user, and avoid unsupported numeric claims."
    )


def _suggested_no_data_fallback_tools(
    *,
    request: Any,
    tool_name: str,
) -> tuple[str, ...]:
    if tool_name not in _BUDGET_STATUS_TOOL_NAMES:
        return ()

    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", None)
    user_query = str(getattr(context, "user_query", "") or "").strip()
    if not _looks_like_weekly_spending_feedback_query(user_query):
        return ()

    return _WEEKLY_SPENDING_FALLBACK_TOOLS


def _looks_like_weekly_spending_feedback_query(user_query: str) -> bool:
    normalized_query = user_query.casefold()
    if not normalized_query:
        return False

    has_week_signal = any(
        token in normalized_query
        for token in ("이번주", "이번 주", "this week", "weekly")
    )
    has_spending_signal = any(
        token in normalized_query
        for token in (
            "소비",
            "지출",
            "내역",
            "피드백",
            "리뷰",
            "분석",
            "spending",
            "expense",
            "expenses",
            "feedback",
            "review",
            "summary",
        )
    )
    return has_week_signal and has_spending_signal


def _find_http_status_error(error: BaseException) -> httpx.HTTPStatusError | None:
    if isinstance(error, httpx.HTTPStatusError):
        return error

    nested = getattr(error, "exceptions", None)
    if not isinstance(nested, (list, tuple)):
        return None

    for item in nested:
        if not isinstance(item, BaseException):
            continue
        resolved = _find_http_status_error(item)
        if resolved is not None:
            return resolved
    return None


def _request_tool_name(request: Any) -> str:
    tool = getattr(request, "tool", None)
    tool_name = str(getattr(tool, "name", "") or "").strip()
    if tool_name:
        return tool_name

    tool_call = getattr(request, "tool_call", {}) or {}
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or "").strip()
    return ""


def _request_tool_arguments(request: Any) -> dict[str, Any]:
    tool_call = getattr(request, "tool_call", {}) or {}
    if not isinstance(tool_call, dict):
        return {}

    args = tool_call.get("args") or {}
    return dict(args) if isinstance(args, dict) else {}


def _request_tool_metadata(request: Any, *, tool_name: str) -> dict[str, Any]:
    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", None)
    tool_call = getattr(request, "tool_call", {}) or {}
    metadata: dict[str, Any] = {
        "tool_name": tool_name,
    }
    if isinstance(tool_call, dict):
        tool_call_id = str(tool_call.get("id") or "").strip()
        if tool_call_id:
            metadata["tool_call_id"] = tool_call_id
    user_id = getattr(context, "user_id", None)
    if isinstance(user_id, int):
        metadata["user_id"] = user_id
    session_id = str(getattr(context, "session_id", "") or "").strip()
    if session_id:
        metadata["session_id"] = session_id
    return metadata


def _build_static_graph():
    graph = StateGraph(_ChatbotGraphViewState)
    graph.add_node("coach_agent", lambda state: state)
    graph.add_edge(START, "coach_agent")
    graph.add_edge("coach_agent", END)
    return graph.compile()
