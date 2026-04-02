from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import Settings

try:
    from langfuse import Langfuse
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    Langfuse = None


_SENSITIVE_KEY_FRAGMENTS = (
    "authorization",
    "api_key",
    "secret",
    "token",
    "image_base64",
)


def _is_sensitive_key(key: str) -> bool:
    lowered_key = key.casefold()
    return any(fragment in lowered_key for fragment in _SENSITIVE_KEY_FRAGMENTS)


def sanitize_langfuse_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return _sanitize_langfuse_mapping(metadata)


def _sanitize_langfuse_mapping(
    mapping: dict[str, Any],
    *,
    parent_key: str | None = None,
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in mapping.items():
        if _should_drop_langfuse_key(key=key, parent_key=parent_key):
            continue
        sanitized[key] = _sanitize_langfuse_value(value, parent_key=key)
    return sanitized


def _should_drop_langfuse_key(*, key: str, parent_key: str | None) -> bool:
    if _is_sensitive_key(key):
        return True

    lowered_key = key.casefold()
    lowered_parent = parent_key.casefold() if parent_key else None
    return lowered_key == "data" and lowered_parent in {"inline_data", "inlinedata"}


def _sanitize_langfuse_value(value: Any, *, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        return _sanitize_langfuse_mapping(value, parent_key=parent_key)
    if isinstance(value, list):
        return [_sanitize_langfuse_value(item, parent_key=parent_key) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _sanitize_langfuse_value(item, parent_key=parent_key) for item in value
        )
    return value


@dataclass(slots=True)
class _ManagedSpan:
    logger: "LangfuseTraceLogger"
    span_context_manager: Any = None
    span: Any = None

    def __enter__(self) -> "_ManagedSpan":
        if self.span_context_manager is not None:
            self.span = self.span_context_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.update(
                metadata={
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                },
                level="ERROR",
                status_message=str(exc),
            )

        if self.span_context_manager is None:
            return False
        return bool(self.span_context_manager.__exit__(exc_type, exc, tb))

    def update(
        self,
        *,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        if self.span is None:
            return

        update_kwargs: dict[str, Any] = {}
        if input is not None:
            update_kwargs["input"] = _sanitize_langfuse_value(input)
        if output is not None:
            update_kwargs["output"] = _sanitize_langfuse_value(output)
        if metadata is not None:
            update_kwargs["metadata"] = sanitize_langfuse_metadata(metadata)
        if model is not None:
            update_kwargs["model"] = model
        if model_parameters is not None:
            update_kwargs["model_parameters"] = _sanitize_langfuse_value(
                model_parameters
            )
        if level is not None:
            update_kwargs["level"] = level
        if status_message is not None:
            update_kwargs["status_message"] = status_message

        if update_kwargs:
            self.span.update(**update_kwargs)

    def end(self) -> None:
        if self.span is not None:
            self.span.end()

    def start_as_current_span(
        self,
        *,
        name: str,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> "_ManagedSpan":
        return self.logger.start_as_current_span(
            name=name,
            input=input,
            output=output,
            metadata=metadata,
        )


@dataclass(slots=True)
class LangfuseTraceLogger:
    client: Any = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @classmethod
    def from_settings(cls, settings: Settings) -> "LangfuseTraceLogger":
        if not settings.langfuse_tracing_enabled:
            return cls()
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            return cls()
        if Langfuse is None:
            return cls()

        return cls(
            client=Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_base_url,
                tracing_enabled=settings.langfuse_tracing_enabled,
            )
        )

    def start_as_current_span(
        self,
        *,
        name: str,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> _ManagedSpan:
        if self.client is None:
            return _ManagedSpan(logger=self)

        return _ManagedSpan(
            logger=self,
            span_context_manager=self.client.start_as_current_span(
                name=name,
                input=_sanitize_langfuse_value(input),
                output=_sanitize_langfuse_value(output),
                metadata=sanitize_langfuse_metadata(metadata),
            ),
        )
