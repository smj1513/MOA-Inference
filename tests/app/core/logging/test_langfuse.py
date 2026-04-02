from app.core.config import Settings
from app.core.logging.langfuse import LangfuseTraceLogger, sanitize_langfuse_metadata


def test_sanitize_langfuse_metadata_removes_sensitive_fields() -> None:
    metadata = sanitize_langfuse_metadata(
        {
            "user_prompt": "make it cinematic",
            "input_image_base64": "secret-image",
            "authorization": "Bearer secret",
            "api_key": "secret-key",
            "nested_api_token": "secret-token",
            "model": "gemini-2.5-flash-image",
            "parts": [
                {"text": "keep the lighting dramatic"},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": "really-secret-image",
                    }
                },
            ],
        }
    )

    assert metadata == {
        "user_prompt": "make it cinematic",
        "model": "gemini-2.5-flash-image",
        "parts": [
            {"text": "keep the lighting dramatic"},
            {
                "inline_data": {
                    "mime_type": "image/png",
                }
            },
        ],
    }


def test_langfuse_logger_is_noop_when_unconfigured() -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        OPENAI_BASE_URL="https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta",
    )

    logger = LangfuseTraceLogger.from_settings(settings)

    assert logger.enabled is False


def test_langfuse_logger_disabled_mode_allows_safe_calls() -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        OPENAI_BASE_URL="https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta",
        LANGFUSE_TRACING_ENABLED=False,
    )

    logger = LangfuseTraceLogger.from_settings(settings)

    with logger.start_as_current_span(
        name="image-generation",
        input={"user_prompt": "edit the product shot"},
        metadata={"input_image_base64": "secret-image"},
    ) as trace:
        trace.update(
            output={
                "generated_image_base64": "secret-image",
                "model": "gemini-2.5-flash-image",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": "secret-image",
                        }
                    }
                ],
            }
        )

        with trace.start_as_current_span(
            name="gms-gemini-generate-content",
            metadata={"authorization": "Bearer secret", "mime_type": "image/png"},
        ) as span:
            span.update(metadata={"api_key": "secret-key", "candidate_count": 1})
