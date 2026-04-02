from __future__ import annotations

import base64

import anyio
import pytest

from app.core.config import Settings
from app.domain.image_generation.service.image_generation_service import (
    ImageGenerationService,
    ImageGenerationConfigurationError,
    UpstreamImageGenerationError,
)


USER_PROMPT = "Turn the reference image into a polished cyberpunk banana poster."
INPUT_IMAGE_URL = "https://bucket.s3.ap-northeast-2.amazonaws.com/reference-image.jpg"
DOWNLOADED_IMAGE_BYTES = b"reference-image"
DOWNLOADED_IMAGE_BASE64 = base64.b64encode(DOWNLOADED_IMAGE_BYTES).decode("ascii")


class _FakeResponse:
    def __init__(
        self,
        *,
        payload: dict | None = None,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        if self.payload is None:
            raise AssertionError("json() should not be called for this fake response")
        return self.payload


class _FakeHttpClient:
    def __init__(
        self,
        *,
        get_response: _FakeResponse | None = None,
        post_response: _FakeResponse | None = None,
    ) -> None:
        self.get_response = get_response or _FakeResponse()
        self.post_response = post_response or _FakeResponse(payload={})
        self.get_calls: list[dict[str, object]] = []
        self.post_calls: list[dict[str, object]] = []

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.get_calls.append(
            {
                "url": url,
                "headers": headers or {},
                "timeout": timeout,
            }
        )
        return self.get_response

    async def post(
        self,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.post_calls.append(
            {
                "path": path,
                "headers": headers or {},
                "json": json or {},
                "timeout": timeout,
            }
        )
        return self.post_response


class _CapturingAsyncClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _run_async(awaitable, /):
    async def runner():
        return await awaitable

    return anyio.run(runner)


class _CapturingSpan:
    def __init__(
        self,
        recorder: list[dict[str, object]],
        *,
        name: str,
        input: object = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.recorder = recorder
        self.recorder.append(
            {
                "event": "start",
                "name": name,
                "input": input,
                "metadata": metadata or {},
            }
        )

    def __enter__(self) -> _CapturingSpan:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def update(self, **kwargs) -> None:
        self.recorder.append({"event": "update", **kwargs})

    def start_as_current_span(
        self,
        *,
        name: str,
        input: object = None,
        metadata: dict[str, object] | None = None,
    ) -> _CapturingSpan:
        return _CapturingSpan(
            self.recorder,
            name=name,
            input=input,
            metadata=metadata,
        )


class _CapturingTracer:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def start_as_current_span(
        self,
        *,
        name: str,
        input: object = None,
        output: object = None,
        metadata: dict[str, object] | None = None,
    ) -> _CapturingSpan:
        del output
        return _CapturingSpan(
            self.events,
            name=name,
            input=input,
            metadata=metadata,
        )


def test_service_downloads_reference_image_and_extracts_generated_inline_data() -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        GOOGLE_AI_BASE_URL="https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta",
    )
    http_client = _FakeHttpClient(
        get_response=_FakeResponse(
            content=DOWNLOADED_IMAGE_BYTES,
            headers={"Content-Type": "image/jpeg; charset=binary"},
        ),
        post_response=_FakeResponse(
            payload={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Edited successfully."},
                                {
                                    "inline_data": {
                                        "mime_type": "image/png",
                                        "data": "encoded-image",
                                    }
                                },
                            ]
                        }
                    }
                ]
            }
        ),
    )
    tracer = _CapturingTracer()

    service = ImageGenerationService(
        settings=settings,
        http_client=http_client,
        tracer=tracer,
    )

    result = _run_async(
        service.generate(
            user_prompt=USER_PROMPT,
            image_url=INPUT_IMAGE_URL,
        )
    )

    assert result == {
        "image_base64": "encoded-image",
        "mime_type": "image/png",
        "model": "gemini-2.5-flash-image",
        "prompt": result["prompt"],
    }
    assert USER_PROMPT in result["prompt"]
    assert "512x512" in result["prompt"]
    assert http_client.get_calls == [
        {
            "url": INPUT_IMAGE_URL,
            "headers": {
                "Accept": "image/*",
            },
            "timeout": 30.0,
        }
    ]
    assert http_client.post_calls == [
        {
            "path": "/models/gemini-2.5-flash-image:generateContent",
            "headers": {
                "Content-Type": "application/json",
                "x-goog-api-key": "relay-key",
            },
            "json": {
                "contents": [
                    {
                        "parts": [
                            {"text": result["prompt"]},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": DOWNLOADED_IMAGE_BASE64,
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {
                    "imageConfig": {
                        "aspectRatio": "1:1",
                    }
                },
            },
            "timeout": 60.0,
        }
    ]
    assert all(DOWNLOADED_IMAGE_BASE64 not in str(event) for event in tracer.events)
    assert all("encoded-image" not in str(event) for event in tracer.events)


def test_service_accepts_camel_case_inline_data_from_upstream() -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        GOOGLE_AI_BASE_URL="https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta",
    )
    service = ImageGenerationService(
        settings=settings,
        http_client=_FakeHttpClient(
            get_response=_FakeResponse(
                content=DOWNLOADED_IMAGE_BYTES,
                headers={"Content-Type": "image/jpeg"},
            ),
            post_response=_FakeResponse(
                payload={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "Edited successfully."},
                                    {
                                        "inlineData": {
                                            "mimeType": "image/png",
                                            "data": "camel-case-image",
                                        }
                                    },
                                ]
                            }
                        }
                    ]
                }
            ),
        ),
        tracer=_CapturingTracer(),
    )

    result = _run_async(
        service.generate(
            user_prompt=USER_PROMPT,
            image_url=INPUT_IMAGE_URL,
        )
    )

    assert result["image_base64"] == "camel-case-image"
    assert result["mime_type"] == "image/png"
    assert result["model"] == "gemini-2.5-flash-image"


def test_service_raises_when_reference_url_does_not_return_image_content_type() -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        GOOGLE_AI_BASE_URL="https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta",
    )
    http_client = _FakeHttpClient(
        get_response=_FakeResponse(
            content=b"<html>not an image</html>",
            headers={"Content-Type": "text/html"},
        ),
    )
    service = ImageGenerationService(
        settings=settings,
        http_client=http_client,
        tracer=_CapturingTracer(),
    )

    with pytest.raises(UpstreamImageGenerationError) as exc_info:
        _run_async(
            service.generate(
                user_prompt=USER_PROMPT,
                image_url=INPUT_IMAGE_URL,
            )
        )

    assert "Reference image URL did not return an image content type" in str(
        exc_info.value
    )
    assert http_client.post_calls == []


def test_service_raises_when_generated_inline_data_is_missing() -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        GOOGLE_AI_BASE_URL="https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta",
    )
    service = ImageGenerationService(
        settings=settings,
        http_client=_FakeHttpClient(
            get_response=_FakeResponse(
                content=DOWNLOADED_IMAGE_BYTES,
                headers={"Content-Type": "image/jpeg"},
            ),
            post_response=_FakeResponse(
                payload={
                    "candidates": [
                        {"content": {"parts": [{"text": "No image generated."}]}}
                    ]
                }
            ),
        ),
        tracer=_CapturingTracer(),
    )

    with pytest.raises(UpstreamImageGenerationError):
        _run_async(
            service.generate(
                user_prompt=USER_PROMPT,
                image_url=INPUT_IMAGE_URL,
            )
        )


def test_service_reports_available_upstream_part_types_for_schema_debugging() -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        GOOGLE_AI_BASE_URL="https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta",
    )
    tracer = _CapturingTracer()
    service = ImageGenerationService(
        settings=settings,
        http_client=_FakeHttpClient(
            get_response=_FakeResponse(
                content=DOWNLOADED_IMAGE_BYTES,
                headers={"Content-Type": "image/jpeg"},
            ),
            post_response=_FakeResponse(
                payload={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "Model returned only a textual description."}
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 42},
                }
            ),
        ),
        tracer=tracer,
    )

    with pytest.raises(UpstreamImageGenerationError) as exc_info:
        _run_async(
            service.generate(
                user_prompt=USER_PROMPT,
                image_url=INPUT_IMAGE_URL,
            )
        )

    assert "available part types: ['text']" in str(exc_info.value)
    assert any(
        event.get("metadata", {}).get("first_candidate_part_types") == ["text"]
        for event in tracer.events
        if event.get("event") == "update"
    )


def test_service_builds_http_client_with_google_ai_base_url(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_async_client(**kwargs):
        captured.update(kwargs)
        return _CapturingAsyncClient(**kwargs)

    monkeypatch.setattr(
        "app.domain.image_generation.service.image_generation_service.httpx.AsyncClient",
        fake_async_client,
    )

    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        OPENAI_BASE_URL="https://chat.example.com/v1",
        GOOGLE_AI_BASE_URL="https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta",
    )
    service = ImageGenerationService(
        settings=settings,
        tracer=_CapturingTracer(),
    )

    client = service._get_http_client()

    assert isinstance(client, _CapturingAsyncClient)
    assert captured["base_url"] == settings.google_ai_base_url
    assert captured["follow_redirects"] is True


def test_service_requires_google_ai_base_url_for_http_client() -> None:
    settings = Settings(
        _env_file=None,
        AI_API_KEY="relay-key",
        OPENAI_BASE_URL="https://chat.example.com/v1",
        GOOGLE_AI_BASE_URL="",
    )
    service = ImageGenerationService(
        settings=settings,
        tracer=_CapturingTracer(),
    )

    with pytest.raises(ImageGenerationConfigurationError) as exc_info:
        service._get_http_client()

    assert "GOOGLE_AI_BASE_URL" in str(exc_info.value)
