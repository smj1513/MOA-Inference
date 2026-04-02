from __future__ import annotations

import base64
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import LangfuseTraceLogger
from app.domain.image_generation.agents.graph import build_image_generation_graph


class ImageGenerationConfigurationError(RuntimeError):
    """Raised when required image-generation configuration is missing."""


class UpstreamImageGenerationError(RuntimeError):
    """Raised when the relay returns a malformed image-generation payload."""


class ImageGenerationService:
    def __init__(
        self,
        *,
        settings: Settings,
        http_client: Any = None,
        tracer: Any = None,
    ) -> None:
        self._settings = settings
        self._http_client = http_client
        self._tracer = tracer or LangfuseTraceLogger.from_settings(settings)
        self._graph = build_image_generation_graph(
            settings=settings,
            image_generator=self._generate_image,
        )

    async def generate(
        self,
        *,
        user_prompt: str,
        image_url: str,
    ) -> dict[str, str]:
        with self._tracer.start_as_current_span(
            name="image-generation",
            input={
                "user_prompt": user_prompt,
            },
            metadata={
                "model": self._settings.image_generation_model,
                "image_source": "url",
            },
        ) as trace:
            reference_image = await self._download_reference_image(image_url)
            result = await self._graph.ainvoke(
                {
                    "user_prompt": user_prompt,
                    "input_image_base64": reference_image["image_base64"],
                    "input_mime_type": reference_image["mime_type"],
                }
            )
            response = {
                "image_base64": result["generated_image_base64"],
                "mime_type": result["output_mime_type"],
                "model": result["model"],
                "prompt": result["final_prompt"],
            }
            trace.update(
                output={
                    "mime_type": response["mime_type"],
                    "model": response["model"],
                    "prompt": response["prompt"],
                },
                metadata={
                    "final_prompt": response["prompt"],
                    "input_mime_type": reference_image["mime_type"],
                },
            )
            return response

    async def _download_reference_image(self, image_url: str) -> dict[str, str]:
        client = self._get_http_client()

        with self._tracer.start_as_current_span(
            name="download-reference-image",
            metadata={
                "image_source": "url",
            },
        ) as span:
            try:
                response = await client.get(
                    image_url,
                    headers={
                        "Accept": "image/*",
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise UpstreamImageGenerationError(
                    f"Reference image download failed: {exc}"
                ) from exc

            mime_type = self._extract_reference_image_mime_type(response)
            image_bytes = self._extract_reference_image_bytes(response)
            image_base64 = base64.b64encode(image_bytes).decode("ascii")

            span.update(
                output={
                    "mime_type": mime_type,
                },
                metadata={
                    "input_mime_type": mime_type,
                },
            )
            return {
                "image_base64": image_base64,
                "mime_type": mime_type,
            }

    async def _generate_image(
        self,
        *,
        prompt: str,
        model: str,
        image_base64: str,
        mime_type: str,
    ) -> dict[str, str]:
        client = self._get_http_client()
        request_payload = self._build_generate_content_request(
            prompt=prompt,
            image_base64=image_base64,
            mime_type=mime_type,
        )

        with self._tracer.start_as_current_span(
            name="gms-gemini-generate-content",
            input={
                "prompt": prompt,
                "mime_type": mime_type,
            },
            metadata={
                "model": model,
            },
        ) as span:
            try:
                response = await client.post(
                    f"/models/{model}:generateContent",
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self._settings.ai_api_key,
                    },
                    json=request_payload,
                    timeout=60.0,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise UpstreamImageGenerationError(
                    f"Upstream image generation request failed: {exc}"
                ) from exc

            response_payload = self._parse_response_payload(response)
            response_schema = self._describe_response_schema(response_payload)
            span.update(metadata=response_schema)

            generated_image = self._extract_generated_inline_data(
                response_payload,
                response_schema=response_schema,
            )
            span.update(
                output={
                    "mime_type": generated_image["mime_type"],
                    "model": model,
                },
                metadata={
                    "output_mime_type": generated_image["mime_type"],
                },
            )
            return {
                "generated_image_base64": generated_image["data"],
                "mime_type": generated_image["mime_type"],
                "model": model,
            }

    def _get_http_client(self) -> Any:
        if self._http_client is not None:
            return self._http_client

        if not self._settings.ai_api_key or not self._settings.google_ai_base_url:
            raise ImageGenerationConfigurationError(
                "AI_API_KEY and GOOGLE_AI_BASE_URL must be configured for image generation"
            )

        self._http_client = httpx.AsyncClient(
            base_url=self._settings.google_ai_base_url,
            follow_redirects=True,
        )
        return self._http_client

    async def aclose(self) -> None:
        http_client = self._http_client
        if http_client is None:
            return

        close = getattr(http_client, "aclose", None)
        if callable(close):
            await close()
            return

        close = getattr(http_client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _build_generate_content_request(
        *,
        prompt: str,
        image_base64: str,
        mime_type: str,
    ) -> dict[str, Any]:
        return {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64,
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
        }

    @staticmethod
    def _extract_reference_image_bytes(response: Any) -> bytes:
        image_bytes = getattr(response, "content", b"") or b""
        if not image_bytes:
            raise UpstreamImageGenerationError(
                "Reference image URL returned an empty response body"
            )
        return bytes(image_bytes)

    def _extract_reference_image_mime_type(self, response: Any) -> str:
        headers = getattr(response, "headers", {}) or {}
        content_type = headers.get("Content-Type") or headers.get("content-type")
        if content_type is None:
            raise UpstreamImageGenerationError(
                "Reference image URL response did not include Content-Type"
            )

        normalized_content_type = content_type.split(";", maxsplit=1)[0].strip().lower()
        if not normalized_content_type.startswith("image/"):
            raise UpstreamImageGenerationError(
                "Reference image URL did not return an image content type: "
                f"{normalized_content_type}"
            )
        return normalized_content_type

    def _parse_response_payload(self, response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        if hasattr(response, "json"):
            payload = response.json()
            if isinstance(payload, dict):
                return payload
        raise UpstreamImageGenerationError(
            "Upstream image generation response could not be parsed as JSON"
        )

    def _extract_generated_inline_data(
        self,
        response_payload: dict[str, Any],
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        candidates = self._get_response_value(response_payload, "candidates") or []
        if not candidates:
            raise UpstreamImageGenerationError(
                "Upstream image generation response did not include candidates"
            )

        for candidate in candidates:
            content = self._get_response_value(candidate, "content") or {}
            parts = self._get_response_value(content, "parts") or []
            for part in parts:
                inline_data = self._get_first_response_value(
                    part,
                    "inline_data",
                    "inlineData",
                )
                if not inline_data:
                    continue

                generated_image_base64 = self._get_response_value(inline_data, "data")
                generated_mime_type = self._get_first_response_value(
                    inline_data,
                    "mime_type",
                    "mimeType",
                )
                if generated_image_base64 and generated_mime_type:
                    return {
                        "data": generated_image_base64,
                        "mime_type": generated_mime_type,
                    }

        available_part_types: list[str] = []
        if response_schema is not None:
            available_part_types = response_schema.get("first_candidate_part_types", [])

        details = ""
        if available_part_types:
            details = f"; available part types: {available_part_types}"

        raise UpstreamImageGenerationError(
            "Upstream image generation response did not include generated inline_data"
            f"{details}"
        )

    @staticmethod
    def _get_response_value(payload: Any, key: str) -> Any:
        if isinstance(payload, dict):
            return payload.get(key)
        return getattr(payload, key, None)

    def _get_first_response_value(self, payload: Any, *keys: str) -> Any:
        for key in keys:
            value = self._get_response_value(payload, key)
            if value is not None:
                return value
        return None

    def _describe_response_schema(
        self,
        response_payload: dict[str, Any],
    ) -> dict[str, Any]:
        candidates = self._get_response_value(response_payload, "candidates") or []
        first_candidate = candidates[0] if candidates else None
        first_content = self._get_response_value(first_candidate, "content") or {}
        first_parts = self._get_response_value(first_content, "parts") or []

        return {
            "top_level_keys": self._list_payload_keys(response_payload),
            "candidate_count": len(candidates),
            "first_candidate_part_types": self._list_part_types(first_parts),
        }

    @staticmethod
    def _list_payload_keys(payload: Any) -> list[str]:
        if payload is None:
            return []
        if isinstance(payload, dict):
            return sorted(str(key) for key in payload.keys())
        if hasattr(payload, "model_dump"):
            dumped = payload.model_dump()
            if isinstance(dumped, dict):
                return sorted(str(key) for key in dumped.keys())
        if hasattr(payload, "__dict__"):
            return sorted(
                str(key)
                for key in payload.__dict__.keys()
                if not str(key).startswith("_")
            )
        return []

    def _list_part_types(self, parts: list[Any]) -> list[str]:
        part_types: list[str] = []
        for part in parts:
            part_keys = self._list_payload_keys(part)
            if "inline_data" in part_keys:
                part_types.append("inline_data")
                continue
            if "inlineData" in part_keys:
                part_types.append("inlineData")
                continue
            if "text" in part_keys:
                part_types.append("text")
                continue
            if part_keys:
                part_types.append(",".join(part_keys))
                continue
            part_types.append("unknown")
        return part_types
