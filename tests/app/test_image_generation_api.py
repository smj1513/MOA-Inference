from __future__ import annotations

from fastapi.testclient import TestClient

from app.domain.image_generation.service.image_generation_service import (
    UpstreamImageGenerationError,
)
from app.main import create_app


class _FakeImageGenerationService:
    def __init__(
        self,
        result: dict[str, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or {
            "image_base64": "encoded-image",
            "mime_type": "image/png",
            "model": "gemini-2.5-flash-image",
            "prompt": "rendered prompt",
        }
        self.error = error
        self.calls: list[dict[str, str]] = []

    async def generate(
        self,
        *,
        user_prompt: str,
        image_url: str,
    ) -> dict[str, str]:
        self.calls.append(
            {
                "user_prompt": user_prompt,
                "image_url": image_url,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result


def test_image_generation_endpoint_returns_base64_payload() -> None:
    service = _FakeImageGenerationService()
    app = create_app(classifier=object(), image_generation_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/v1/image-generation",
            json={
                "user_prompt": "Turn the reference image into a polished cyberpunk banana poster.",
                "image_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/reference-image.jpg",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "image_base64": "encoded-image",
        "mime_type": "image/png",
        "model": "gemini-2.5-flash-image",
        "prompt": "rendered prompt",
    }
    assert service.calls == [
        {
            "user_prompt": "Turn the reference image into a polished cyberpunk banana poster.",
            "image_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/reference-image.jpg",
        }
    ]


def test_image_generation_endpoint_rejects_blank_or_invalid_fields() -> None:
    app = create_app(
        classifier=object(),
        image_generation_service=_FakeImageGenerationService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/image-generation",
            json={
                "user_prompt": "   ",
                "image_url": "not-a-url",
            },
        )

    assert response.status_code == 422


def test_image_generation_endpoint_maps_upstream_errors_to_bad_gateway() -> None:
    app = create_app(
        classifier=object(),
        image_generation_service=_FakeImageGenerationService(
            error=UpstreamImageGenerationError("missing generated image")
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/image-generation",
            json={
                "user_prompt": "Turn the reference image into a polished cyberpunk banana poster.",
                "image_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/reference-image.jpg",
            },
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "missing generated image"}


def test_image_generation_openapi_describes_public_image_url_requirement_in_korean() -> None:
    app = create_app(
        classifier=object(),
        image_generation_service=_FakeImageGenerationService(),
    )

    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    request_schema = response.json()["components"]["schemas"]["ImageGenerationRequest"]
    image_url_schema = request_schema["properties"]["image_url"]
    operation = response.json()["paths"]["/v1/image-generation"]["post"]

    assert "공개적으로 접근 가능한 입력 이미지 URL" in image_url_schema["description"]
    assert "서버가 이 URL에서 이미지를 내려받아 Gemini로 전달합니다" in image_url_schema[
        "description"
    ]
    assert "https://" in image_url_schema["description"]
    assert "image_url에는 공개적으로 접근 가능한 입력 이미지 URL을 넣습니다" in operation[
        "description"
    ]
    assert "S3" in operation["description"]
