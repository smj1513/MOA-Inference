from pydantic import AnyHttpUrl, BaseModel, Field, field_validator


class ImageGenerationRequest(BaseModel):
    user_prompt: str
    image_url: AnyHttpUrl = Field(
        description=(
            "공개적으로 접근 가능한 입력 이미지 URL입니다. "
            "서버가 이 URL에서 이미지를 내려받아 Gemini로 전달합니다. "
            "예: https://bucket.s3.ap-northeast-2.amazonaws.com/reference-image.jpg"
        ),
        examples=[
            "https://bucket.s3.ap-northeast-2.amazonaws.com/reference-image.jpg"
        ],
    )

    @field_validator("user_prompt")
    @classmethod
    def validate_non_blank_user_prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("user_prompt must not be blank")
        return normalized


class ImageGenerationResponse(BaseModel):
    image_base64: str
    mime_type: str
    model: str
    prompt: str
