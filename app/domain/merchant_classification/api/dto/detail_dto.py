from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.text_validation import validate_non_blank_text


class MerchantInferenceRequest(BaseModel):
    merchant_text: str

    @field_validator("merchant_text")
    @classmethod
    def validate_merchant_text(cls, value: str) -> str:
        return validate_non_blank_text(value, "merchant_text")


class MerchantInferenceResponse(BaseModel):
    merchant_text: str
    normalized_merchant_text: str
    raw_label: str
    final_label: str
    confidence: float
    fallback_applied: bool
    threshold_used: float


class MerchantBatchInferenceRequest(BaseModel):
    merchant_texts: list[str] = Field(min_length=1)

    @field_validator("merchant_texts")
    @classmethod
    def validate_merchant_texts(cls, value: list[str]) -> list[str]:
        return [validate_non_blank_text(item, "merchant_texts") for item in value]


class MerchantBatchInferenceResponse(BaseModel):
    results: list[MerchantInferenceResponse]
