from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_classifier, get_settings
from app.api.serializers import serialize_prediction
from app.domain.merchant_classification.api.dto.detail_dto import (
    MerchantBatchInferenceRequest,
    MerchantBatchInferenceResponse,
    MerchantInferenceRequest,
    MerchantInferenceResponse,
)

router = APIRouter(prefix="/v1", tags=["inference"])


@router.post("/merchant", response_model=MerchantInferenceResponse)
def infer_merchant(
    request: MerchantInferenceRequest,
    classifier=Depends(get_classifier),
) -> dict[str, object]:
    return serialize_prediction(classifier.predict_one(request.merchant_text))


@router.post("/merchant/batch", response_model=MerchantBatchInferenceResponse)
def infer_merchant_batch(
    request: MerchantBatchInferenceRequest,
    classifier=Depends(get_classifier),
    settings=Depends(get_settings),
) -> dict[str, list[dict[str, object]]]:
    if len(request.merchant_texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"merchant_texts must not exceed {settings.max_batch_size} items",
        )
    return {
        "results": [
            serialize_prediction(result)
            for result in classifier.predict_many(request.merchant_texts)
        ]
    }
