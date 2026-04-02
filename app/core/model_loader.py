from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import Settings
from app.domain.merchant_classification.service.merchant_classifier_service import (
    MerchantClassifierService,
)

EXPECTED_MODEL_TYPE = "electra"
EXPECTED_MODEL_ARCHITECTURE = "ElectraForSequenceClassification"
EXPECTED_V3_ARCHITECTURE = "MerchantAttentionPoolingClassifier"
EXPECTED_V3_POOLING_STRATEGY = "token_attention_pooling"


def _validate_loaded_sequence_classifier(model, model_source: str) -> None:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    architectures = list(getattr(config, "architectures", []) or [])
    pooling_strategy = getattr(config, "pooling_strategy", None)

    if model_type != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"Model '{model_source}' resolved to model_type='{model_type}', "
            f"architectures={architectures}, expected model_type="
            f"'{EXPECTED_MODEL_TYPE}' and architecture "
            f"'{EXPECTED_MODEL_ARCHITECTURE}'."
        )

    if not architectures:
        return

    if EXPECTED_MODEL_ARCHITECTURE in architectures:
        return

    if (
        EXPECTED_V3_ARCHITECTURE in architectures
        and pooling_strategy == EXPECTED_V3_POOLING_STRATEGY
    ):
        return

    if architectures:
        raise ValueError(
            f"Model '{model_source}' resolved to architectures={architectures}, "
            f"pooling_strategy='{pooling_strategy}', expected "
            f"'{EXPECTED_MODEL_ARCHITECTURE}' or "
            f"'{EXPECTED_V3_ARCHITECTURE}' with pooling_strategy="
            f"'{EXPECTED_V3_POOLING_STRATEGY}'."
        )


def load_sequence_classifier(settings: Settings) -> dict[str, object]:
    model_source = settings.model_source
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        revision=settings.model_revision,
        token=settings.hf_token,
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        revision=settings.model_revision,
        token=settings.hf_token,
        trust_remote_code=True,
    )
    _validate_loaded_sequence_classifier(model, model_source)
    model = model.to(settings.model_device)
    model.eval()

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


def build_merchant_classifier(settings: Settings) -> MerchantClassifierService:
    artifacts = load_sequence_classifier(settings)
    return MerchantClassifierService(
        tokenizer=artifacts["tokenizer"],
        model=artifacts["model"],
        fallback_label=settings.fallback_label,
        fallback_threshold=settings.fallback_threshold,
        inference_batch_size=settings.inference_batch_size,
        device=settings.model_device,
    )
