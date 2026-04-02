from __future__ import annotations

from app.domain.merchant_classification.entity.resolved_prediction import (
    ResolvedPrediction,
)


def resolve_final_label(
    raw_label: str,
    confidence: float,
    fallback_label: str,
    fallback_threshold: float,
) -> ResolvedPrediction:
    final_label = fallback_label if confidence < fallback_threshold else raw_label
    return ResolvedPrediction(
        raw_label=raw_label,
        final_label=final_label,
        confidence=confidence,
        fallback_applied=final_label != raw_label,
    )
