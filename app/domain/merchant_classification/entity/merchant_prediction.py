from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class MerchantPrediction:
    merchant_text: str
    normalized_merchant_text: str
    raw_label: str
    final_label: str
    confidence: float
    fallback_applied: bool
    threshold_used: float

    def to_dict(self) -> dict[str, str | float | bool]:
        return asdict(self)
