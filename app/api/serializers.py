from __future__ import annotations

from typing import Any


def serialize_prediction(prediction: Any) -> dict[str, Any]:
    if hasattr(prediction, "to_dict"):
        return prediction.to_dict()
    return prediction
