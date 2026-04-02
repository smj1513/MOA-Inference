from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ResolvedPrediction:
    raw_label: str
    final_label: str
    confidence: float
    fallback_applied: bool
