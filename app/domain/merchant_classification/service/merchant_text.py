from __future__ import annotations

import re
import unicodedata


def clean_raw_merchant_text(merchant_text: str) -> str:
    raw_text = unicodedata.normalize("NFKC", str(merchant_text or ""))
    raw_text = raw_text.replace("\r", " ").replace("\n", " ")
    raw_text = re.sub(r"\s+", " ", raw_text).strip()
    return raw_text


def normalize_service_merchant_text(merchant_text: str) -> str:
    text = clean_raw_merchant_text(merchant_text)
    return text.casefold()


def build_model_input_text(merchant_text: str) -> str:
    raw_text = clean_raw_merchant_text(merchant_text)
    normalized_text = normalize_service_merchant_text(raw_text)
    return f"{raw_text} [SEP] {normalized_text}"
