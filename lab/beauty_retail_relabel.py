from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd


BEAUTY_CATEGORY_ID = 6
BEAUTY_CATEGORY_NAME = "미용"
SHOPPING_CATEGORY_ID = 9
SHOPPING_CATEGORY_NAME = "쇼핑/패션"

BEAUTY_RETAIL_KEYWORDS = (
    "올리브영",
    "씨제이올리브영",
    "랄라블라",
    "롭스",
    "미샤",
    "아리따움",
    "토니모리",
    "이니스프리",
    "네이처리퍼블릭",
    "더페이스샵",
    "스킨푸드",
    "에뛰드",
    "에뛰드하우스",
    "홀리카홀리카",
    "클리오",
    "롬앤",
    "에스쁘아",
    "바닐라코",
    "오휘",
    "숨",
    "닥터지",
    "화장품",
    "코스메틱",
    "뷰티스토어",
    "드럭스토어",
)

BEAUTY_SERVICE_KEYWORDS = (
    "미용실",
    "헤어",
    "네일",
    "피부관리",
    "피부 관리",
    "에스테틱",
    "왁싱",
    "속눈썹",
    "마사지",
    "안마",
    "사우나",
    "목욕",
    "메이크업",
)


def _normalize_text(text: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.replace("\r", " ").replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def _contains_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword.casefold() in text for keyword in keywords)


def is_beauty_retail_text(text: object) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if not _contains_any_keyword(normalized, BEAUTY_RETAIL_KEYWORDS):
        return False
    if _contains_any_keyword(normalized, BEAUTY_SERVICE_KEYWORDS):
        return False
    return True


def relabel_beauty_retail_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    updated_df = df.copy()

    beauty_mask = (
        updated_df["category_id"].eq(BEAUTY_CATEGORY_ID)
        & updated_df["category_name"].eq(BEAUTY_CATEGORY_NAME)
    )
    retail_mask = updated_df["merchant_text"].map(is_beauty_retail_text)
    relabel_mask = beauty_mask & retail_mask

    updated_df.loc[relabel_mask, "category_id"] = SHOPPING_CATEGORY_ID
    updated_df.loc[relabel_mask, "category_name"] = SHOPPING_CATEGORY_NAME

    changed_df = updated_df.loc[relabel_mask].copy().reset_index(drop=True)
    return updated_df, changed_df
