import pandas as pd

from lab.beauty_retail_relabel import (
    is_beauty_retail_text,
    relabel_beauty_retail_rows,
)


def _row(merchant_text: str, category_id: int, category_name: str) -> dict:
    return {
        "sample_id": f"S-{merchant_text}",
        "split": "train",
        "category_id": category_id,
        "category_name": category_name,
        "merchant_family_id": f"F-{merchant_text}",
        "merchant_family": merchant_text,
        "merchant_text": merchant_text,
        "normalized_merchant_text": merchant_text.casefold(),
        "family_type": "seed",
        "family_source": "unit-test",
        "variant_source": "unit-test",
    }


def test_is_beauty_retail_text_matches_cosmetics_retail_names() -> None:
    assert is_beauty_retail_text("씨제이올리브영 구미인동점")
    assert is_beauty_retail_text("미샤 대구월배점")
    assert is_beauty_retail_text("랄라블라 수원인계점")
    assert is_beauty_retail_text("정다운화장품")
    assert is_beauty_retail_text("한빛코스메틱")


def test_is_beauty_retail_text_excludes_beauty_service_names() -> None:
    assert not is_beauty_retail_text("제이헤어")
    assert not is_beauty_retail_text("오늘네일")
    assert not is_beauty_retail_text("마린피부관리")
    assert not is_beauty_retail_text("정통마사지")
    assert not is_beauty_retail_text("행복사우나")


def test_relabel_beauty_retail_rows_moves_only_matching_beauty_rows() -> None:
    source_df = pd.DataFrame(
        [
            _row("씨제이올리브영 구미인동점", 6, "미용"),
            _row("미샤 대구월배점", 6, "미용"),
            _row("제이헤어", 6, "미용"),
            _row("오늘네일", 6, "미용"),
            _row("올리브영 판교점", 9, "쇼핑/패션"),
        ]
    )

    updated_df, changed_df = relabel_beauty_retail_rows(source_df)

    moved_names = set(changed_df["merchant_text"])
    assert moved_names == {"씨제이올리브영 구미인동점", "미샤 대구월배점"}

    moved_df = updated_df[updated_df["merchant_text"].isin(moved_names)].copy()
    assert set(moved_df["category_id"]) == {9}
    assert set(moved_df["category_name"]) == {"쇼핑/패션"}

    service_df = updated_df[updated_df["merchant_text"].isin({"제이헤어", "오늘네일"})].copy()
    assert set(service_df["category_id"]) == {6}
    assert set(service_df["category_name"]) == {"미용"}

    shopping_row = updated_df[updated_df["merchant_text"] == "올리브영 판교점"].iloc[0]
    assert shopping_row["category_id"] == 9
    assert shopping_row["category_name"] == "쇼핑/패션"
