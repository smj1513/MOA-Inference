import shutil
from pathlib import Path

import pandas as pd

from lab.merchant_info_augmentation import (
    augment_dataset_splits,
    normalize_merchant_text,
)


DATASET_COLUMNS = [
    "sample_id",
    "split",
    "category_id",
    "category_name",
    "merchant_family_id",
    "merchant_family",
    "merchant_text",
    "normalized_merchant_text",
    "family_type",
    "family_source",
    "variant_source",
]


def _base_split(
    split: str,
    sample_id: str,
    category_id: int,
    category_name: str,
    merchant_text: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "split": split,
                "category_id": category_id,
                "category_name": category_name,
                "merchant_family_id": f"BASE-{sample_id}",
                "merchant_family": merchant_text,
                "merchant_text": merchant_text,
                "normalized_merchant_text": normalize_merchant_text(merchant_text),
                "family_type": "base",
                "family_source": "test_fixture",
                "variant_source": "test_fixture",
            }
        ],
        columns=DATASET_COLUMNS,
    )


def test_normalize_merchant_text_applies_nfkc_whitespace_and_casefold() -> None:
    raw_text = "  STARBUCKS\r\n 강남　R점  "

    assert normalize_merchant_text(raw_text) == "starbucks 강남 r점"


def test_augment_dataset_splits_merges_merchant_info_into_train_only() -> None:
    train_df = _base_split("train", "S0001", 0, "식비", "기존한식당")
    valid_df = _base_split("valid", "S0002", 1, "카페", "기존카페")
    test_df = _base_split("test", "S0003", 0, "식비", "기존분식집")

    tmp_root = Path("tests/.tmp/merchant_info_augmentation")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    merchant_info_dir = tmp_root / "merchant_info"
    merchant_info_dir.mkdir(parents=True)
    merchant_info_path = merchant_info_dir / "소상공인시장진흥공단_상가(상권)정보_서울_202512.csv"

    merchant_rows = []
    for idx in range(5):
        merchant_rows.append({"상호명": f"카페상호{idx}", "상권업종소분류코드": "I21201"})
    for idx in range(5):
        merchant_rows.append({"상호명": f"식당상호{idx}", "상권업종소분류코드": "I20101"})
    merchant_rows.extend(
        [
            {"상호명": "카페상호0", "상권업종소분류코드": "I21201"},
            {"상호명": "기존한식당", "상권업종소분류코드": "I20101"},
            {"상호명": "미지원업종", "상권업종소분류코드": "S20701"},
        ]
    )
    pd.DataFrame(merchant_rows).to_csv(merchant_info_path, index=False, encoding="utf-8-sig")

    mapping_path = merchant_info_dir / "소분류코드_소비카테고리_매핑.csv"
    pd.DataFrame(
        [
            {"상권업종소분류코드": "I21201", "소비카테고리": "카페"},
            {"상권업종소분류코드": "I20101", "소비카테고리": "식비"},
            {"상권업종소분류코드": "S20701", "소비카테고리": "미용"},
        ]
    ).to_csv(mapping_path, index=False, encoding="utf-8-sig")

    output_dir = tmp_root / "augmented_outputs"

    result = augment_dataset_splits(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        merchant_info_dir=merchant_info_dir,
        mapping_path=mapping_path,
        output_dir=output_dir,
        seed=42,
    )

    augmented_all_df = result["augmented_all_df"]
    augmented_train_df = result["augmented_train_df"]
    augmented_valid_df = result["augmented_valid_df"]
    augmented_test_df = result["augmented_test_df"]

    assert len(augmented_all_df) == 10
    assert len(augmented_train_df) == 10
    assert augmented_valid_df.empty
    assert augmented_test_df.empty

    assert set(augmented_all_df["category_name"]) == {"식비", "카페"}
    assert "기존한식당" not in set(augmented_all_df["merchant_text"])
    assert augmented_all_df["merchant_text"].tolist().count("카페상호0") == 1
    assert augmented_all_df["normalized_merchant_text"].str.len().min() > 0
    assert augmented_all_df["merchant_family_id"].nunique() == len(augmented_all_df)

    assert len(result["train_df"]) == len(train_df) + 10
    assert len(result["valid_df"]) == len(valid_df)
    assert len(result["test_df"]) == len(test_df)

    assert (output_dir / "augmented_all.csv").exists()
    assert (output_dir / "augmented_train.csv").exists()
    assert (output_dir / "augmented_valid.csv").exists()
    assert (output_dir / "augmented_test.csv").exists()

    summary = result["summary"]
    assert summary["base_counts"] == {"train": 1, "valid": 1, "test": 1}
    assert summary["augmented_counts"] == {"all": 10, "train": 10, "valid": 0, "test": 0}
    assert summary["merged_counts"] == {"train": 11, "valid": 1, "test": 1}

    shutil.rmtree(tmp_root)


def test_augment_dataset_splits_can_merge_beauty_rows_into_valid_and_test() -> None:
    train_df = pd.concat(
        [
            _base_split("train", "S1001", 9, "쇼핑/패션", "기존쇼핑"),
            _base_split("train", "S1004", 6, "미용", "기존미용"),
        ],
        ignore_index=True,
    )
    valid_df = _base_split("valid", "S1002", 13, "카페", "기존카페")
    test_df = _base_split("test", "S1003", 10, "식비", "기존식당")

    tmp_root = Path("tests/.tmp/merchant_info_augmentation_beauty_eval")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    merchant_info_dir = tmp_root / "merchant_info"
    merchant_info_dir.mkdir(parents=True)
    merchant_info_path = merchant_info_dir / "소상공인시장진흥공단_상가(상권)정보_서울_202512.csv"

    merchant_rows = []
    for idx in range(10):
        merchant_rows.append({"상호명": f"뷰티상호{idx}", "상권업종소분류코드": "S20701"})
    for idx in range(5):
        merchant_rows.append({"상호명": f"카페상호{idx}", "상권업종소분류코드": "I21201"})
    pd.DataFrame(merchant_rows).to_csv(merchant_info_path, index=False, encoding="utf-8-sig")

    mapping_path = merchant_info_dir / "소분류코드_소비카테고리_매핑.csv"
    pd.DataFrame(
        [
            {"상권업종소분류코드": "S20701", "소비카테고리": "미용"},
            {"상권업종소분류코드": "I21201", "소비카테고리": "카페"},
        ]
    ).to_csv(mapping_path, index=False, encoding="utf-8-sig")

    result = augment_dataset_splits(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        merchant_info_dir=merchant_info_dir,
        mapping_path=mapping_path,
        output_dir=tmp_root / "outputs",
        seed=42,
        eval_augmented_categories=("미용",),
        eval_split_ratio=(0.8, 0.1, 0.1),
    )

    assert len(result["augmented_all_df"]) == 15
    assert len(result["augmented_train_df"]) == 13
    assert len(result["augmented_valid_df"]) == 1
    assert len(result["augmented_test_df"]) == 1

    assert set(result["augmented_valid_df"]["category_name"]) == {"미용"}
    assert set(result["augmented_test_df"]["category_name"]) == {"미용"}

    train_counts = result["augmented_train_df"]["category_name"].value_counts().to_dict()
    assert train_counts == {"미용": 8, "카페": 5}

    assert len(result["train_df"]) == 15
    assert len(result["valid_df"]) == 2
    assert len(result["test_df"]) == 2

    summary = result["summary"]
    assert summary["base_counts"] == {"train": 2, "valid": 1, "test": 1}
    assert summary["augmented_counts"] == {"all": 15, "train": 13, "valid": 1, "test": 1}
    assert summary["merged_counts"] == {"train": 15, "valid": 2, "test": 2}

    shutil.rmtree(tmp_root)
