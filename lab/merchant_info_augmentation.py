from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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

MERCHANT_INFO_FILE_GLOB = "소상공인시장진흥공단_상가(상권)정보_*.csv"
MERCHANT_INFO_USECOLS = ["상호명", "상권업종소분류코드"]
MAPPING_USECOLS = ["상권업종소분류코드", "소비카테고리"]


def normalize_merchant_text(merchant_text: object) -> str:
    text = unicodedata.normalize("NFKC", str(merchant_text or ""))
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def _clean_merchant_text(merchant_text: object) -> str:
    text = unicodedata.normalize("NFKC", str(merchant_text or ""))
    text = text.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _empty_dataset_df() -> pd.DataFrame:
    return pd.DataFrame(columns=DATASET_COLUMNS)


def _normalize_category_names(category_names: Iterable[str] | None) -> tuple[str, ...]:
    if category_names is None:
        return ()
    normalized = []
    for category_name in category_names:
        cleaned = str(category_name or "").strip()
        if cleaned:
            normalized.append(cleaned)
    return tuple(dict.fromkeys(normalized))


def _compute_split_sizes(row_count: int, split_ratio: tuple[float, float, float]) -> tuple[int, int, int]:
    if row_count <= 0:
        return (0, 0, 0)

    train_ratio, valid_ratio, test_ratio = split_ratio
    split_sizes = [
        int(np.floor(row_count * train_ratio)),
        int(np.floor(row_count * valid_ratio)),
        int(np.floor(row_count * test_ratio)),
    ]
    remainder = row_count - sum(split_sizes)
    ratio_order = np.argsort(np.array(split_ratio))[::-1]
    for offset in range(remainder):
        split_sizes[int(ratio_order[offset % len(ratio_order)])] += 1
    return tuple(int(size) for size in split_sizes)


def load_mapping_df(mapping_path: Path) -> pd.DataFrame:
    mapping_df = pd.read_csv(mapping_path, encoding="utf-8-sig", usecols=MAPPING_USECOLS)
    mapping_df = mapping_df.dropna(subset=MAPPING_USECOLS).copy()
    mapping_df["상권업종소분류코드"] = mapping_df["상권업종소분류코드"].astype(str).str.strip()
    mapping_df["소비카테고리"] = mapping_df["소비카테고리"].astype(str).str.strip()
    mapping_df = mapping_df[
        (mapping_df["상권업종소분류코드"] != "") & (mapping_df["소비카테고리"] != "")
    ]
    return mapping_df.drop_duplicates(subset=["상권업종소분류코드"]).reset_index(drop=True)


def load_merchant_info_df(merchant_info_dir: Path) -> pd.DataFrame:
    merchant_info_dir = Path(merchant_info_dir)
    csv_paths = sorted(merchant_info_dir.glob(MERCHANT_INFO_FILE_GLOB))
    if not csv_paths:
        raise FileNotFoundError(
            f"merchant info CSV files were not found under {merchant_info_dir}"
        )

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=MERCHANT_INFO_USECOLS)
        frame["source_file"] = csv_path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def build_label_lookup(base_splits: Iterable[pd.DataFrame]) -> pd.DataFrame:
    label_df = pd.concat(
        [df[["category_id", "category_name"]] for df in base_splits],
        ignore_index=True,
    )
    return label_df.drop_duplicates().sort_values("category_id").reset_index(drop=True)


def _build_existing_pairs(existing_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        return pd.DataFrame(columns=["normalized_merchant_text", "category_name"])

    pairs_df = existing_df.copy()
    if "normalized_merchant_text" not in pairs_df.columns:
        pairs_df["normalized_merchant_text"] = pairs_df["merchant_text"].map(normalize_merchant_text)

    pairs_df["normalized_merchant_text"] = pairs_df["normalized_merchant_text"].map(
        normalize_merchant_text
    )
    pairs_df["category_name"] = pairs_df["category_name"].astype(str).str.strip()
    return pairs_df[["normalized_merchant_text", "category_name"]].drop_duplicates().reset_index(
        drop=True
    )


def build_augmented_candidates(
    merchant_info_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    label_lookup_df: pd.DataFrame,
    existing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    candidates_df = merchant_info_df.copy()
    candidates_df["merchant_text"] = candidates_df["상호명"].map(_clean_merchant_text)
    candidates_df["상권업종소분류코드"] = candidates_df["상권업종소분류코드"].astype(str).str.strip()
    candidates_df = candidates_df[
        (candidates_df["merchant_text"] != "") & (candidates_df["상권업종소분류코드"] != "")
    ].copy()

    candidates_df = candidates_df.merge(mapping_df, on="상권업종소분류코드", how="left")
    candidates_df = candidates_df.rename(columns={"소비카테고리": "category_name"})
    candidates_df = candidates_df.dropna(subset=["category_name"]).copy()
    candidates_df["category_name"] = candidates_df["category_name"].astype(str).str.strip()

    candidates_df = candidates_df.merge(label_lookup_df, on="category_name", how="inner")
    candidates_df["normalized_merchant_text"] = candidates_df["merchant_text"].map(
        normalize_merchant_text
    )
    candidates_df = candidates_df[candidates_df["normalized_merchant_text"] != ""].copy()
    candidates_df = candidates_df.drop_duplicates(
        subset=["normalized_merchant_text", "category_name"]
    ).reset_index(drop=True)

    if existing_df is not None:
        existing_pairs_df = _build_existing_pairs(existing_df)
        candidates_df = candidates_df.merge(
            existing_pairs_df,
            on=["normalized_merchant_text", "category_name"],
            how="left",
            indicator=True,
        )
        candidates_df = candidates_df[candidates_df["_merge"] == "left_only"].drop(
            columns=["_merge"]
        )

    candidates_df = candidates_df.reset_index(drop=True)
    row_count = len(candidates_df)
    candidates_df["sample_id"] = [f"AUGS{idx:07d}" for idx in range(1, row_count + 1)]
    candidates_df["merchant_family_id"] = [f"AUGF{idx:07d}" for idx in range(1, row_count + 1)]
    candidates_df["merchant_family"] = candidates_df["merchant_text"]
    candidates_df["family_type"] = "merchant_info"
    candidates_df["family_source"] = "smallbiz_market_mapping"
    candidates_df["variant_source"] = "merchant_info_official_name"
    candidates_df["split"] = ""
    return candidates_df[DATASET_COLUMNS].reset_index(drop=True)


def assign_augmented_split(
    augmented_df: pd.DataFrame,
    seed: int = 42,
    eval_augmented_categories: Iterable[str] | None = None,
    eval_split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, pd.DataFrame]:
    if augmented_df.empty:
        empty_df = _empty_dataset_df()
        return {"train": empty_df.copy(), "valid": empty_df.copy(), "test": empty_df.copy()}

    normalized_eval_categories = set(_normalize_category_names(eval_augmented_categories))
    if not normalized_eval_categories:
        rng = np.random.default_rng(seed)
        shuffled_df = augmented_df.iloc[rng.permutation(len(augmented_df))].reset_index(drop=True).copy()
        shuffled_df["split"] = "train"

        empty_df = _empty_dataset_df()
        return {
            "train": shuffled_df[DATASET_COLUMNS].reset_index(drop=True),
            "valid": empty_df.copy(),
            "test": empty_df.copy(),
        }

    if len(eval_split_ratio) != 3:
        raise ValueError("eval_split_ratio must contain exactly three values")

    ratio_sum = float(sum(eval_split_ratio))
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("eval_split_ratio must sum to 1.0")

    rng = np.random.default_rng(seed)
    train_frames: list[pd.DataFrame] = []
    valid_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    default_train_df = augmented_df[
        ~augmented_df["category_name"].isin(normalized_eval_categories)
    ].copy()
    if not default_train_df.empty:
        default_train_df = default_train_df.iloc[rng.permutation(len(default_train_df))].reset_index(drop=True)
        default_train_df["split"] = "train"
        train_frames.append(default_train_df[DATASET_COLUMNS])

    for category_name in sorted(normalized_eval_categories):
        category_df = augmented_df[augmented_df["category_name"] == category_name].copy()
        if category_df.empty:
            continue

        category_df = category_df.iloc[rng.permutation(len(category_df))].reset_index(drop=True)
        train_size, valid_size, test_size = _compute_split_sizes(len(category_df), eval_split_ratio)

        train_end = train_size
        valid_end = train_size + valid_size

        category_train_df = category_df.iloc[:train_end].copy()
        category_valid_df = category_df.iloc[train_end:valid_end].copy()
        category_test_df = category_df.iloc[valid_end:valid_end + test_size].copy()

        if not category_train_df.empty:
            category_train_df["split"] = "train"
            train_frames.append(category_train_df[DATASET_COLUMNS])
        if not category_valid_df.empty:
            category_valid_df["split"] = "valid"
            valid_frames.append(category_valid_df[DATASET_COLUMNS])
        if not category_test_df.empty:
            category_test_df["split"] = "test"
            test_frames.append(category_test_df[DATASET_COLUMNS])

    empty_df = _empty_dataset_df()
    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else empty_df.copy()
    valid_df = pd.concat(valid_frames, ignore_index=True) if valid_frames else empty_df.copy()
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else empty_df.copy()
    return {
        "train": train_df.reset_index(drop=True),
        "valid": valid_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def save_augmented_splits(
    augmented_all_df: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "all": output_dir / "augmented_all.csv",
        "train": output_dir / "augmented_train.csv",
        "valid": output_dir / "augmented_valid.csv",
        "test": output_dir / "augmented_test.csv",
    }

    augmented_all_df.to_csv(output_paths["all"], index=False, encoding="utf-8-sig")
    split_frames["train"].to_csv(output_paths["train"], index=False, encoding="utf-8-sig")
    split_frames["valid"].to_csv(output_paths["valid"], index=False, encoding="utf-8-sig")
    split_frames["test"].to_csv(output_paths["test"], index=False, encoding="utf-8-sig")
    return {key: str(path) for key, path in output_paths.items()}


def augment_dataset_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    merchant_info_dir: Path,
    mapping_path: Path,
    output_dir: Path | None = None,
    seed: int = 42,
    eval_augmented_categories: Iterable[str] | None = None,
    eval_split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, object]:
    merchant_info_df = load_merchant_info_df(merchant_info_dir)
    mapping_df = load_mapping_df(mapping_path)
    label_lookup_df = build_label_lookup([train_df, valid_df, test_df])
    existing_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    augmented_all_df = build_augmented_candidates(
        merchant_info_df=merchant_info_df,
        mapping_df=mapping_df,
        label_lookup_df=label_lookup_df,
        existing_df=existing_df,
    )
    split_frames = assign_augmented_split(
        augmented_df=augmented_all_df,
        seed=seed,
        eval_augmented_categories=eval_augmented_categories,
        eval_split_ratio=eval_split_ratio,
    )

    merged_train_df = pd.concat([train_df.copy(), split_frames["train"]], ignore_index=True)
    merged_valid_df = pd.concat([valid_df.copy(), split_frames["valid"]], ignore_index=True).reset_index(drop=True)
    merged_test_df = pd.concat([test_df.copy(), split_frames["test"]], ignore_index=True).reset_index(drop=True)

    output_paths: dict[str, str] = {}
    if output_dir is not None:
        output_paths = save_augmented_splits(
            augmented_all_df=augmented_all_df,
            split_frames=split_frames,
            output_dir=output_dir,
        )

    summary = {
        "base_counts": {
            "train": len(train_df),
            "valid": len(valid_df),
            "test": len(test_df),
        },
        "augmented_counts": {
            "all": len(augmented_all_df),
            "train": len(split_frames["train"]),
            "valid": len(split_frames["valid"]),
            "test": len(split_frames["test"]),
        },
        "merged_counts": {
            "train": len(merged_train_df),
            "valid": len(merged_valid_df),
            "test": len(merged_test_df),
        },
    }

    return {
        "train_df": merged_train_df,
        "valid_df": merged_valid_df,
        "test_df": merged_test_df,
        "augmented_all_df": augmented_all_df,
        "augmented_train_df": split_frames["train"],
        "augmented_valid_df": split_frames["valid"],
        "augmented_test_df": split_frames["test"],
        "output_paths": output_paths,
        "summary": summary,
    }
