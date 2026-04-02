import json
from pathlib import Path


def _load_notebook_source(notebook_path: str) -> str:
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_training_notebook_has_no_kfold_flow() -> None:
    all_source = _load_notebook_source("lab/koelectra_merchant_training_kfold.ipynb")

    assert "GroupKFold" not in all_source
    assert "cv_df =" not in all_source
    assert "kfold/cv_metrics.csv" not in all_source
    assert "load_hf_cv_metrics" not in all_source
    assert "## GroupKFold Results" not in all_source
    assert "NUM_FOLDS" not in all_source
    assert "Final Training" in all_source


def test_training_notebook_has_no_early_stopping() -> None:
    all_source = _load_notebook_source("lab/koelectra_merchant_training_kfold.ipynb")

    assert "EarlyStoppingCallback" not in all_source
    assert "EARLY_STOPPING_PATIENCE" not in all_source
    assert "early_stopping_patience" not in all_source


def test_training_notebook_enables_beauty_eval_augmentation() -> None:
    all_source = _load_notebook_source("lab/koelectra_merchant_training_kfold.ipynb")

    assert "MERCHANT_INFO_EVAL_AUGMENTED_CATEGORIES" in all_source
    assert "MERCHANT_INFO_EVAL_SPLIT_RATIOS" in all_source
    assert "eval_augmented_categories=MERCHANT_INFO_EVAL_AUGMENTED_CATEGORIES" in all_source
    assert "eval_split_ratio=MERCHANT_INFO_EVAL_SPLIT_RATIOS" in all_source


def test_training_notebook_reloads_merchant_info_augmentation_module() -> None:
    all_source = _load_notebook_source("lab/koelectra_merchant_training_kfold.ipynb")

    assert "import importlib" in all_source
    assert "import merchant_info_augmentation" in all_source
    assert "merchant_info_augmentation = importlib.reload(merchant_info_augmentation)" in all_source
    assert "augment_dataset_splits = merchant_info_augmentation.augment_dataset_splits" in all_source


def test_training_notebook_preserves_korean_text_sections() -> None:
    all_source = _load_notebook_source("lab/koelectra_merchant_training_kfold.ipynb")

    assert "이 노트북은 업로드된 merchant category dataset을 사용해 다음을 수행합니다." in all_source
    assert "기본 전제:" in all_source
    assert "# 입력 텍스트 구성" in all_source
    assert "최종 모델은 다음 방식으로 학습합니다." in all_source
    assert "스타벅스 강남R점 [SEP] 스타벅스 강남R점" in all_source


def test_training_notebook_contains_baseline_vs_v3_comparison_flow() -> None:
    all_source = _load_notebook_source("lab/koelectra_merchant_training_kfold.ipynb")

    assert "MerchantAttentionPoolingClassifier" in all_source
    assert "NoiseAugmentedMerchantDataset" in all_source
    assert "RawMerchantOnlyDataset" in all_source
    assert "BASELINE_EXPERIMENT_NAME = 'baseline'" in all_source
    assert "kakao1513/merchant-consumption-category-discriminator-v3" in all_source
    assert "comparison_rows" in all_source
    assert "## 4) Baseline vs V3 Comparison" in all_source
    assert "baseline_results" in all_source
    assert "v2_results" not in all_source


def test_training_notebook_uses_raw_merchant_text_for_baseline_inputs() -> None:
    all_source = _load_notebook_source("lab/koelectra_merchant_training_kfold.ipynb")

    assert "return record['merchant_text']" in all_source
    assert "build_raw_service_input_record" in all_source
    assert "'normalized_input': False" in all_source
    assert "'normalized_input': True" in all_source


def test_training_notebook_preserves_korean_runtime_literals() -> None:
    all_source = _load_notebook_source("lab/koelectra_merchant_training_kfold.ipynb")

    assert "MERCHANT_INFO_MAPPING_PATH = MERCHANT_INFO_DIR / '소분류코드_소비카테고리_매핑.csv'" in all_source
    assert "MERCHANT_INFO_EVAL_AUGMENTED_CATEGORIES = ('미용',)" in all_source
    assert "SERVICE_FALLBACK_LABEL = '기타서비스'" in all_source
    assert "'카페'" in all_source
    assert "'쇼핑/패션'" in all_source
    assert "examples = [" in all_source
    assert "build_hf_model_card_ko" in all_source
    assert "Merchant Consumption Category Discriminator V3" in all_source
    assert "trust_remote_code=True" in all_source


def test_training_notebook_exports_training_curves_and_deltas() -> None:
    all_source = _load_notebook_source("lab/koelectra_merchant_training_kfold.ipynb")

    assert "trainer.state.log_history" in all_source
    assert "training_history.csv" in all_source
    assert "training_curves.png" in all_source
    assert "training_history_summary.csv" in all_source
    assert "## 2) Training Curves" in all_source
    assert "train_loss_delta" in all_source
    assert "eval_loss_delta" in all_source
    assert "eval_f1_macro_delta" in all_source
    assert "eval_accuracy_delta" in all_source


def test_training_notebook_uses_attention_score_dtype_for_mask_fill() -> None:
    all_source = _load_notebook_source("lab/koelectra_merchant_training_kfold.ipynb")

    assert all_source.count("torch.finfo(attention_scores.dtype).min") >= 2
    assert "torch.finfo(hidden_states.dtype).min" not in all_source


def test_frozen_baseline_training_notebook_contains_frozen_encoder_comparison_flow() -> None:
    all_source = _load_notebook_source("lab/koelectra_merchant_training_kfold_frozen_baseline.ipynb")

    assert "FROZEN_BASELINE_EXPERIMENT_NAME = 'frozen_baseline'" in all_source
    assert "FrozenMerchantLinearProbeClassifier" in all_source
    assert "for param in self.encoder.parameters():" in all_source
    assert "param.requires_grad = False" in all_source
    assert (
        "input_strategy': 'merchant_text_only'" in all_source
        or '"input_strategy": "merchant_text_only"' in all_source
        or "input_strategy='merchant_text_only'" in all_source
    )
    assert "## 4) Frozen Baseline vs V3 Comparison" in all_source
    assert "frozen baseline" in all_source.lower()
    assert "attention_pooling_with_noise_augmentation" in all_source
    assert "merchant_text [SEP] normalized_merchant_text" not in all_source
    assert "normalized_input': True" not in all_source


def test_frozen_baseline_training_notebook_has_no_literal_unicode_escape_korean_tokens() -> None:
    all_source = _load_notebook_source("lab/koelectra_merchant_training_kfold_frozen_baseline.ipynb")

    assert r"\uae30\ud0c0\uc11c\ube44\uc2a4" not in all_source
    assert r"\uc2a4\ud0c0\ubc85\uc2a4 \uac15\ub0a8R\uc810" not in all_source
    assert r"\ubbfc\uc81c\ucee4\ud53c(\uc8fd\uc804\uc810)" not in all_source
    assert r"\ucca0\uc218\uc815\uc721\uc810 2\ud638\uc810" not in all_source
