from __future__ import annotations

import torch

from app.domain.merchant_classification.service.merchant_classifier_service import (
    MerchantClassifierService,
)


class _FakeBatchEncoding(dict):
    def to(self, device: str) -> "_FakeBatchEncoding":
        return self


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        input_texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        return_tensors: str,
    ) -> _FakeBatchEncoding:
        assert padding is True
        assert truncation is True
        assert return_tensors == "pt"
        self.calls.append(list(input_texts))

        token_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        max_length = max(len(text) for text in input_texts)

        for text in input_texts:
            tokens = [ord(char) for char in text]
            padding_width = max_length - len(tokens)
            token_rows.append(tokens + ([0] * padding_width))
            attention_rows.append(([1] * len(tokens)) + ([0] * padding_width))

        return _FakeBatchEncoding(
            input_ids=torch.tensor(token_rows, dtype=torch.long),
            attention_mask=torch.tensor(attention_rows, dtype=torch.long),
        )


class _FakeModelConfig:
    id2label = {
        0: "CAFE",
        1: "MEAL",
    }


class _FakeModel:
    config = _FakeModelConfig()

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        logits: list[list[float]] = []
        for row, mask in zip(input_ids, attention_mask, strict=True):
            text = "".join(
                chr(token)
                for token, included in zip(row.tolist(), mask.tolist(), strict=True)
                if included
            )
            if "STARBUCKS" in text.upper():
                logits.append([8.0, 1.0])
            else:
                logits.append([1.0, 8.0])
        return type(
            "FakeOutput",
            (),
            {"logits": torch.tensor(logits, dtype=torch.float32)},
        )()


def _build_service(*, inference_batch_size: int = 128) -> MerchantClassifierService:
    return MerchantClassifierService(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
        fallback_label="FALLBACK",
        fallback_threshold=0.5,
        inference_batch_size=inference_batch_size,
    )


def test_predict_many_uses_cleaned_raw_merchant_text_for_model_input() -> None:
    tokenizer = _FakeTokenizer()
    service = MerchantClassifierService(
        tokenizer=tokenizer,
        model=_FakeModel(),
        fallback_label="FALLBACK",
        fallback_threshold=0.5,
    )

    service.predict_many(["  Starbucks\nGangnam  "])

    assert tokenizer.calls == [["Starbucks Gangnam"]]


def test_predict_many_matches_single_prediction_for_same_merchant_text() -> None:
    service = _build_service()

    single_result = service.predict_one("Starbucks Gangnam")
    batch_result = service.predict_many(
        [
            "A very long merchant name that forces extra batch padding",
            "Starbucks Gangnam",
        ]
    )[1]

    assert batch_result == single_result


def test_predict_many_preserves_single_prediction_across_internal_chunks() -> None:
    service = _build_service(inference_batch_size=2)

    single_result = service.predict_one("Starbucks Gangnam")
    batch_result = service.predict_many(
        [
            "Alpha Store",
            "Beta Restaurant",
            "Starbucks Gangnam",
        ]
    )[2]

    assert batch_result == single_result
