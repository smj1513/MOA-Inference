from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


class _FakeClassifier:
    def __init__(self) -> None:
        self.predict_one_calls: list[str] = []
        self.predict_many_calls: list[list[str]] = []

    def predict_one(self, merchant_text: str) -> dict[str, object]:
        self.predict_one_calls.append(merchant_text)
        return {
            "merchant_text": merchant_text,
            "normalized_merchant_text": merchant_text.strip().lower(),
            "raw_label": "RAW",
            "final_label": "FINAL",
            "confidence": 0.97,
            "fallback_applied": False,
            "threshold_used": 0.5,
        }

    def predict_many(self, merchant_texts: list[str]) -> list[dict[str, object]]:
        self.predict_many_calls.append(merchant_texts)
        return [self.predict_one(merchant_text) for merchant_text in merchant_texts]


def test_merchant_inference_endpoint_moves_to_v1_prefix() -> None:
    classifier = _FakeClassifier()
    app = create_app(
        classifier=classifier,
        image_generation_service=object(),
        chatbot_service=object(),
    )

    with TestClient(app) as client:
        response = client.post("/v1/merchant", json={"merchant_text": "  Test Shop  "})
        legacy_response = client.post(
            "/v1/inference/merchant",
            json={"merchant_text": "  Test Shop  "},
        )

    assert response.status_code == 200
    assert response.json() == {
        "merchant_text": "  Test Shop  ",
        "normalized_merchant_text": "test shop",
        "raw_label": "RAW",
        "final_label": "FINAL",
        "confidence": 0.97,
        "fallback_applied": False,
        "threshold_used": 0.5,
    }
    assert legacy_response.status_code == 404
    assert classifier.predict_one_calls == ["  Test Shop  "]


def test_merchant_batch_inference_endpoint_moves_to_v1_prefix() -> None:
    classifier = _FakeClassifier()
    app = create_app(
        classifier=classifier,
        image_generation_service=object(),
        chatbot_service=object(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/merchant/batch",
            json={"merchant_texts": ["First Shop", "Second Shop"]},
        )
        legacy_response = client.post(
            "/v1/inference/merchant/batch",
            json={"merchant_texts": ["First Shop", "Second Shop"]},
        )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "merchant_text": "First Shop",
                "normalized_merchant_text": "first shop",
                "raw_label": "RAW",
                "final_label": "FINAL",
                "confidence": 0.97,
                "fallback_applied": False,
                "threshold_used": 0.5,
            },
            {
                "merchant_text": "Second Shop",
                "normalized_merchant_text": "second shop",
                "raw_label": "RAW",
                "final_label": "FINAL",
                "confidence": 0.97,
                "fallback_applied": False,
                "threshold_used": 0.5,
            },
        ]
    }
    assert legacy_response.status_code == 404
    assert classifier.predict_many_calls == [["First Shop", "Second Shop"]]
