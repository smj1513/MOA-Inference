from __future__ import annotations

from types import SimpleNamespace

from app.core.config import Settings
from app.core import model_loader


class _FakeLoadedModel:
    def __init__(
        self,
        *,
        model_type: str = "electra",
        architectures: list[str] | None = None,
        pooling_strategy: str | None = None,
    ) -> None:
        self.config = SimpleNamespace(
            model_type=model_type,
            architectures=architectures or [],
            pooling_strategy=pooling_strategy,
        )
        self.device: str | None = None
        self.eval_called = False

    def to(self, device: str) -> "_FakeLoadedModel":
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True


def test_settings_default_model_id_points_to_v3_repo() -> None:
    settings = Settings(_env_file=None)

    assert settings.model_id == "kakao1513/merchant-consumption-category-discriminator-v3"
    assert settings.fallback_threshold == 0.5


def test_validate_loaded_sequence_classifier_accepts_attention_pooling_v3_model() -> None:
    model = _FakeLoadedModel(
        architectures=["MerchantAttentionPoolingClassifier"],
        pooling_strategy="token_attention_pooling",
    )

    model_loader._validate_loaded_sequence_classifier(model, "repo")


def test_load_sequence_classifier_uses_trust_remote_code(monkeypatch) -> None:
    tokenizer_calls: list[tuple[str, dict[str, object]]] = []
    model_calls: list[tuple[str, dict[str, object]]] = []
    fake_tokenizer = object()
    fake_model = _FakeLoadedModel(
        architectures=["MerchantAttentionPoolingClassifier"],
        pooling_strategy="token_attention_pooling",
    )

    def fake_tokenizer_from_pretrained(
        model_source: str,
        **kwargs: object,
    ) -> object:
        tokenizer_calls.append((model_source, kwargs))
        return fake_tokenizer

    def fake_model_from_pretrained(
        model_source: str,
        **kwargs: object,
    ) -> _FakeLoadedModel:
        model_calls.append((model_source, kwargs))
        return fake_model

    monkeypatch.setattr(
        model_loader,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=fake_tokenizer_from_pretrained),
    )
    monkeypatch.setattr(
        model_loader,
        "AutoModelForSequenceClassification",
        SimpleNamespace(from_pretrained=fake_model_from_pretrained),
    )

    settings = Settings(
        _env_file=None,
        MODEL_ID="kakao1513/merchant-consumption-category-discriminator-v3",
    )

    artifacts = model_loader.load_sequence_classifier(settings)

    assert artifacts == {
        "tokenizer": fake_tokenizer,
        "model": fake_model,
    }
    assert tokenizer_calls == [
        (
            "kakao1513/merchant-consumption-category-discriminator-v3",
            {
                "revision": None,
                "token": None,
                "trust_remote_code": True,
            },
        )
    ]
    assert model_calls == [
        (
            "kakao1513/merchant-consumption-category-discriminator-v3",
            {
                "revision": None,
                "token": None,
                "trust_remote_code": True,
            },
        )
    ]
    assert fake_model.device == "cpu"
    assert fake_model.eval_called is True
