import json
from pathlib import Path

import httpx

from lab.finapi import merchant_registration


def test_fetch_registered_merchant_names_uses_inquire_merchant_list_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL(merchant_registration.INQUIRE_MERCHANT_LIST_URL)

        payload = json.loads(request.content.decode("utf-8"))
        assert payload["Header"]["apiName"] == "inquireMerchantList"
        assert payload["Header"]["apiServiceCode"] == "inquireMerchantList"

        return httpx.Response(
            200,
            json={
                "Header": {
                    "responseCode": "H0000",
                    "responseMessage": "정상처리 되었습니다.",
                },
                "REC": [
                    {"merchantName": "스타벅스"},
                    {"merchantName": " 코스트코 "},
                    {"merchantName": "스타벅스"},
                    {"merchantName": ""},
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        merchant_names = merchant_registration.fetch_registered_merchant_names(
            api_key="test-api-key",
            client=client,
        )

    assert merchant_names == {"스타벅스", "코스트코"}


def test_exclude_existing_merchant_names_skips_registered_names_and_duplicates() -> None:
    pending_names, skipped_names = merchant_registration.exclude_existing_merchant_names(
        merchant_names=["새가게", "스타벅스", " 둘째가게 ", "새가게", "코스트코"],
        existing_merchant_names={"스타벅스", "코스트코"},
    )

    assert pending_names == ["새가게", "둘째가게"]
    assert skipped_names == ["스타벅스", "코스트코"]


def test_register_merchants_reports_progress_for_each_processed_name() -> None:
    observed_progress: list[tuple[int, int, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        merchant_name = payload["merchantName"]
        return httpx.Response(
            200,
            json={
                "Header": {
                    "responseCode": "H0000",
                    "responseMessage": "정상처리 되었습니다.",
                },
                "REC": [
                    {
                        "merchantId": len(observed_progress) + 1,
                        "merchantName": merchant_name,
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        success_rows, failure_rows = merchant_registration.register_merchants(
            merchant_names=["첫째가게", "둘째가게"],
            api_key="test-api-key",
            client=client,
            progress_callback=lambda completed, total, merchant_name: observed_progress.append(
                (completed, total, merchant_name)
            ),
        )

    assert failure_rows == []
    assert success_rows == [
        {"merchantId": "1", "merchantName": "첫째가게"},
        {"merchantId": "2", "merchantName": "둘째가게"},
    ]
    assert observed_progress == [
        (1, 2, "첫째가게"),
        (2, 2, "둘째가게"),
    ]


def test_register_merchants_notebook_uses_existing_list_dedup_and_tqdm() -> None:
    notebook_path = Path("lab/register_merchants.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "from tqdm.auto import tqdm" in all_source
    assert "fetch_registered_merchant_names" in all_source
    assert "exclude_existing_merchant_names" in all_source
    assert "with tqdm(" in all_source
    assert "progress_callback=update_registration_progress" in all_source


def test_register_merchants_notebook_preserves_korean_text() -> None:
    notebook_path = Path("lab/register_merchants.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "\uc0c1\uac00 \uc815\ubcf4 CSV" in all_source
    assert "_\ub300\uad6c_" in all_source
    assert "_\uacbd\ubd81_" in all_source
    assert "\ud658\uacbd \ubcc0\uc218\uac00 \ube44\uc5b4 \uc788\uc2b5\ub2c8\ub2e4." in all_source


def test_register_merchants_notebook_reloads_local_module() -> None:
    notebook_path = Path("lab/register_merchants.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "import importlib" in all_source
    assert "from lab.finapi import merchant_registration" in all_source
    assert "merchant_registration = importlib.reload(merchant_registration)" in all_source
