from __future__ import annotations

from datetime import date, datetime

import pytest

from app.domain.chatbot.repository.chatbot_repository import ChatbotRepository


class _FakeRunner:
    def __init__(self, responses: list[list[dict[str, object]]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def execute_select(
        self,
        sql: str,
        *,
        parameters: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append({"sql": sql, "parameters": parameters or {}})
        if not self._responses:
            return []
        return self._responses.pop(0)


def test_get_weekly_budget_status_returns_standard_contract() -> None:
    runner = _FakeRunner(
        responses=[
            [
                {
                    "id": 11,
                    "week_start_date": date(2026, 3, 23),
                    "week_end_date": date(2026, 3, 29),
                    "planned_income_amount": 1000000,
                    "planned_essential_amount": 400000,
                    "planned_saving_amount": 300000,
                    "rollover_in_amount": 50000,
                    "manual_adjustment_amount": -20000,
                    "final_free_amount": 230000,
                    "actual_spent_amount": 370000,
                    "status": "ACTIVE",
                }
            ]
        ]
    )
    repository = ChatbotRepository(runner=runner)

    result = repository.get_weekly_budget_status(
        user_id=7,
        base_date=date(2026, 3, 26),
    )

    assert set(result) == {"as_of", "window", "summary", "evidence", "warnings"}
    assert result["window"] == {
        "date_from": "2026-03-23",
        "date_to": "2026-03-29",
    }
    assert result["warnings"] == []
    assert result["summary"]["final_free_amount"] == 230000
    assert result["summary"]["actual_spent_amount"] == 370000
    assert result["evidence"][0]["kind"] == "budget_week"
    assert result["evidence"][0]["budget_week_id"] == 11
    assert result["evidence"][0]["final_free_amount"] == 230000
    assert runner.calls[0]["parameters"] == {
        "user_id": 7,
        "base_date": date(2026, 3, 26),
    }


def test_get_weekly_budget_status_includes_adjustment_and_rollover_evidence() -> None:
    runner = _FakeRunner(
        responses=[
            [
                {
                    "id": 11,
                    "week_start_date": date(2026, 3, 23),
                    "week_end_date": date(2026, 3, 29),
                    "planned_income_amount": 1000000,
                    "planned_essential_amount": 400000,
                    "planned_saving_amount": 300000,
                    "rollover_in_amount": 50000,
                    "manual_adjustment_amount": -20000,
                    "final_free_amount": 230000,
                    "actual_spent_amount": 370000,
                    "status": "ACTIVE",
                }
            ],
            [
                {
                    "id": 31,
                    "plan_id": 51,
                    "adjustment_amount": -20000,
                    "adjustment_type": "MANUAL",
                    "reason": "Midweek correction",
                    "created_at": date(2026, 3, 24),
                }
            ],
            [
                {
                    "id": 41,
                    "source_budget_week_id": 10,
                    "target_budget_week_id": 11,
                    "rollover_amount": 50000,
                    "rollover_step": "APPLIED",
                    "note": "Prior week carryover",
                    "status": "APPLIED",
                    "resolved_at": None,
                }
            ],
        ]
    )
    repository = ChatbotRepository(runner=runner)

    result = repository.get_weekly_budget_status(
        user_id=7,
        base_date=date(2026, 3, 26),
    )

    kinds = [item["kind"] for item in result["evidence"]]
    assert kinds == ["budget_week", "budget_adjustments", "budget_rollovers"]
    assert result["evidence"][1]["adjustment_count"] == 1
    assert result["evidence"][2]["rollover_count"] == 1
    assert result["evidence"][2]["items"][0]["resolved_at"] is None
    assert runner.calls[1]["parameters"] == {
        "user_id": 7,
        "budget_week_id": 11,
    }
    assert runner.calls[2]["parameters"] == {
        "user_id": 7,
        "budget_week_id": 11,
    }


def test_get_weekly_budget_status_warns_when_no_budget_week_exists() -> None:
    runner = _FakeRunner(responses=[[]])
    repository = ChatbotRepository(runner=runner)

    result = repository.get_weekly_budget_status(
        user_id=7,
        base_date=date(2026, 3, 26),
    )

    assert result["summary"] == {}
    assert result["evidence"] == []
    assert result["warnings"]


def test_get_goal_progress_overview_returns_plan_summaries_and_scopes_by_user() -> None:
    runner = _FakeRunner(
        responses=[
            [
                {
                    "id": 21,
                    "title": "Emergency fund",
                    "description": "Build a safety net",
                    "target_amount": 5000000,
                    "saved_amount": 1500000,
                    "start_date": date(2026, 1, 1),
                    "end_date": date(2026, 12, 31),
                    "priority": 3,
                    "status": "ACTIVE",
                    "auto_allocate_enabled": True,
                }
            ]
        ]
    )
    repository = ChatbotRepository(runner=runner)

    result = repository.get_goal_progress_overview(
        user_id=7,
        status_filter="ACTIVE",
        limit=5,
    )

    assert result["warnings"] == []
    assert result["summary"]["goal_count"] == 1
    assert result["summary"]["plans"][0]["title"] == "Emergency fund"
    assert result["summary"]["plans"][0]["saved_amount"] == 1500000
    assert runner.calls[0]["parameters"] == {
        "user_id": 7,
        "status_filter": "ACTIVE",
        "limit": 5,
    }


def test_get_goal_progress_overview_warns_when_no_plans_exist() -> None:
    runner = _FakeRunner(responses=[[]])
    repository = ChatbotRepository(runner=runner)

    result = repository.get_goal_progress_overview(
        user_id=7,
        limit=5,
    )

    assert result["summary"]["goal_count"] == 0
    assert result["warnings"] == ["No plans found for the requested filters."]
    assert runner.calls[0]["parameters"] == {
        "user_id": 7,
        "limit": 5,
    }


def test_get_goal_progress_overview_omits_status_filter_when_not_provided() -> None:
    runner = _FakeRunner(responses=[[]])
    repository = ChatbotRepository(runner=runner)

    repository.get_goal_progress_overview(
        user_id=7,
        status_filter=None,
        limit=5,
    )

    assert "status_filter" not in runner.calls[0]["parameters"]
    assert runner.calls[0]["parameters"] == {
        "user_id": 7,
        "limit": 5,
    }


@pytest.mark.parametrize("limit", [0, -1])
def test_get_goal_progress_overview_rejects_non_positive_limit(limit: int) -> None:
    runner = _FakeRunner(responses=[[]])
    repository = ChatbotRepository(runner=runner)

    with pytest.raises(ValueError):
        repository.get_goal_progress_overview(
            user_id=7,
            limit=limit,
        )


def test_get_spending_summary_returns_standard_contract_and_metadata_warning() -> None:
    runner = _FakeRunner(
        responses=[
            [
                {
                    "total_spent": 230000,
                    "transaction_count": 3,
                    "average_transaction_amount": 76666.67,
                    "largest_transaction_amount": 120000,
                    "largest_transaction_memo": "Team lunch",
                    "category_count": 2,
                }
            ],
            [],
        ]
    )
    repository = ChatbotRepository(runner=runner)

    result = repository.get_spending_summary(
        user_id=7,
        date_from=date(2026, 3, 1),
        date_to=date(2026, 3, 26),
    )

    assert set(result) == {"as_of", "window", "summary", "evidence", "warnings"}
    assert result["window"] == {
        "date_from": "2026-03-01",
        "date_to": "2026-03-26",
    }
    assert result["summary"]["total_spent"] == 230000
    assert result["summary"]["transaction_count"] == 3
    assert result["evidence"][0]["kind"] == "spending_summary"
    assert result["evidence"][0]["total_spent"] == 230000
    assert result["warnings"] == [
        "No analysis metadata found for the requested user."
    ]
    assert runner.calls[0]["parameters"] == {
        "user_id": 7,
        "date_from": date(2026, 3, 1),
        "date_to": date(2026, 3, 26),
    }
    assert "account_transaction_override" in runner.calls[0]["sql"]
    assert "card_transaction_override" in runner.calls[0]["sql"]
    assert "users_id = :user_id" in runner.calls[0]["sql"]
    assert "transaction_analysis_metadata" in runner.calls[1]["sql"]


def test_get_spending_summary_uses_only_analyzed_spending_transactions() -> None:
    runner = _FakeRunner(
        responses=[
            [
                {
                    "total_spent": 230000,
                    "transaction_count": 3,
                    "average_transaction_amount": 76666.67,
                    "largest_transaction_amount": 120000,
                    "largest_transaction_memo": "Team lunch",
                    "category_count": 2,
                }
            ],
            [],
        ]
    )
    repository = ChatbotRepository(runner=runner)

    repository.get_spending_summary(
        user_id=7,
        date_from=date(2026, 3, 1),
        date_to=date(2026, 3, 26),
    )

    assert "transaction_analysis_result" in runner.calls[0]["sql"]
    assert "analysis_included = true" in runner.calls[0]["sql"]
    assert "expense_scope IS NOT NULL" in runner.calls[0]["sql"]


def test_get_spending_by_category_returns_ranked_categories_and_comparison() -> None:
    runner = _FakeRunner(
        responses=[
            [
                {
                    "category_id": 1,
                    "category_name": "식비",
                    "spent_amount": 150000,
                    "transaction_count": 4,
                },
                {
                    "category_id": 2,
                    "category_name": "교통",
                    "spent_amount": 80000,
                    "transaction_count": 6,
                },
            ],
            [
                {
                    "comparison_total_spent": 180000,
                }
            ],
            [],
        ]
    )
    repository = ChatbotRepository(runner=runner)

    result = repository.get_spending_by_category(
        user_id=7,
        date_from=date(2026, 3, 1),
        date_to=date(2026, 3, 26),
        include_comparison=True,
    )

    assert result["summary"]["total_spent"] == 230000
    assert result["summary"]["category_count"] == 2
    assert result["summary"]["comparison_total_spent"] == 180000
    assert result["evidence"][0]["category_name"] == "식비"
    assert result["evidence"][1]["category_name"] == "교통"
    assert result["warnings"] == [
        "No analysis metadata found for the requested user."
    ]
    assert runner.calls[0]["parameters"] == {
        "user_id": 7,
        "date_from": date(2026, 3, 1),
        "date_to": date(2026, 3, 26),
    }
    assert runner.calls[1]["parameters"] == {
        "user_id": 7,
        "date_from": date(2026, 2, 3),
        "date_to": date(2026, 2, 28),
    }
    assert "category" in runner.calls[0]["sql"]
    assert "transaction_analysis_metadata" in runner.calls[2]["sql"]


def test_get_spending_by_category_uses_analysis_result_category() -> None:
    runner = _FakeRunner(
        responses=[
            [
                {
                    "category_id": 1,
                    "category_name": "식비",
                    "spent_amount": 150000,
                    "transaction_count": 4,
                }
            ],
            [],
        ]
    )
    repository = ChatbotRepository(runner=runner)

    repository.get_spending_by_category(
        user_id=7,
        date_from=date(2026, 3, 1),
        date_to=date(2026, 3, 26),
        include_comparison=False,
    )

    assert "transaction_analysis_result" in runner.calls[0]["sql"]
    assert "analysis_included = true" in runner.calls[0]["sql"]
    assert "expense_scope IS NOT NULL" in runner.calls[0]["sql"]
    assert "cat.id = ar.category_id" in runner.calls[0]["sql"]


def test_get_spending_by_category_omits_comparison_when_disabled() -> None:
    runner = _FakeRunner(
        responses=[
            [
                {
                    "category_id": 1,
                    "category_name": "식비",
                    "spent_amount": 150000,
                    "transaction_count": 4,
                }
            ],
            [],
        ]
    )
    repository = ChatbotRepository(runner=runner)

    result = repository.get_spending_by_category(
        user_id=7,
        date_from=date(2026, 3, 1),
        date_to=date(2026, 3, 26),
        include_comparison=False,
    )

    assert result["summary"]["comparison_total_spent"] is None
    assert len(runner.calls) == 2
    assert runner.calls[1]["sql"].strip().lower().startswith("select")


def test_get_recent_spending_anomalies_returns_recent_candidates_and_metadata_warning() -> None:
    runner = _FakeRunner(
        responses=[
            [
                {
                    "transaction_id": 101,
                    "category_name": "식비",
                    "transaction_balance": 125000,
                    "transaction_date_time": datetime(2026, 3, 25, 12, 0),
                    "transaction_memo": "Team lunch",
                    "final_label": "ANOMALY",
                    "raw_label": "SPENDING",
                    "confidence": 0.41,
                    "threshold_used": 0.7,
                    "analysis_included": True,
                    "excluded_reason": None,
                }
            ],
            [],
        ]
    )
    repository = ChatbotRepository(runner=runner)

    result = repository.get_recent_spending_anomalies(
        user_id=7,
        lookback_days=14,
        limit=3,
    )

    assert result["summary"]["anomaly_count"] == 1
    assert result["summary"]["lookback_days"] == 14
    assert result["summary"]["limit"] == 3
    assert result["evidence"][0]["kind"] == "spending_anomaly"
    assert result["evidence"][0]["transaction_id"] == 101
    assert result["warnings"] == [
        "No analysis metadata found for the requested user."
    ]
    assert runner.calls[0]["parameters"]["user_id"] == 7
    assert runner.calls[0]["parameters"]["limit"] == 3
    assert "transaction_analysis_result" in runner.calls[0]["sql"]
    assert "transaction_analysis_metadata" in runner.calls[1]["sql"]
