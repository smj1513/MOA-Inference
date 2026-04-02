from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.core.database import ReadOnlyDatabase, validate_user_id


def _as_iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _as_iso_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).isoformat()
    return str(value)


def _build_standard_contract(
    *,
    as_of: str,
    window: dict[str, str | None],
    summary: dict[str, Any],
    evidence: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "as_of": as_of,
        "window": window,
        "summary": summary,
        "evidence": evidence,
        "warnings": warnings,
    }


def _build_budget_week_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "budget_week",
        "budget_week_id": row.get("id"),
        "week_start_date": _as_iso_date(row.get("week_start_date")),
        "week_end_date": _as_iso_date(row.get("week_end_date")),
        "planned_income_amount": row.get("planned_income_amount"),
        "planned_essential_amount": row.get("planned_essential_amount"),
        "planned_saving_amount": row.get("planned_saving_amount"),
        "rollover_in_amount": row.get("rollover_in_amount"),
        "manual_adjustment_amount": row.get("manual_adjustment_amount"),
        "final_free_amount": row.get("final_free_amount"),
        "actual_spent_amount": row.get("actual_spent_amount"),
        "status": row.get("status"),
    }


def _build_adjustment_evidence(
    *,
    budget_week_id: Any,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "kind": "budget_adjustments",
        "budget_week_id": budget_week_id,
        "adjustment_count": len(rows),
        "total_adjustment_amount": sum((row.get("adjustment_amount") or 0) for row in rows),
        "items": [
            {
                "adjustment_id": row.get("id"),
                "plan_id": row.get("plan_id"),
                "adjustment_amount": row.get("adjustment_amount"),
                "adjustment_type": row.get("adjustment_type"),
                "reason": row.get("reason"),
                "created_at": _as_iso_datetime(row.get("created_at")),
            }
            for row in rows
        ],
    }


def _build_rollover_evidence(
    *,
    budget_week_id: Any,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "kind": "budget_rollovers",
        "budget_week_id": budget_week_id,
        "rollover_count": len(rows),
        "total_rollover_amount": sum((row.get("rollover_amount") or 0) for row in rows),
        "items": [
            {
                "rollover_id": row.get("id"),
                "source_budget_week_id": row.get("source_budget_week_id"),
                "target_budget_week_id": row.get("target_budget_week_id"),
                "rollover_amount": row.get("rollover_amount"),
                "rollover_step": row.get("rollover_step"),
                "note": row.get("note"),
                "status": row.get("status"),
                "created_at": _as_iso_datetime(row.get("created_at")),
                "resolved_at": _as_iso_datetime(row.get("resolved_at")),
            }
            for row in rows
        ],
    }


def _build_user_scoped_transactions_cte() -> str:
    return """
    WITH scoped_transactions AS (
        SELECT DISTINCT
            scoped_base.id,
            scoped_base.category_id,
            scoped_base.transaction_balance,
            scoped_base.transaction_date_time,
            scoped_base.transaction_memo
        FROM (
            SELECT
                tovr.id,
                tovr.category_id,
                tovr.transaction_balance,
                tovr.transaction_date_time,
                tovr.transaction_memo
            FROM transaction_override tovr
            JOIN account_transaction_override ato
              ON ato.id = tovr.id
            JOIN account a
              ON a.id = ato.account_id
            WHERE a.users_id = :user_id
              AND tovr.is_deleted = false

            UNION ALL

            SELECT
                tovr.id,
                tovr.category_id,
                tovr.transaction_balance,
                tovr.transaction_date_time,
                tovr.transaction_memo
            FROM transaction_override tovr
            JOIN card_transaction_override cto
              ON cto.id = tovr.id
            JOIN card c
              ON c.card_unique_no = cto.card_unique_no
            JOIN account a
              ON a.id = c.withdrawal_account_no
            WHERE a.users_id = :user_id
              AND tovr.is_deleted = false
        ) scoped_base
    )
    """


def _build_transaction_scope_warning() -> str:
    return "No analysis metadata found for the requested user."


@dataclass(slots=True)
class ChatbotRepository:
    runner: Any

    @classmethod
    def from_database(cls, database: ReadOnlyDatabase) -> "ChatbotRepository":
        return cls(runner=database)

    def get_weekly_budget_status(
        self,
        *,
        user_id: int,
        base_date: date,
    ) -> dict[str, Any]:
        validate_user_id(user_id)
        rows = self.runner.execute_select(
            """
            SELECT
                id,
                week_start_date,
                week_end_date,
                planned_income_amount,
                planned_essential_amount,
                planned_saving_amount,
                rollover_in_amount,
                manual_adjustment_amount,
                final_free_amount,
                actual_spent_amount,
                status
            FROM budget_week
            WHERE users_id = :user_id
              AND is_deleted = false
              AND week_start_date <= :base_date
              AND week_end_date >= :base_date
            ORDER BY week_start_date DESC
            LIMIT 1
            """,
            parameters={
                "user_id": user_id,
                "base_date": base_date,
            },
        )

        as_of = datetime.now(timezone.utc).astimezone().isoformat()
        if not rows:
            return _build_standard_contract(
                as_of=as_of,
                window={
                    "date_from": _as_iso_date(base_date),
                    "date_to": _as_iso_date(base_date),
                },
                summary={},
                evidence=[],
                warnings=["No weekly budget found for the requested date."],
            )

        row = rows[0]
        summary = {
            "budget_week_id": row.get("id"),
            "planned_income_amount": row.get("planned_income_amount"),
            "planned_essential_amount": row.get("planned_essential_amount"),
            "planned_saving_amount": row.get("planned_saving_amount"),
            "rollover_in_amount": row.get("rollover_in_amount"),
            "manual_adjustment_amount": row.get("manual_adjustment_amount"),
            "final_free_amount": row.get("final_free_amount"),
            "actual_spent_amount": row.get("actual_spent_amount"),
            "status": row.get("status"),
        }
        adjustment_rows = self.runner.execute_select(
            """
            SELECT
                ba.id,
                ba.plan_id,
                ba.adjustment_amount,
                ba.adjustment_type,
                ba.reason,
                ba.created_at
            FROM budget_adjustment ba
            JOIN budget_week bw
              ON bw.id = ba.budget_week_id
            WHERE bw.id = :budget_week_id
              AND bw.users_id = :user_id
              AND ba.is_deleted = false
            ORDER BY ba.created_at DESC, ba.id DESC
            LIMIT 5
            """,
            parameters={
                "user_id": user_id,
                "budget_week_id": row.get("id"),
            },
        )
        rollover_rows = self.runner.execute_select(
            """
            SELECT
                br.id,
                br.source_budget_week_id,
                br.target_budget_week_id,
                br.rollover_amount,
                br.rollover_step,
                br.note,
                br.status,
                br.created_at,
                br.resolved_at
            FROM budget_rollover br
            JOIN budget_week bw
              ON bw.id = br.target_budget_week_id
            WHERE bw.id = :budget_week_id
              AND bw.users_id = :user_id
              AND br.is_deleted = false
            ORDER BY br.created_at DESC, br.id DESC
            LIMIT 5
            """,
            parameters={
                "user_id": user_id,
                "budget_week_id": row.get("id"),
            },
        )
        evidence = [
            _build_budget_week_evidence(row),
            _build_adjustment_evidence(
                budget_week_id=row.get("id"),
                rows=adjustment_rows,
            ),
            _build_rollover_evidence(
                budget_week_id=row.get("id"),
                rows=rollover_rows,
            ),
        ]
        return _build_standard_contract(
            as_of=as_of,
            window={
                "date_from": _as_iso_date(row.get("week_start_date")),
                "date_to": _as_iso_date(row.get("week_end_date")),
            },
            summary=summary,
            evidence=evidence,
            warnings=[],
        )

    def get_goal_progress_overview(
        self,
        *,
        user_id: int,
        status_filter: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        validate_user_id(user_id)
        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        where_clauses = [
            "users_id = :user_id",
            "is_deleted = false",
        ]
        parameters: dict[str, Any] = {
            "user_id": user_id,
            "limit": limit,
        }
        if status_filter is not None:
            where_clauses.append("status = :status_filter")
            parameters["status_filter"] = status_filter

        sql = f"""
            SELECT
                id,
                title,
                description,
                target_amount,
                saved_amount,
                start_date,
                end_date,
                priority,
                status,
                auto_allocate_enabled
            FROM plan
            WHERE {" AND ".join(where_clauses)}
            ORDER BY priority DESC, id DESC
            LIMIT :limit
        """
        rows = self.runner.execute_select(sql, parameters=parameters)

        as_of = datetime.now(timezone.utc).astimezone().isoformat()
        plan_summaries: list[dict[str, Any]] = []
        for row in rows:
            target_amount = row.get("target_amount") or 0
            saved_amount = row.get("saved_amount") or 0
            progress_ratio = None
            if target_amount:
                progress_ratio = round(float(saved_amount) / float(target_amount), 4)

            plan_summaries.append(
                {
                    "plan_id": row.get("id"),
                    "title": row.get("title"),
                    "description": row.get("description"),
                    "target_amount": target_amount,
                    "saved_amount": saved_amount,
                    "progress_ratio": progress_ratio,
                    "start_date": _as_iso_date(row.get("start_date")),
                    "end_date": _as_iso_date(row.get("end_date")),
                    "priority": row.get("priority"),
                    "status": row.get("status"),
                    "auto_allocate_enabled": row.get("auto_allocate_enabled"),
                }
            )

        warnings: list[str] = []
        if not plan_summaries:
            warnings.append("No plans found for the requested filters.")

        return _build_standard_contract(
            as_of=as_of,
            window={
                "date_from": None,
                "date_to": None,
            },
            summary={
                "goal_count": len(plan_summaries),
                "plans": plan_summaries,
            },
            evidence=plan_summaries,
            warnings=warnings,
        )

    def _load_latest_analysis_metadata(self, *, user_id: int) -> tuple[list[dict[str, Any]], list[str]]:
        rows = self.runner.execute_select(
            """
            SELECT
                id,
                requested_months,
                transaction_count,
                analyzed_at,
                latest_transaction_at,
                window_start_at,
                window_end_at
            FROM transaction_analysis_metadata
            WHERE users_id = :user_id
              AND is_deleted = false
            ORDER BY analyzed_at DESC, id DESC
            LIMIT 1
            """,
            parameters={
                "user_id": user_id,
            },
        )
        warnings: list[str] = []
        if not rows:
            warnings.append(_build_transaction_scope_warning())
        return rows, warnings

    def get_spending_summary(
        self,
        *,
        user_id: int,
        date_from: date,
        date_to: date,
    ) -> dict[str, Any]:
        validate_user_id(user_id)
        if date_from > date_to:
            raise ValueError("date_from must be on or before date_to")

        rows = self.runner.execute_select(
            f"""
            {_build_user_scoped_transactions_cte()}
            SELECT
                COALESCE(SUM(ABS(COALESCE(tx.transaction_balance, 0))), 0) AS total_spent,
                COUNT(*) AS transaction_count,
                COALESCE(ROUND(AVG(ABS(COALESCE(tx.transaction_balance, 0)))::numeric, 2), 0) AS average_transaction_amount,
                COALESCE(MAX(ABS(COALESCE(tx.transaction_balance, 0))), 0) AS largest_transaction_amount,
                (
                    SELECT tx2.transaction_memo
                    FROM scoped_transactions tx2
                    JOIN transaction_analysis_result ar2
                      ON ar2.transaction_override_id = tx2.id
                     AND ar2.is_deleted = false
                    WHERE tx2.transaction_date_time >= :date_from
                      AND tx2.transaction_date_time < (:date_to + INTERVAL '1 day')
                      AND ar2.analysis_included = true
                      AND ar2.expense_scope IS NOT NULL
                    ORDER BY ABS(COALESCE(tx2.transaction_balance, 0)) DESC,
                             tx2.transaction_date_time DESC,
                             tx2.id DESC
                    LIMIT 1
                ) AS largest_transaction_memo,
                COUNT(DISTINCT ar.category_id) AS category_count
            FROM scoped_transactions tx
            JOIN transaction_analysis_result ar
              ON ar.transaction_override_id = tx.id
             AND ar.is_deleted = false
            WHERE tx.transaction_date_time >= :date_from
              AND tx.transaction_date_time < (:date_to + INTERVAL '1 day')
              AND ar.analysis_included = true
              AND ar.expense_scope IS NOT NULL
            """,
            parameters={
                "user_id": user_id,
                "date_from": date_from,
                "date_to": date_to,
            },
        )
        metadata_rows, warnings = self._load_latest_analysis_metadata(user_id=user_id)

        as_of = datetime.now(timezone.utc).astimezone().isoformat()
        summary_row = rows[0] if rows else {}
        summary = {
            "total_spent": summary_row.get("total_spent", 0),
            "transaction_count": summary_row.get("transaction_count", 0),
            "average_transaction_amount": summary_row.get("average_transaction_amount", 0),
            "largest_transaction_amount": summary_row.get("largest_transaction_amount", 0),
            "largest_transaction_memo": summary_row.get("largest_transaction_memo"),
            "category_count": summary_row.get("category_count", 0),
        }
        if metadata_rows:
            metadata_row = metadata_rows[0]
            summary["analysis_window_start_at"] = _as_iso_datetime(
                metadata_row.get("window_start_at")
            )
            summary["analysis_window_end_at"] = _as_iso_datetime(
                metadata_row.get("window_end_at")
            )
            summary["analysis_requested_months"] = metadata_row.get("requested_months")

        if not rows:
            warnings.append("No spending transactions found for the requested window.")

        evidence = [
            {
                "kind": "spending_summary",
                "date_from": _as_iso_date(date_from),
                "date_to": _as_iso_date(date_to),
                "total_spent": summary["total_spent"],
                "transaction_count": summary["transaction_count"],
                "average_transaction_amount": summary["average_transaction_amount"],
                "largest_transaction_amount": summary["largest_transaction_amount"],
                "largest_transaction_memo": summary["largest_transaction_memo"],
                "category_count": summary["category_count"],
            }
        ]
        return _build_standard_contract(
            as_of=as_of,
            window={
                "date_from": _as_iso_date(date_from),
                "date_to": _as_iso_date(date_to),
            },
            summary=summary,
            evidence=evidence,
            warnings=warnings,
        )

    def get_spending_by_category(
        self,
        *,
        user_id: int,
        date_from: date,
        date_to: date,
        include_comparison: bool,
    ) -> dict[str, Any]:
        validate_user_id(user_id)
        if date_from > date_to:
            raise ValueError("date_from must be on or before date_to")

        comparison_rows: list[dict[str, Any]] = []
        if include_comparison:
            window_days = (date_to - date_from).days + 1
            comparison_date_from = date_from - timedelta(days=window_days)
            comparison_date_to = date_from - timedelta(days=1)
        else:
            comparison_date_from = None
            comparison_date_to = None

        current_rows = self.runner.execute_select(
            f"""
            {_build_user_scoped_transactions_cte()}
            SELECT
                COALESCE(ar.category_id, 0) AS category_id,
                COALESCE(cat.name, '미분류') AS category_name,
                COUNT(*) AS transaction_count,
                COALESCE(SUM(ABS(COALESCE(tx.transaction_balance, 0))), 0) AS spent_amount
            FROM scoped_transactions tx
            JOIN transaction_analysis_result ar
              ON ar.transaction_override_id = tx.id
             AND ar.is_deleted = false
            LEFT JOIN category cat
              ON cat.id = ar.category_id
            WHERE tx.transaction_date_time >= :date_from
              AND tx.transaction_date_time < (:date_to + INTERVAL '1 day')
              AND ar.analysis_included = true
              AND ar.expense_scope IS NOT NULL
            GROUP BY COALESCE(ar.category_id, 0), COALESCE(cat.name, '미분류')
            ORDER BY spent_amount DESC, category_name ASC
            """,
            parameters={
                "user_id": user_id,
                "date_from": date_from,
                "date_to": date_to,
            },
        )
        comparison_total_spent = None
        if include_comparison:
            comparison_rows = self.runner.execute_select(
                f"""
                {_build_user_scoped_transactions_cte()}
                SELECT
                    COALESCE(SUM(ABS(COALESCE(tx.transaction_balance, 0))), 0) AS comparison_total_spent
                FROM scoped_transactions tx
                JOIN transaction_analysis_result ar
                  ON ar.transaction_override_id = tx.id
                 AND ar.is_deleted = false
                WHERE tx.transaction_date_time >= :date_from
                  AND tx.transaction_date_time < (:date_to + INTERVAL '1 day')
                  AND ar.analysis_included = true
                  AND ar.expense_scope IS NOT NULL
                """,
                parameters={
                    "user_id": user_id,
                    "date_from": comparison_date_from,
                    "date_to": comparison_date_to,
                },
            )
            if comparison_rows:
                comparison_total_spent = comparison_rows[0].get("comparison_total_spent")

        metadata_rows, warnings = self._load_latest_analysis_metadata(user_id=user_id)
        if not current_rows:
            warnings.append("No spending transactions found for the requested window.")

        total_spent = sum((row.get("spent_amount") or 0) for row in current_rows)
        summary = {
            "total_spent": total_spent,
            "category_count": len(current_rows),
            "top_category_name": current_rows[0].get("category_name") if current_rows else None,
            "top_category_spent_amount": current_rows[0].get("spent_amount") if current_rows else None,
            "comparison_total_spent": comparison_total_spent,
            "comparison_delta_amount": (
                total_spent - comparison_total_spent
                if comparison_total_spent is not None
                else None
            ),
        }
        if metadata_rows:
            metadata_row = metadata_rows[0]
            summary["analysis_window_start_at"] = _as_iso_datetime(
                metadata_row.get("window_start_at")
            )
            summary["analysis_window_end_at"] = _as_iso_datetime(
                metadata_row.get("window_end_at")
            )

        evidence = [
            {
                "kind": "spending_category",
                "category_id": row.get("category_id"),
                "category_name": row.get("category_name"),
                "spent_amount": row.get("spent_amount"),
                "transaction_count": row.get("transaction_count"),
            }
            for row in current_rows
        ]
        return _build_standard_contract(
            as_of=datetime.now(timezone.utc).astimezone().isoformat(),
            window={
                "date_from": _as_iso_date(date_from),
                "date_to": _as_iso_date(date_to),
            },
            summary=summary,
            evidence=evidence,
            warnings=warnings,
        )

    def get_recent_spending_anomalies(
        self,
        *,
        user_id: int,
        lookback_days: int,
        limit: int,
        base_date: date | None = None,
    ) -> dict[str, Any]:
        validate_user_id(user_id)
        if lookback_days <= 0:
            raise ValueError("lookback_days must be a positive integer")
        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        date_to = base_date or date.today()
        date_from = date_to - timedelta(days=lookback_days - 1)

        rows = self.runner.execute_select(
            f"""
            {_build_user_scoped_transactions_cte()}
            SELECT
                tx.id AS transaction_id,
                tx.transaction_date_time,
                tx.transaction_balance,
                tx.transaction_memo,
                COALESCE(cat.name, '미분류') AS category_name,
                ar.final_label,
                ar.raw_label,
                ar.confidence,
                ar.threshold_used,
                ar.analysis_included,
                ar.excluded_reason,
                ar.expense_scope,
                ar.fallback_applied
            FROM scoped_transactions tx
            LEFT JOIN category cat
              ON cat.id = tx.category_id
            LEFT JOIN transaction_analysis_result ar
              ON ar.transaction_override_id = tx.id
             AND ar.is_deleted = false
            WHERE tx.transaction_date_time >= :date_from
              AND tx.transaction_date_time < (:date_to + INTERVAL '1 day')
              AND ar.analysis_included = true
              AND (
                  ar.final_label IN ('ANOMALY', 'UNUSUAL', 'SUSPICIOUS')
                  OR ar.confidence IS NULL
                  OR ar.threshold_used IS NULL
                  OR ar.confidence < ar.threshold_used
                  OR ar.fallback_applied = true
              )
            ORDER BY tx.transaction_date_time DESC, tx.id DESC
            LIMIT :limit
            """,
            parameters={
                "user_id": user_id,
                "date_from": date_from,
                "date_to": date_to,
                "limit": limit,
            },
        )
        metadata_rows, warnings = self._load_latest_analysis_metadata(user_id=user_id)
        if not rows:
            warnings.append("No recent spending anomalies found for the requested window.")

        summary = {
            "anomaly_count": len(rows),
            "lookback_days": lookback_days,
            "limit": limit,
            "latest_anomaly_at": _as_iso_datetime(rows[0].get("transaction_date_time"))
            if rows
            else None,
        }
        if metadata_rows:
            metadata_row = metadata_rows[0]
            summary["analysis_window_start_at"] = _as_iso_datetime(
                metadata_row.get("window_start_at")
            )
            summary["analysis_window_end_at"] = _as_iso_datetime(
                metadata_row.get("window_end_at")
            )

        evidence = [
            {
                "kind": "spending_anomaly",
                "transaction_id": row.get("transaction_id"),
                "transaction_date_time": _as_iso_datetime(row.get("transaction_date_time")),
                "transaction_balance": row.get("transaction_balance"),
                "transaction_memo": row.get("transaction_memo"),
                "category_name": row.get("category_name"),
                "final_label": row.get("final_label"),
                "raw_label": row.get("raw_label"),
                "confidence": row.get("confidence"),
                "threshold_used": row.get("threshold_used"),
                "analysis_included": row.get("analysis_included"),
                "excluded_reason": row.get("excluded_reason"),
                "expense_scope": row.get("expense_scope"),
                "fallback_applied": row.get("fallback_applied"),
            }
            for row in rows
        ]
        return _build_standard_contract(
            as_of=datetime.now(timezone.utc).astimezone().isoformat(),
            window={
                "date_from": _as_iso_date(date_from),
                "date_to": _as_iso_date(date_to),
            },
            summary=summary,
            evidence=evidence,
            warnings=warnings,
        )
