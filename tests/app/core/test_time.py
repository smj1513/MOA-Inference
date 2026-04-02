from __future__ import annotations

from datetime import UTC, date, datetime

from app.core.time import current_date_in_timezone


def test_current_date_in_timezone_converts_utc_clock_to_kst_date() -> None:
    resolved = current_date_in_timezone(
        "Asia/Seoul",
        now_factory=lambda: datetime(2026, 3, 26, 15, 30, tzinfo=UTC),
    )

    assert resolved == date(2026, 3, 27)
