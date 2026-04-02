from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo


def current_date_in_timezone(
    timezone_name: str,
    *,
    now_factory: Callable[[], datetime] | None = None,
) -> date:
    resolved_now = (now_factory or (lambda: datetime.now(UTC)))()
    if resolved_now.tzinfo is None:
        resolved_now = resolved_now.replace(tzinfo=UTC)
    return resolved_now.astimezone(ZoneInfo(timezone_name)).date()
