from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.core.database import validate_user_id


class ConversationNotFoundError(LookupError):
    """Raised when a conversation does not exist for the requested user."""


class ConversationExpiredError(RuntimeError):
    """Raised when a conversation exists but its TTL has expired."""


@dataclass(slots=True)
class _ConversationMessage:
    role: str
    content: str
    created_at: datetime


@dataclass(slots=True)
class _ConversationRecord:
    conversation_id: str
    user_id: int
    created_at: datetime
    last_activity_at: datetime
    expires_at: datetime
    messages: list[_ConversationMessage] = field(default_factory=list)
    summary_text: str = ""
    summarized_message_count: int = 0


class ConversationMemoryStore:
    def __init__(
        self,
        *,
        ttl: timedelta,
        now_factory: callable | None = None,
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("ttl must be positive")

        self._ttl = ttl
        self._now_factory = now_factory or (lambda: datetime.now(UTC))
        self._conversations: dict[tuple[int, str], _ConversationRecord] = {}

    def append_user_message(
        self,
        *,
        user_id: int,
        conversation_id: str,
        content: str,
    ) -> None:
        record = self._get_or_create_record(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        timestamp = self._now()
        record.messages.append(
            _ConversationMessage(role="user", content=content, created_at=timestamp)
        )
        self._touch(record, timestamp=timestamp)

    def append_assistant_message(
        self,
        *,
        user_id: int,
        conversation_id: str,
        content: str,
    ) -> None:
        record = self._get_or_create_record(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        timestamp = self._now()
        record.messages.append(
            _ConversationMessage(role="assistant", content=content, created_at=timestamp)
        )
        self._touch(record, timestamp=timestamp)

    def ensure_active_conversation(
        self,
        *,
        user_id: int,
        conversation_id: str,
    ) -> None:
        self._get_active_record(user_id=user_id, conversation_id=conversation_id)

    def get_conversation(
        self,
        *,
        user_id: int,
        conversation_id: str,
    ) -> dict[str, object]:
        record = self._get_active_record(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return {
            "conversation_id": record.conversation_id,
            "created_at": record.created_at.isoformat(),
            "last_activity_at": record.last_activity_at.isoformat(),
            "expires_at": record.expires_at.isoformat(),
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "created_at": message.created_at.isoformat(),
                }
                for message in record.messages
            ],
        }

    def get_latest_conversation(
        self,
        *,
        user_id: int,
    ) -> dict[str, object]:
        latest_record = self._find_latest_active_record(user_id=user_id)
        if latest_record is None:
            raise ConversationNotFoundError("conversation was not found")

        return self._serialize_record_summary(latest_record)

    def get_or_create_latest_conversation(
        self,
        *,
        user_id: int,
        conversation_id_factory: Callable[[], str],
    ) -> dict[str, object]:
        latest_record = self._find_latest_active_record(user_id=user_id)
        if latest_record is not None:
            return self._serialize_record_summary(latest_record)

        record = self._get_or_create_record(
            user_id=user_id,
            conversation_id=conversation_id_factory(),
        )
        return self._serialize_record_summary(record)

    def get_prompt_context(
        self,
        *,
        user_id: int,
        conversation_id: str,
    ) -> dict[str, object]:
        record = self._get_active_record(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return {
            "summary_text": record.summary_text,
            "summarized_message_count": record.summarized_message_count,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "created_at": message.created_at.isoformat(),
                }
                for message in record.messages[record.summarized_message_count :]
            ],
        }

    def save_summary_state(
        self,
        *,
        user_id: int,
        conversation_id: str,
        summary_text: str,
        summarized_message_count: int,
    ) -> None:
        record = self._get_active_record(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if summarized_message_count < 0:
            raise ValueError("summarized_message_count must not be negative")
        if summarized_message_count > len(record.messages):
            raise ValueError("summarized_message_count exceeds message count")

        record.summary_text = summary_text
        record.summarized_message_count = summarized_message_count

    def _get_or_create_record(
        self,
        *,
        user_id: int,
        conversation_id: str,
    ) -> _ConversationRecord:
        key = self._build_key(user_id=user_id, conversation_id=conversation_id)
        record = self._conversations.get(key)
        if record is None:
            timestamp = self._now()
            record = _ConversationRecord(
                conversation_id=conversation_id.strip(),
                user_id=validate_user_id(user_id),
                created_at=timestamp,
                last_activity_at=timestamp,
                expires_at=timestamp + self._ttl,
            )
            self._conversations[key] = record
            return record

        self._raise_if_expired(key=key, record=record)
        return record

    def _get_active_record(
        self,
        *,
        user_id: int,
        conversation_id: str,
    ) -> _ConversationRecord:
        key = self._build_key(user_id=user_id, conversation_id=conversation_id)
        record = self._conversations.get(key)
        if record is None:
            raise ConversationNotFoundError("conversation was not found")

        self._raise_if_expired(key=key, record=record)
        return record

    def _find_latest_active_record(
        self,
        *,
        user_id: int,
    ) -> _ConversationRecord | None:
        validated_user_id = validate_user_id(user_id)
        latest_record: _ConversationRecord | None = None
        expired_keys: list[tuple[int, str]] = []
        current_time = self._now()

        for key, record in self._conversations.items():
            if record.user_id != validated_user_id:
                continue
            if current_time > record.expires_at:
                expired_keys.append(key)
                continue
            if (
                latest_record is None
                or record.last_activity_at > latest_record.last_activity_at
            ):
                latest_record = record

        for key in expired_keys:
            self._conversations.pop(key, None)

        return latest_record

    def _raise_if_expired(
        self,
        *,
        key: tuple[int, str],
        record: _ConversationRecord,
    ) -> None:
        if self._now() <= record.expires_at:
            return

        self._conversations.pop(key, None)
        raise ConversationExpiredError("conversation has expired")

    def _touch(
        self,
        record: _ConversationRecord,
        *,
        timestamp: datetime,
    ) -> None:
        record.last_activity_at = timestamp
        record.expires_at = timestamp + self._ttl

    def _build_key(
        self,
        *,
        user_id: int,
        conversation_id: str,
    ) -> tuple[int, str]:
        normalized_conversation_id = (conversation_id or "").strip()
        if not normalized_conversation_id:
            raise ValueError("conversation_id must not be blank")
        return (validate_user_id(user_id), normalized_conversation_id)

    @staticmethod
    def _serialize_record_summary(record: _ConversationRecord) -> dict[str, object]:
        return {
            "conversation_id": record.conversation_id,
            "created_at": record.created_at.isoformat(),
            "last_activity_at": record.last_activity_at.isoformat(),
            "expires_at": record.expires_at.isoformat(),
        }

    def _now(self) -> datetime:
        timestamp = self._now_factory()
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=UTC)
        return timestamp
