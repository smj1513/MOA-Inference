from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain.chatbot.repository.conversation_memory_store import (
    ConversationExpiredError,
    ConversationMemoryStore,
    ConversationNotFoundError,
)


def test_store_creates_conversation_and_appends_messages() -> None:
    now = datetime(2026, 3, 26, 9, 0, tzinfo=UTC)
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: now,
    )

    store.append_user_message(user_id=7, conversation_id="conv-1", content="hello")
    store.append_assistant_message(
        user_id=7,
        conversation_id="conv-1",
        content="world",
    )

    conversation = store.get_conversation(user_id=7, conversation_id="conv-1")

    assert conversation["conversation_id"] == "conv-1"
    assert [message["role"] for message in conversation["messages"]] == [
        "user",
        "assistant",
    ]
    assert conversation["expires_at"] == "2026-03-29T09:00:00+00:00"


def test_store_refreshes_expiration_on_new_activity() -> None:
    current = {"now": datetime(2026, 3, 26, 9, 0, tzinfo=UTC)}
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: current["now"],
    )

    store.append_user_message(user_id=7, conversation_id="conv-1", content="first")
    current["now"] = datetime(2026, 3, 27, 12, 0, tzinfo=UTC)
    store.append_assistant_message(
        user_id=7,
        conversation_id="conv-1",
        content="second",
    )

    conversation = store.get_conversation(user_id=7, conversation_id="conv-1")

    assert conversation["expires_at"] == "2026-03-30T12:00:00+00:00"


def test_store_rejects_expired_conversation() -> None:
    current = {"now": datetime(2026, 3, 26, 9, 0, tzinfo=UTC)}
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: current["now"],
    )

    store.append_user_message(user_id=7, conversation_id="conv-1", content="hello")
    current["now"] = datetime(2026, 3, 29, 9, 0, 1, tzinfo=UTC)

    with pytest.raises(ConversationExpiredError):
        store.get_conversation(user_id=7, conversation_id="conv-1")


def test_store_keeps_same_conversation_id_isolated_per_user() -> None:
    now = datetime(2026, 3, 26, 9, 0, tzinfo=UTC)
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: now,
    )

    store.append_user_message(user_id=7, conversation_id="same", content="u7")
    store.append_user_message(user_id=8, conversation_id="same", content="u8")

    first = store.get_conversation(user_id=7, conversation_id="same")
    second = store.get_conversation(user_id=8, conversation_id="same")

    assert first["messages"][0]["content"] == "u7"
    assert second["messages"][0]["content"] == "u8"


def test_store_returns_latest_active_conversation_for_user() -> None:
    current = {"now": datetime(2026, 3, 26, 9, 0, tzinfo=UTC)}
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: current["now"],
    )

    store.append_user_message(user_id=7, conversation_id="conv-1", content="first")
    current["now"] = datetime(2026, 3, 26, 10, 0, tzinfo=UTC)
    store.append_user_message(user_id=7, conversation_id="conv-2", content="second")
    current["now"] = datetime(2026, 3, 26, 11, 0, tzinfo=UTC)
    store.append_user_message(user_id=8, conversation_id="other-user", content="third")

    latest = store.get_latest_conversation(user_id=7)

    assert latest == {
        "conversation_id": "conv-2",
        "created_at": "2026-03-26T10:00:00+00:00",
        "last_activity_at": "2026-03-26T10:00:00+00:00",
        "expires_at": "2026-03-29T10:00:00+00:00",
    }


def test_store_latest_conversation_ignores_expired_records() -> None:
    current = {"now": datetime(2026, 3, 26, 9, 0, tzinfo=UTC)}
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: current["now"],
    )

    store.append_user_message(user_id=7, conversation_id="expired", content="old")
    current["now"] = datetime(2026, 3, 29, 9, 0, 1, tzinfo=UTC)

    with pytest.raises(ConversationNotFoundError):
        store.get_latest_conversation(user_id=7)


def test_store_get_or_create_latest_creates_empty_conversation_when_missing() -> None:
    now = datetime(2026, 3, 26, 9, 0, tzinfo=UTC)
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: now,
    )

    latest = store.get_or_create_latest_conversation(
        user_id=7,
        conversation_id_factory=lambda: "generated-conv",
    )

    assert latest == {
        "conversation_id": "generated-conv",
        "created_at": "2026-03-26T09:00:00+00:00",
        "last_activity_at": "2026-03-26T09:00:00+00:00",
        "expires_at": "2026-03-29T09:00:00+00:00",
    }
    conversation = store.get_conversation(user_id=7, conversation_id="generated-conv")
    assert conversation["messages"] == []


def test_store_get_or_create_latest_reuses_active_conversation() -> None:
    current = {"now": datetime(2026, 3, 26, 9, 0, tzinfo=UTC)}
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: current["now"],
    )

    store.append_user_message(user_id=7, conversation_id="conv-1", content="hello")
    current["now"] = datetime(2026, 3, 26, 10, 0, tzinfo=UTC)

    latest = store.get_or_create_latest_conversation(
        user_id=7,
        conversation_id_factory=lambda: "generated-conv",
    )

    assert latest["conversation_id"] == "conv-1"


def test_store_get_or_create_latest_recreates_expired_conversation() -> None:
    current = {"now": datetime(2026, 3, 26, 9, 0, tzinfo=UTC)}
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: current["now"],
    )

    store.append_user_message(user_id=7, conversation_id="expired-conv", content="hello")
    current["now"] = datetime(2026, 3, 29, 9, 0, 1, tzinfo=UTC)

    latest = store.get_or_create_latest_conversation(
        user_id=7,
        conversation_id_factory=lambda: "new-conv",
    )

    assert latest["conversation_id"] == "new-conv"


def test_store_can_persist_internal_summary_state_for_prompt_compaction() -> None:
    now = datetime(2026, 3, 26, 9, 0, tzinfo=UTC)
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: now,
    )
    store.append_user_message(user_id=7, conversation_id="conv-1", content="u1")
    store.append_assistant_message(user_id=7, conversation_id="conv-1", content="a1")
    store.append_user_message(user_id=7, conversation_id="conv-1", content="u2")

    store.save_summary_state(
        user_id=7,
        conversation_id="conv-1",
        summary_text="summary of prior turns",
        summarized_message_count=2,
    )

    prompt_context = store.get_prompt_context(user_id=7, conversation_id="conv-1")

    assert prompt_context == {
        "summary_text": "summary of prior turns",
        "summarized_message_count": 2,
        "messages": [
            {
                "role": "user",
                "content": "u2",
                "created_at": "2026-03-26T09:00:00+00:00",
            }
        ],
    }


def test_store_keeps_external_conversation_payload_unchanged_when_summary_exists() -> None:
    now = datetime(2026, 3, 26, 9, 0, tzinfo=UTC)
    store = ConversationMemoryStore(
        ttl=timedelta(days=3),
        now_factory=lambda: now,
    )
    store.append_user_message(user_id=7, conversation_id="conv-1", content="u1")
    store.append_assistant_message(user_id=7, conversation_id="conv-1", content="a1")
    store.append_user_message(user_id=7, conversation_id="conv-1", content="u2")
    store.save_summary_state(
        user_id=7,
        conversation_id="conv-1",
        summary_text="summary of prior turns",
        summarized_message_count=2,
    )

    conversation = store.get_conversation(user_id=7, conversation_id="conv-1")

    assert "summary_text" not in conversation
    assert "summarized_message_count" not in conversation
    assert [message["content"] for message in conversation["messages"]] == [
        "u1",
        "a1",
        "u2",
    ]
