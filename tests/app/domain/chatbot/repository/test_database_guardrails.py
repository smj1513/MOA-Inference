import pytest

from app.core.database import (
    ReadOnlyDatabase,
    RequestScopedReadOnlyDatabase,
    assert_select_only_sql,
    build_thread_id,
    jdbc_to_sqlalchemy_dsn,
    validate_user_id,
)


class _FakeMappingsResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def all(self) -> list[dict[str, object]]:
        return self._rows


class _FakeResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> _FakeMappingsResult:
        return _FakeMappingsResult(self._rows)


class _FakeBeginContext:
    def __init__(self) -> None:
        self.rollback_calls = 0
        self.is_active = True

    def __enter__(self) -> "_FakeBeginContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def rollback(self) -> None:
        self.rollback_calls += 1
        self.is_active = False


class _FakeConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.driver_sql_calls: list[str] = []
        self.execute_calls: list[tuple[str, dict[str, object]]] = []
        self.begin_contexts: list[_FakeBeginContext] = []
        self.close_calls = 0

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def begin(self) -> _FakeBeginContext:
        context = _FakeBeginContext()
        self.begin_contexts.append(context)
        return context

    def exec_driver_sql(self, statement: str) -> None:
        self.driver_sql_calls.append(statement)

    def execute(self, statement, parameters: dict[str, object]) -> _FakeResult:
        self.execute_calls.append((str(statement), parameters))
        return _FakeResult(self.rows)

    def close(self) -> None:
        self.close_calls += 1


class _FakeEngine:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.connection = _FakeConnection(rows)
        self.connect_calls = 0

    def connect(self) -> _FakeConnection:
        self.connect_calls += 1
        return self.connection


def test_build_thread_id_namespaces_conversation_by_user() -> None:
    assert build_thread_id(user_id=7, conversation_id="abc") == "moa:7:abc"


def test_validate_user_id_rejects_non_positive_values() -> None:
    try:
        validate_user_id(0)
    except ValueError as exc:
        assert "user_id" in str(exc)
    else:
        raise AssertionError("validate_user_id should reject zero")


def test_assert_select_only_sql_rejects_non_select_statements() -> None:
    try:
        assert_select_only_sql("DELETE FROM plan")
    except ValueError as exc:
        assert "SELECT" in str(exc)
    else:
        raise AssertionError("non-select SQL must be rejected")


def test_assert_select_only_sql_allows_read_only_cte_queries() -> None:
    sql = """
    WITH scoped_transactions AS (
        SELECT 1 AS id
    )
    SELECT id
    FROM scoped_transactions
    """

    assert assert_select_only_sql(sql) == sql.strip()


def test_assert_select_only_sql_rejects_write_keywords_inside_cte() -> None:
    sql = """
    WITH deleted_rows AS (
        DELETE FROM plan
        RETURNING id
    )
    SELECT id
    FROM deleted_rows
    """

    with pytest.raises(ValueError, match="read-only"):
        assert_select_only_sql(sql)


def test_assert_select_only_sql_allows_leading_sql_comments() -> None:
    sql = """
    -- scoped read query
    WITH scoped_transactions AS (
        SELECT 1 AS id
    )
    SELECT id
    FROM scoped_transactions
    """

    assert assert_select_only_sql(sql) == sql.strip()


def test_jdbc_to_sqlalchemy_dsn_converts_jdbc_url_to_psycopg_dsn() -> None:
    assert (
        jdbc_to_sqlalchemy_dsn(
            jdbc_url="jdbc:postgresql://postgres:5432/amagetdone",
            username="d101",
            password="p@ss:word/1#x",
        )
        == "postgresql+psycopg://d101:p%40ss%3Aword%2F1%23x@postgres:5432/amagetdone"
    )


def test_read_only_database_executes_select_in_read_only_transaction() -> None:
    engine = _FakeEngine(rows=[{"id": 1}])
    database = ReadOnlyDatabase(engine=engine)

    result = database.execute_select(
        "SELECT 1 AS id",
        parameters={"user_id": 7},
    )

    assert result == [{"id": 1}]
    assert engine.connection.driver_sql_calls == ["SET TRANSACTION READ ONLY"]
    assert engine.connection.execute_calls == [
        ("SELECT 1 AS id", {"user_id": 7}),
    ]


def test_read_only_database_from_dsn_uses_bounded_pool_configuration(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_engine(dsn: str, **kwargs):
        captured["dsn"] = dsn
        captured["kwargs"] = kwargs
        return _FakeEngine(rows=[])

    monkeypatch.setattr("app.core.database.create_engine", fake_create_engine)

    database = ReadOnlyDatabase.from_dsn("postgresql+psycopg://user:pass@db/app")

    assert isinstance(database, ReadOnlyDatabase)
    assert captured == {
        "dsn": "postgresql+psycopg://user:pass@db/app",
        "kwargs": {
            "future": True,
            "pool_pre_ping": True,
            "pool_size": 3,
            "max_overflow": 0,
            "pool_timeout": 30,
            "pool_recycle": 1800,
        },
    }


def test_request_scoped_database_reuses_single_connection_for_multiple_selects() -> None:
    engine = _FakeEngine(rows=[{"id": 1}])
    database = ReadOnlyDatabase(engine=engine)
    scoped = database.create_request_scoped()

    first = scoped.execute_select("SELECT 1 AS id", parameters={"user_id": 7})
    second = scoped.execute_select("SELECT 2 AS id", parameters={"user_id": 8})
    scoped.close()

    assert isinstance(scoped, RequestScopedReadOnlyDatabase)
    assert first == [{"id": 1}]
    assert second == [{"id": 1}]
    assert engine.connect_calls == 1
    assert engine.connection.driver_sql_calls == ["SET TRANSACTION READ ONLY"]
    assert engine.connection.execute_calls == [
        ("SELECT 1 AS id", {"user_id": 7}),
        ("SELECT 2 AS id", {"user_id": 8}),
    ]
    assert engine.connection.close_calls == 1
    assert len(engine.connection.begin_contexts) == 1
    assert engine.connection.begin_contexts[0].rollback_calls == 1
