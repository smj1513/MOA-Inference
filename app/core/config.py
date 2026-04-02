from urllib.parse import quote

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_id: str = Field(
        default="kakao1513/merchant-consumption-category-discriminator-v3",
        alias="MODEL_ID",
    )
    model_device: str = Field(default="cpu", alias="MODEL_DEVICE")
    fallback_label: str = Field(default="기타서비스", alias="FALLBACK_LABEL")
    fallback_threshold: float = Field(default=0.50, alias="FALLBACK_THRESHOLD")
    top_k: int = Field(default=3, alias="TOP_K")
    max_batch_size: int = Field(default=1000, alias="MAX_BATCH_SIZE")
    inference_batch_size: int = Field(default=128, alias="INFERENCE_BATCH_SIZE")
    model_revision: str | None = Field(default=None, alias="MODEL_REVISION")
    model_local_dir: str | None = Field(default=None, alias="MODEL_LOCAL_DIR")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    api_root_path: str = Field(default="", alias="API_ROOT_PATH")
    ai_api_key: str = Field(default="", alias="AI_API_KEY")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    google_ai_base_url: str = Field(default="", alias="GOOGLE_AI_BASE_URL")
    chat_model: str = Field(default="gpt-5-mini", alias="CHAT_MODEL")
    more_finance_mcp_url: str = Field(default="", alias="MORE_FINANCE_MCP_URL")
    db_url: str = Field(default="", alias="DB_URL")
    db_username: str = Field(default="", alias="DB_USERNAME")
    db_password: str = Field(default="", alias="DB_PASSWORD")
    db_pool_size: int = Field(default=3, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=0, alias="DB_MAX_OVERFLOW")
    db_pool_timeout_seconds: int = Field(default=30, alias="DB_POOL_TIMEOUT_SECONDS")
    db_pool_recycle_seconds: int = Field(default=1800, alias="DB_POOL_RECYCLE_SECONDS")
    chatbot_stream_status_events_enabled: bool = Field(
        default=True,
        alias="CHATBOT_STREAM_STATUS_EVENTS_ENABLED",
    )
    chatbot_timezone: str = Field(
        default="Asia/Seoul",
        alias="CHATBOT_TIMEZONE",
    )
    chatbot_default_goal_limit: int = Field(
        default=5,
        alias="CHATBOT_DEFAULT_GOAL_LIMIT",
    )
    image_generation_model: str = Field(
        default="gemini-2.5-flash-image",
        alias="IMAGE_GENERATION_MODEL",
    )
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str | None = Field(default=None, alias="LANGFUSE_BASE_URL")
    langfuse_tracing_enabled: bool = Field(
        default=True,
        alias="LANGFUSE_TRACING_ENABLED",
    )

    model_config = SettingsConfigDict(
        env_file=(".env", "app/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def model_source(self) -> str:
        return self.model_local_dir or self.model_id

    @property
    def database_dsn(self) -> str:
        return build_postgresql_dsn(
            jdbc_url=self.db_url,
            username=self.db_username,
            password=self.db_password,
        )

    @field_validator(
        "model_revision",
        "model_local_dir",
        "hf_token",
        "langfuse_public_key",
        "langfuse_secret_key",
        "langfuse_base_url",
        mode="before",
    )
    @classmethod
    def blank_strings_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


def build_postgresql_dsn(*, jdbc_url: str, username: str, password: str) -> str:
    if not jdbc_url or not username or not password:
        return ""

    prefix = "jdbc:postgresql://"
    if not jdbc_url.startswith(prefix):
        return ""

    remainder = jdbc_url.removeprefix(prefix)
    host_port, separator, database_name = remainder.partition("/")
    if not separator or not host_port or not database_name:
        return ""

    host, port_separator, port = host_port.partition(":")
    if not host:
        return ""

    normalized_port = port if port_separator and port else "5432"
    encoded_username = quote(username, safe="")
    encoded_password = quote(password, safe="")
    return (
        "postgresql+psycopg://"
        f"{encoded_username}:{encoded_password}@{host}:{normalized_port}/{database_name}"
    )
