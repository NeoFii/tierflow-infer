"""Shared base settings for all services."""

from typing import List, Union

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseServiceSettings(BaseSettings):
    """Common settings shared across services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "Eucal AI"
    SERVICE_NAME: str = "service"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Eucal AI API"

    API_V1_PREFIX: str = "/api/v1"
    PORT: int = 8000

    DEBUG: bool = False
    ENV: str = "development"

    ALLOWED_HOSTS: Union[str, List[str]] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]
    PRODUCTION_ALLOWED_HOSTS: Union[str, List[str]] = []

    INTERNAL_SECRET: str = ""
    INTERNAL_REQUEST_TTL_SECONDS: int = 30
    INTERNAL_HTTP_MAX_RETRIES: int = 1
    INTERNAL_HTTP_RETRY_BACKOFF_SECONDS: float = 0.2
    INTERNAL_HTTP_CIRCUIT_BREAKER_THRESHOLD: int = 3
    INTERNAL_HTTP_CIRCUIT_BREAKER_COOLDOWN_SECONDS: float = 30.0
    TIMEZONE: str = "Asia/Shanghai"

    DATABASE_URL: str = ""
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_ECHO: bool = False

    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    JWT_SECRET_KEY_MIN_LENGTH: int = 32

    COOKIE_SECURE: bool = True
    COOKIE_SAMESITE: str = "strict"

    REDIS_URL: str = "redis://127.0.0.1:6379/0"

    SNOWFLAKE_WORKER_ID: int = 1
    SNOWFLAKE_DATACENTER_ID: int = 1

    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True

    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"
    LOG_MAX_DAYS: int = 30
    LOG_FILE_PREFIX: str = "service"
    LOG_TO_FILE: bool = False
    LOG_FILE_MAX_BYTES: int = 50 * 1024 * 1024
    LOG_FILE_BACKUP_COUNT: int = 5
    LOG_RING_BUFFER_CAPACITY: int = 2000

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, value):
        """Parse CORS host lists from env strings or arrays."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                import json

                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            return [host.strip() for host in value.split(",") if host.strip()]
        return ["http://localhost:5173", "http://localhost:3000"]

    @model_validator(mode="after")
    def validate_required_fields(self) -> "BaseServiceSettings":
        """Validate required security-sensitive settings."""
        if not self.JWT_SECRET_KEY or self.JWT_SECRET_KEY in {"your-secret-key", "change-me"}:
            raise ValueError("JWT_SECRET_KEY must be explicitly configured")

        if len(self.JWT_SECRET_KEY) < self.JWT_SECRET_KEY_MIN_LENGTH:
            raise ValueError(
                f"JWT_SECRET_KEY length must be at least {self.JWT_SECRET_KEY_MIN_LENGTH}"
            )

        if not self.INTERNAL_SECRET:
            raise ValueError("INTERNAL_SECRET must be configured")

        if len(self.INTERNAL_SECRET) < 32:
            raise ValueError("INTERNAL_SECRET length must be at least 32")

        return self

    @property
    def cors_allowed_hosts(self) -> List[str]:
        """Return the effective allowed CORS origins."""
        if not self.DEBUG and self.PRODUCTION_ALLOWED_HOSTS:
            if isinstance(self.PRODUCTION_ALLOWED_HOSTS, str):
                return [
                    host.strip()
                    for host in self.PRODUCTION_ALLOWED_HOSTS.split(",")
                    if host.strip()
                ]
            return self.PRODUCTION_ALLOWED_HOSTS

        if isinstance(self.ALLOWED_HOSTS, str):
            return [host.strip() for host in self.ALLOWED_HOSTS.split(",") if host.strip()]
        return self.ALLOWED_HOSTS
