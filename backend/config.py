"""Configuration management for Tinker Platform backend.

Environment variables can be set via .env file or system environment.
All sensitive values (API keys, secrets) MUST be provided via environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Tinker Platform"
    app_version: str = "0.1.0"
    debug: bool = False

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False

    # Database
    database_url: str = "sqlite:///tinker_platform.db"

    # Security - NO DEFAULTS FOR SENSITIVE VALUES!
    tinker_api_key: Optional[str] = None  # MUST be set via environment
    secret_key: str = "dev-secret-key-change-in-production"  # For JWT/sessions
    encryption_key: Optional[str] = None  # Fernet key for encrypting sensitive data

    # Redis (for Celery task queue)
    redis_url: str = "redis://localhost:6379/0"

    # Monitoring
    sentry_dsn: Optional[str] = None  # Sentry error tracking
    environment: str = "development"  # development, staging, production

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # File paths
    artifacts_dir: Path = Path("artifacts")
    cookbook_path: Optional[Path] = None  # Auto-detected if None

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Job execution
    job_timeout_seconds: int = 3600  # 1 hour default
    job_poll_interval_seconds: float = 0.5
    max_concurrent_jobs: int = 3

    # File reading limits
    log_tail_lines: int = 200
    max_file_size_mb: int = 100

    # Model catalog
    default_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Polling intervals (frontend)
    run_list_poll_interval_ms: int = 3000
    run_detail_poll_interval_ms: int = 2000

    # Validators for production security
    @field_validator("secret_key")
    @classmethod
    def secret_key_must_be_strong(cls, v: str, info) -> str:
        """Validate that SECRET_KEY is changed from default in production."""
        environment = info.data.get("environment", "development")

        if environment == "production":
            if v == "dev-secret-key-change-in-production":
                raise ValueError(
                    "SECRET_KEY must be changed from default in production! "
                    "Generate a strong random key for production use."
                )
            if len(v) < 32:
                raise ValueError(
                    "SECRET_KEY must be at least 32 characters long in production"
                )
        return v

    @field_validator("encryption_key")
    @classmethod
    def encryption_key_must_be_valid_fernet(cls, v: Optional[str], info) -> Optional[str]:
        """Validate that ENCRYPTION_KEY is a valid Fernet key in production."""
        if v is None:
            return v

        environment = info.data.get("environment", "development")

        if environment == "production":
            try:
                from cryptography.fernet import Fernet
                Fernet(v.encode())
            except Exception as e:
                raise ValueError(
                    f"ENCRYPTION_KEY must be a valid Fernet key in production. "
                    f"Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\". "
                    f"Error: {e}"
                )
        return v

    @field_validator("database_url")
    @classmethod
    def database_url_must_be_postgres_in_production(cls, v: str, info) -> str:
        """Validate that DATABASE_URL uses PostgreSQL in production."""
        environment = info.data.get("environment", "development")

        if environment == "production" and v.startswith("sqlite"):
            raise ValueError(
                "SQLite is not recommended for production! "
                "Please use PostgreSQL (postgresql://...) for production deployments."
            )
        return v

    def validate_required_settings(self) -> None:
        """Validate that required settings are present.

        Raises:
            ValueError: If required settings are missing
        """
        if not self.tinker_api_key:
            raise ValueError(
                "TINKER_API_KEY environment variable is required! "
                "Please set it in your .env file or environment."
            )

    def get_cookbook_path(self) -> Path:
        """Get path to tinker-cookbook, with auto-detection.

        Returns:
            Path to tinker-cookbook directory

        Raises:
            FileNotFoundError: If cookbook path cannot be found
        """
        if self.cookbook_path and self.cookbook_path.exists():
            return self.cookbook_path

        # Try to auto-detect
        backend_dir = Path(__file__).parent

        # Check if tinker-cookbook is a sibling directory
        possible_paths = [
            backend_dir.parent.parent / "tinker-cookbook",
            backend_dir.parent / "tinker-cookbook",
            Path.home() / "tinker-cookbook",
            Path("/Users") / os.getenv("USER", "") / "Desktop" / "tinker-cookbook",
            Path("C:/Users") / os.getenv("USERNAME", "") / "Desktop" / "tinker-cookbook",
        ]

        for path in possible_paths:
            if path.exists() and (path / "training").exists():
                return path

        raise FileNotFoundError(
            "Could not find tinker-cookbook directory. "
            "Please set COOKBOOK_PATH environment variable."
        )


# Global settings instance
settings = Settings()


# Logging configuration
def setup_logging() -> None:
    """Configure logging for the application."""
    import logging.config

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": settings.log_format,
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": "logs/backend.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": settings.log_level,
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "handlers": ["file"],
                "level": "WARNING",  # Only log warnings/errors from SQL
                "propagate": False,
            },
        },
    }

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    logging.config.dictConfig(LOGGING_CONFIG)


# Export commonly used settings
__all__ = [
    "settings",
    "Settings",
    "setup_logging",
]
