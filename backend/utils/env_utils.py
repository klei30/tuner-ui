"""
Environment Utilities
=====================

Utilities for managing environment variables and configuration.
"""

import os
from typing import Optional


class EnvironmentError(Exception):
    """Raised when required environment variables are missing or invalid."""
    pass


def get_required_env(key: str, context: str = "operation") -> str:
    """
    Get a required environment variable or raise an error.

    Args:
        key: Environment variable key
        context: Context for the error message (e.g., "training", "evaluation")

    Returns:
        Environment variable value

    Raises:
        EnvironmentError: If the environment variable is not set

    Example:
        >>> api_key = get_required_env("TINKER_API_KEY", "training")
    """
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"{key} environment variable is required for {context}. "
            f"Please set it in your .env file or environment."
        )
    return value


def setup_training_environment(api_key: Optional[str] = None, hf_token: Optional[str] = None) -> None:
    """
    Setup environment variables required for training.

    Sets up TINKER_API_KEY, HF_TOKEN, and PYTHONIOENCODING for training operations.

    Args:
        api_key: Optional API key to use. If None, reads from environment.
        hf_token: Optional HuggingFace token for gated models. If provided,
                  will be set as HF_TOKEN environment variable.

    Raises:
        EnvironmentError: If API key is not provided and not in environment

    Example:
        >>> setup_training_environment()  # Uses env var
        >>> setup_training_environment("my-key", "hf_xxx")  # Uses provided keys
    """
    if api_key is None:
        api_key = get_required_env("TINKER_API_KEY", "training")

    os.environ["TINKER_API_KEY"] = api_key
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # Set HuggingFace token if provided (for gated models like Llama)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token


def get_optional_env(key: str, default: str = "") -> str:
    """
    Get an optional environment variable with a default value.

    Args:
        key: Environment variable key
        default: Default value if not set

    Returns:
        Environment variable value or default

    Example:
        >>> log_level = get_optional_env("LOG_LEVEL", "INFO")
    """
    return os.environ.get(key, default)


def validate_api_key(api_key: str) -> bool:
    """
    Validate that an API key has the expected format.

    Tinker API keys start with "tml-" followed by alphanumeric characters.

    Args:
        api_key: API key to validate

    Returns:
        True if valid format, False otherwise

    Example:
        >>> validate_api_key("tml-abc123...")
        True
        >>> validate_api_key("invalid-key")
        False
    """
    if not api_key:
        return False

    # Tinker API keys start with "tml-"
    if not api_key.startswith("tml-"):
        return False

    # Should have reasonable length (at least 20 chars total)
    if len(api_key) < 20:
        return False

    return True


def setup_test_environment() -> None:
    """
    Setup environment for testing.

    Sets test-specific environment variables to avoid affecting production.
    """
    os.environ["TESTING"] = "true"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # Use test API key if not already set
    if "TINKER_API_KEY" not in os.environ:
        os.environ["TINKER_API_KEY"] = "tml-test-key-for-unit-tests"
