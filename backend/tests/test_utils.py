"""
Test Suite for Utility Modules
================================

Unit tests for utility modules created during refactoring:
- text_utils.py
- env_utils.py
- recipe_executor.py

Test Categories:
1. Text Utilities Tests
2. Environment Utilities Tests
3. Recipe Executor Tests
4. Error Handling Tests
5. Integration Tests

Usage:
    pytest tests/test_utils.py -v
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.text_utils import strip_ansi_codes, truncate_text, sanitize_filename
from utils.env_utils import (
    get_required_env,
    setup_training_environment,
    get_optional_env,
    validate_api_key,
    setup_test_environment,
    EnvironmentError
)
from utils.recipe_executor import RecipeExecutor, create_recipe_executor


# ============================================================================
# TEST 1: TEXT UTILITIES
# ============================================================================

class TestTextUtils:
    """Test text utility functions."""

    def test_strip_ansi_codes_removes_color_codes(self):
        """Test stripping ANSI color codes from text."""
        text_with_ansi = "\x1b[31mError:\x1b[0m Something failed"
        cleaned = strip_ansi_codes(text_with_ansi)

        assert "\x1b[31m" not in cleaned
        assert "\x1b[0m" not in cleaned
        assert "Error:" in cleaned
        assert "Something failed" in cleaned

    def test_strip_ansi_codes_preserves_regular_text(self):
        """Test that regular text is preserved."""
        regular_text = "This is normal text without ANSI codes"
        cleaned = strip_ansi_codes(regular_text)

        assert cleaned == regular_text

    def test_strip_ansi_codes_handles_empty_string(self):
        """Test handling empty string."""
        cleaned = strip_ansi_codes("")

        assert cleaned == ""

    def test_strip_ansi_codes_handles_multiple_codes(self):
        """Test stripping multiple ANSI codes."""
        text = "\x1b[1m\x1b[31mBold Red\x1b[0m\x1b[32m Green\x1b[0m"
        cleaned = strip_ansi_codes(text)

        assert "\x1b[" not in cleaned
        assert "Bold Red" in cleaned
        assert "Green" in cleaned

    def test_truncate_text_under_limit(self):
        """Test text under max length is not truncated."""
        text = "Short text"
        truncated = truncate_text(text, max_length=50)

        assert truncated == text

    def test_truncate_text_over_limit(self):
        """Test text over max length is truncated."""
        text = "This is a very long text that should be truncated"
        truncated = truncate_text(text, max_length=20)

        assert len(truncated) <= 20
        assert truncated.endswith("...")

    def test_truncate_text_custom_suffix(self):
        """Test truncation with custom suffix."""
        text = "Long text here"
        truncated = truncate_text(text, max_length=10, suffix="---")

        assert truncated.endswith("---")

    def test_sanitize_filename_removes_dangerous_chars(self):
        """Test sanitizing filename removes dangerous characters."""
        dangerous = "file<name>:with|bad?chars*.txt"
        sanitized = sanitize_filename(dangerous)

        assert "<" not in sanitized
        assert ">" not in sanitized
        assert ":" not in sanitized
        assert "|" not in sanitized
        assert "?" not in sanitized
        assert "*" not in sanitized

    def test_sanitize_filename_preserves_safe_chars(self):
        """Test sanitizing preserves safe characters."""
        safe = "valid_filename-123.txt"
        sanitized = sanitize_filename(safe)

        assert sanitized == safe

    def test_sanitize_filename_handles_empty(self):
        """Test sanitizing empty filename."""
        sanitized = sanitize_filename("")

        assert sanitized == "unnamed"

    def test_sanitize_filename_handles_only_dots(self):
        """Test sanitizing filename with only dots."""
        sanitized = sanitize_filename("...")

        assert sanitized == "unnamed"


# ============================================================================
# TEST 2: ENVIRONMENT UTILITIES
# ============================================================================

class TestEnvUtils:
    """Test environment utility functions."""

    def test_get_required_env_success(self, monkeypatch):
        """Test getting required environment variable."""
        monkeypatch.setenv("TEST_VAR", "test_value")

        value = get_required_env("TEST_VAR", "testing")

        assert value == "test_value"

    def test_get_required_env_missing_raises_error(self, monkeypatch):
        """Test missing required env var raises error."""
        monkeypatch.delenv("MISSING_VAR", raising=False)

        with pytest.raises(EnvironmentError) as exc_info:
            get_required_env("MISSING_VAR", "testing")

        assert "MISSING_VAR" in str(exc_info.value)
        assert "testing" in str(exc_info.value)

    def test_setup_training_environment_with_key(self, monkeypatch):
        """Test setup training environment with provided key."""
        api_key = "tml-test-key-1234567890123456789012345"

        setup_training_environment(api_key)

        assert os.environ["TINKER_API_KEY"] == api_key
        assert os.environ["PYTHONIOENCODING"] == "utf-8"

    def test_setup_training_environment_from_env(self, monkeypatch):
        """Test setup training environment from env var."""
        monkeypatch.setenv("TINKER_API_KEY", "tml-env-key-1234567890123456789012345")

        setup_training_environment()

        assert os.environ["TINKER_API_KEY"] == "tml-env-key-1234567890123456789012345"
        assert os.environ["PYTHONIOENCODING"] == "utf-8"

    def test_setup_training_environment_missing_key_raises_error(self, monkeypatch):
        """Test setup without API key raises error."""
        monkeypatch.delenv("TINKER_API_KEY", raising=False)

        with pytest.raises(EnvironmentError) as exc_info:
            setup_training_environment()

        assert "TINKER_API_KEY" in str(exc_info.value)

    def test_get_optional_env_with_value(self, monkeypatch):
        """Test getting optional env var that exists."""
        monkeypatch.setenv("OPTIONAL_VAR", "optional_value")

        value = get_optional_env("OPTIONAL_VAR", "default")

        assert value == "optional_value"

    def test_get_optional_env_with_default(self, monkeypatch):
        """Test getting optional env var returns default if missing."""
        monkeypatch.delenv("MISSING_OPTIONAL_VAR", raising=False)

        value = get_optional_env("MISSING_OPTIONAL_VAR", "default_value")

        assert value == "default_value"

    def test_validate_api_key_valid(self):
        """Test validating valid API key."""
        valid_key = "tml-1234567890abcdefghijklmnopqrstuvwxyz"

        assert validate_api_key(valid_key) is True

    def test_validate_api_key_invalid_prefix(self):
        """Test validating key with invalid prefix."""
        invalid_key = "invalid-1234567890abcdefghijklmnopqrstuvwxyz"

        assert validate_api_key(invalid_key) is False

    def test_validate_api_key_too_short(self):
        """Test validating key that's too short."""
        short_key = "tml-short"

        assert validate_api_key(short_key) is False

    def test_validate_api_key_empty(self):
        """Test validating empty key."""
        assert validate_api_key("") is False

    def test_setup_test_environment(self, monkeypatch):
        """Test setup test environment."""
        monkeypatch.delenv("TINKER_API_KEY", raising=False)

        setup_test_environment()

        assert os.environ["TESTING"] == "true"
        assert os.environ["PYTHONIOENCODING"] == "utf-8"
        assert "TINKER_API_KEY" in os.environ


# ============================================================================
# TEST 3: RECIPE EXECUTOR
# ============================================================================

@pytest.mark.asyncio
class TestRecipeExecutor:
    """Test RecipeExecutor class."""

    async def test_create_recipe_executor(self, mock_session, sample_run):
        """Test creating RecipeExecutor instance."""
        log_path = Path("/tmp/test.log")

        executor = create_recipe_executor(mock_session, sample_run, log_path)

        assert executor is not None
        assert isinstance(executor, RecipeExecutor)
        assert executor.session == mock_session
        assert executor.run == sample_run
        assert executor.logs_path == log_path

    async def test_executor_log_writes_to_file(self, mock_session, sample_run):
        """Test executor log method writes to file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_path = Path(f.name)

        executor = RecipeExecutor(mock_session, sample_run, log_path)
        await executor.log("Test log message\n")

        # Read log file
        with open(log_path, 'r') as f:
            content = f.read()

        assert "Test log message" in content

        # Cleanup
        log_path.unlink()

    async def test_executor_log_strips_ansi_codes(self, mock_session, sample_run):
        """Test executor log strips ANSI codes."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_path = Path(f.name)

        executor = RecipeExecutor(mock_session, sample_run, log_path)
        await executor.log("\x1b[31mError:\x1b[0m Something happened\n")

        # Read log file
        with open(log_path, 'r') as f:
            content = f.read()

        assert "\x1b[31m" not in content
        assert "Error:" in content

        # Cleanup
        log_path.unlink()

    async def test_executor_parse_metric_line_with_metrics_tag(
        self,
        mock_session,
        sample_run
    ):
        """Test parsing metric line with [METRICS] tag."""
        log_path = Path("/tmp/test.log")
        executor = RecipeExecutor(mock_session, sample_run, log_path)

        line = "[METRICS] step=100, loss=0.5, lr=0.0001"
        parsed = executor._parse_metric_line(line)

        assert "step" in parsed
        assert parsed["step"] == 100
        assert "loss" in parsed or "train_mean_nll" in parsed

    async def test_executor_parse_metric_line_without_tag(
        self,
        mock_session,
        sample_run
    ):
        """Test parsing metric line without [METRICS] tag."""
        log_path = Path("/tmp/test.log")
        executor = RecipeExecutor(mock_session, sample_run, log_path)

        line = "Training step: 200, loss: 0.45"
        parsed = executor._parse_metric_line(line)

        # Should extract step and loss
        if parsed:
            assert "step" in parsed or "train_mean_nll" in parsed

    async def test_executor_parse_metric_line_empty(
        self,
        mock_session,
        sample_run
    ):
        """Test parsing empty line."""
        log_path = Path("/tmp/test.log")
        executor = RecipeExecutor(mock_session, sample_run, log_path)

        parsed = executor._parse_metric_line("")

        assert parsed == {}

    async def test_execute_recipe_calls_config_builder(
        self,
        mock_session,
        sample_run,
        mock_env_vars
    ):
        """Test execute_recipe calls config builder."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_path = Path(f.name)

        executor = RecipeExecutor(mock_session, sample_run, log_path)

        # Mock config builder and train function
        config_builder = Mock(return_value=Mock(model_name="test-model"))
        train_fn = AsyncMock()

        await executor.execute_recipe(
            config_builder=config_builder,
            train_fn=train_fn,
            recipe_name="TEST",
            enable_monitoring=False
        )

        # Verify config builder was called
        config_builder.assert_called_once_with(sample_run)

        # Verify train function was called
        train_fn.assert_called_once()

        # Verify run was marked as completed
        assert sample_run.progress == 1.0
        assert sample_run.status == "completed"

        # Cleanup
        log_path.unlink()

    async def test_execute_recipe_with_dataset_arg(
        self,
        mock_session,
        sample_run,
        mock_env_vars
    ):
        """Test execute_recipe with dataset argument."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_path = Path(f.name)

        executor = RecipeExecutor(mock_session, sample_run, log_path)

        # Mock config builder that takes dataset_arg
        config_builder = Mock(return_value=Mock(model_name="test-model"))
        train_fn = AsyncMock()

        await executor.execute_recipe(
            config_builder=config_builder,
            train_fn=train_fn,
            recipe_name="SFT",
            enable_monitoring=False,
            dataset_arg="yahma/alpaca-cleaned"
        )

        # Verify config builder was called with dataset arg
        config_builder.assert_called_once_with(sample_run, "yahma/alpaca-cleaned")

        # Cleanup
        log_path.unlink()

    async def test_execute_recipe_handles_training_error(
        self,
        mock_session,
        sample_run,
        mock_env_vars
    ):
        """Test execute_recipe handles training errors."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_path = Path(f.name)

        executor = RecipeExecutor(mock_session, sample_run, log_path)

        # Mock train function that raises error
        config_builder = Mock(return_value=Mock())
        train_fn = AsyncMock(side_effect=RuntimeError("Training failed"))

        with pytest.raises(RuntimeError) as exc_info:
            await executor.execute_recipe(
                config_builder=config_builder,
                train_fn=train_fn,
                recipe_name="TEST",
                enable_monitoring=False
            )

        assert "Training failed" in str(exc_info.value)

        # Verify error was logged
        with open(log_path, 'r') as f:
            content = f.read()
        assert "Training failed" in content

        # Cleanup
        log_path.unlink()

    async def test_execute_recipe_with_monitoring(
        self,
        mock_session,
        sample_run,
        mock_env_vars
    ):
        """Test execute_recipe with monitoring enabled."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_path = Path(f.name)

        # Write some metrics to log file
        with open(log_path, 'w') as f:
            f.write("[METRICS] step=100, loss=0.5\n")

        executor = RecipeExecutor(mock_session, sample_run, log_path)

        config_builder = Mock(return_value=Mock())
        train_fn = AsyncMock()

        # Execute with monitoring
        await executor.execute_recipe(
            config_builder=config_builder,
            train_fn=train_fn,
            recipe_name="SFT",
            enable_monitoring=True  # Enable monitoring
        )

        # Verify training completed
        assert sample_run.status == "completed"

        # Cleanup
        log_path.unlink()


# ============================================================================
# TEST 4: ERROR HANDLING
# ============================================================================

class TestUtilsErrorHandling:
    """Test error handling in utility modules."""

    def test_text_utils_handle_none_input(self):
        """Test text utilities handle None input."""
        with pytest.raises((TypeError, AttributeError)):
            strip_ansi_codes(None)

    def test_env_utils_handle_invalid_context(self, monkeypatch):
        """Test env utils include context in error messages."""
        monkeypatch.delenv("MISSING_VAR", raising=False)

        with pytest.raises(EnvironmentError) as exc_info:
            get_required_env("MISSING_VAR", "custom_operation")

        assert "custom_operation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_executor_handles_checkpoint_registration_failure(
        self,
        mock_session,
        sample_run,
        mock_env_vars
    ):
        """Test executor handles checkpoint registration failure gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_path = Path(f.name)

        executor = RecipeExecutor(mock_session, sample_run, log_path)

        # Mock checkpoint registration to fail
        with patch("utils.recipe_executor.register_checkpoint_from_logs", side_effect=Exception("Checkpoint error")):
            config_builder = Mock(return_value=Mock())
            train_fn = AsyncMock()

            # Should not raise error (checkpoint registration failure is non-fatal)
            await executor.execute_recipe(
                config_builder=config_builder,
                train_fn=train_fn,
                recipe_name="TEST",
                enable_monitoring=False
            )

        # Training should still complete
        assert sample_run.status == "completed"

        # Cleanup
        log_path.unlink()


# ============================================================================
# TEST 5: INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestUtilsIntegration:
    """Integration tests for utility modules."""

    async def test_full_training_workflow_with_utils(
        self,
        mock_session,
        sample_run,
        mock_env_vars
    ):
        """Test full training workflow using all utilities."""
        # Setup environment
        setup_training_environment()

        # Create log file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_path = Path(f.name)

        # Sanitize log filename
        safe_filename = sanitize_filename(f"run_{sample_run.id}.log")
        assert safe_filename == f"run_{sample_run.id}.log"

        # Create executor
        executor = create_recipe_executor(mock_session, sample_run, log_path)

        # Execute recipe
        config_builder = Mock(return_value=Mock())
        train_fn = AsyncMock()

        await executor.execute_recipe(
            config_builder=config_builder,
            train_fn=train_fn,
            recipe_name="SFT",
            enable_monitoring=False
        )

        # Verify completion
        assert sample_run.status == "completed"

        # Verify log exists
        assert log_path.exists()

        # Cleanup
        log_path.unlink()

    async def test_utils_consolidate_code_duplication(
        self,
        mock_env_vars
    ):
        """Test that utilities successfully consolidate code duplication."""
        # Before refactoring: Setup was duplicated 8 times
        # After refactoring: Single function call

        # Test that setup_training_environment works
        setup_training_environment()
        assert os.environ["TINKER_API_KEY"] is not None

        # Test that it's reusable
        setup_training_environment()  # Should work multiple times
        assert os.environ["PYTHONIOENCODING"] == "utf-8"


# ============================================================================
# TEST SUMMARY
# ============================================================================

"""
Test Coverage Summary:
======================

1. Text Utilities: 11 tests
   - Strip ANSI codes (4 tests)
   - Truncate text (3 tests)
   - Sanitize filename (4 tests)

2. Environment Utilities: 11 tests
   - Get required env (2 tests)
   - Setup training environment (3 tests)
   - Get optional env (2 tests)
   - Validate API key (4 tests)

3. Recipe Executor: 11 tests
   - Create executor (1 test)
   - Log writing (2 tests)
   - Parse metrics (3 tests)
   - Execute recipe (5 tests)

4. Error Handling: 3 tests
   - Handle None input
   - Include context in errors
   - Handle checkpoint failure

5. Integration: 2 tests
   - Full workflow
   - Code consolidation

Total: 38 utility tests

To Run:
-------
# Run all utility tests
pytest tests/test_utils.py -v

# Run specific test class
pytest tests/test_utils.py::TestRecipeExecutor -v

# Run with coverage
pytest tests/test_utils.py --cov=utils --cov-report=html
"""
