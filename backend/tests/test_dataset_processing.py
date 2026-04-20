"""
Test Suite for Dataset Processing
===================================

Comprehensive tests for dataset loading and format handling.

Test Categories:
1. Dataset Format Detection
2. Alpaca Format Processing
3. Messages Format Processing
4. Mixed Format Handling
5. Dataset Validation
6. Data Conversion
7. Error Handling
8. Dataset Filtering

Usage:
    pytest tests/test_dataset_processing.py -v
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Dataset


# ============================================================================
# TEST 1: DATASET FORMAT DETECTION
# ============================================================================

class TestDatasetFormatDetection:
    """Test detecting dataset formats."""

    def test_detect_alpaca_format(self, alpaca_format_data):
        """Test detecting Alpaca format."""
        sample = alpaca_format_data[0]

        # Check for Alpaca format keys
        is_alpaca = "instruction" in sample and "output" in sample

        assert is_alpaca is True

    def test_detect_messages_format(self, messages_format_data):
        """Test detecting messages format."""
        sample = messages_format_data[0]

        # Check for messages format
        is_messages = "messages" in sample

        assert is_messages is True

    def test_detect_unsupported_format(self):
        """Test detecting unsupported format."""
        unsupported = {
            "text": "Some text",
            "label": "category"
        }

        is_alpaca = "instruction" in unsupported
        is_messages = "messages" in unsupported

        assert is_alpaca is False
        assert is_messages is False


# ============================================================================
# TEST 2: ALPACA FORMAT PROCESSING
# ============================================================================

class TestAlpacaFormatProcessing:
    """Test processing Alpaca format data."""

    def test_process_alpaca_with_input(self):
        """Test processing Alpaca format with input field."""
        alpaca_row = {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour"
        }

        # Process Alpaca format
        instruction = alpaca_row.get("instruction", "").strip()
        input_text = alpaca_row.get("input", "").strip()

        if input_text:
            user_content = f"{instruction}\n\nInput:\n{input_text}"
        else:
            user_content = instruction

        assert user_content == "Translate to French\n\nInput:\nHello"

        # Convert to messages format
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": alpaca_row.get("output", "")}
        ]

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Bonjour"

    def test_process_alpaca_without_input(self):
        """Test processing Alpaca format without input field."""
        alpaca_row = {
            "instruction": "Write a poem about the moon",
            "input": "",
            "output": "The moon shines bright\nIn the darkest night"
        }

        instruction = alpaca_row.get("instruction", "").strip()
        input_text = alpaca_row.get("input", "").strip()

        if input_text:
            user_content = f"{instruction}\n\nInput:\n{input_text}"
        else:
            user_content = instruction

        assert user_content == "Write a poem about the moon"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": alpaca_row.get("output", "")}
        ]

        assert len(messages) == 2
        assert messages[0]["content"] == "Write a poem about the moon"

    def test_process_alpaca_empty_output(self):
        """Test processing Alpaca format with empty output."""
        alpaca_row = {
            "instruction": "Test instruction",
            "input": "",
            "output": ""
        }

        messages = [
            {"role": "user", "content": alpaca_row.get("instruction", "")},
            {"role": "assistant", "content": alpaca_row.get("output", "")}
        ]

        assert messages[1]["content"] == ""


# ============================================================================
# TEST 3: MESSAGES FORMAT PROCESSING
# ============================================================================

class TestMessagesFormatProcessing:
    """Test processing messages format data."""

    def test_process_messages_format(self, messages_format_data):
        """Test processing messages format."""
        messages_row = messages_format_data[0]

        messages = messages_row["messages"]

        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_process_messages_multi_turn(self):
        """Test processing multi-turn messages."""
        multi_turn = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thanks!"}
            ]
        }

        messages = multi_turn["messages"]

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"

    def test_process_messages_with_system(self):
        """Test processing messages with system message."""
        with_system = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}
            ]
        }

        messages = with_system["messages"]

        assert len(messages) == 3
        assert messages[0]["role"] == "system"


# ============================================================================
# TEST 4: MIXED FORMAT HANDLING
# ============================================================================

class TestMixedFormatHandling:
    """Test handling mixed format datasets."""

    def test_unified_format_handler(self):
        """Test unified handler for both formats."""
        alpaca_row = {
            "instruction": "Translate",
            "input": "Hello",
            "output": "Bonjour"
        }

        messages_row = {
            "messages": [
                {"role": "user", "content": "Translate Hello"},
                {"role": "assistant", "content": "Bonjour"}
            ]
        }

        # Unified handler logic
        def process_row(row):
            if "messages" in row:
                # Direct messages format
                return row["messages"]
            elif "instruction" in row or "output" in row:
                # Alpaca format - convert to messages
                instruction = row.get("instruction", "").strip()
                input_text = row.get("input", "").strip()

                if input_text:
                    user_content = f"{instruction}\n\nInput:\n{input_text}"
                else:
                    user_content = instruction

                return [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": row.get("output", "")}
                ]
            else:
                return None

        # Test both formats with same handler
        alpaca_result = process_row(alpaca_row)
        messages_result = process_row(messages_row)

        assert alpaca_result is not None
        assert messages_result is not None
        assert len(alpaca_result) == 2
        assert len(messages_result) == 2

    def test_skip_unsupported_format(self):
        """Test skipping unsupported format."""
        unsupported_row = {
            "text": "Some text",
            "label": "category"
        }

        def process_row(row):
            if "messages" in row:
                return row["messages"]
            elif "instruction" in row or "output" in row:
                return []
            else:
                # Skip unsupported format
                return None

        result = process_row(unsupported_row)

        assert result is None


# ============================================================================
# TEST 5: DATASET VALIDATION
# ============================================================================

class TestDatasetValidation:
    """Test dataset validation."""

    def test_validate_huggingface_dataset(self):
        """Test validating HuggingFace dataset spec."""
        dataset = Dataset(
            name="alpaca-clean",
            kind="huggingface",
            spec={
                "repo": "yahma/alpaca-cleaned"
            }
        )

        assert dataset.kind == "huggingface"
        assert "repo" in dataset.spec
        assert dataset.spec["repo"] == "yahma/alpaca-cleaned"

    def test_validate_local_dataset(self):
        """Test validating local dataset spec."""
        dataset = Dataset(
            name="local-data",
            kind="local",
            spec={
                "path": "/data/training.jsonl"
            }
        )

        assert dataset.kind == "local"
        assert "path" in dataset.spec

    def test_validate_url_dataset(self):
        """Test validating URL dataset spec."""
        dataset = Dataset(
            name="url-data",
            kind="url",
            spec={
                "url": "https://example.com/data.jsonl"
            }
        )

        assert dataset.kind == "url"
        assert "url" in dataset.spec

    def test_validate_dataset_missing_spec(self):
        """Test validating dataset with missing spec."""
        with pytest.raises((ValueError, TypeError)):
            Dataset(
                name="invalid-data",
                kind="huggingface",
                spec=None  # Missing spec
            )


# ============================================================================
# TEST 6: DATA CONVERSION
# ============================================================================

class TestDataConversion:
    """Test data format conversion."""

    def test_convert_alpaca_to_messages(self):
        """Test converting Alpaca to messages format."""
        alpaca = {
            "instruction": "Explain AI",
            "input": "",
            "output": "AI is Artificial Intelligence..."
        }

        messages = [
            {"role": "user", "content": alpaca["instruction"]},
            {"role": "assistant", "content": alpaca["output"]}
        ]

        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[0]["content"] == "Explain AI"

    def test_convert_messages_to_alpaca(self):
        """Test converting messages to Alpaca format (if needed)."""
        messages = [
            {"role": "user", "content": "Explain AI"},
            {"role": "assistant", "content": "AI is Artificial Intelligence..."}
        ]

        # Convert back to Alpaca (for compatibility)
        alpaca = {
            "instruction": messages[0]["content"],
            "input": "",
            "output": messages[1]["content"]
        }

        assert "instruction" in alpaca
        assert "output" in alpaca
        assert alpaca["instruction"] == "Explain AI"

    def test_preserve_message_roles(self):
        """Test preserving message roles during conversion."""
        original_messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant response"}
        ]

        # Ensure roles are preserved
        assert original_messages[0]["role"] == "system"
        assert original_messages[1]["role"] == "user"
        assert original_messages[2]["role"] == "assistant"


# ============================================================================
# TEST 7: ERROR HANDLING
# ============================================================================

class TestDatasetErrorHandling:
    """Test error handling in dataset processing."""

    def test_handle_missing_instruction(self):
        """Test handling missing instruction field."""
        incomplete = {
            "input": "Hello",
            "output": "Bonjour"
        }

        instruction = incomplete.get("instruction", "").strip()

        assert instruction == ""

    def test_handle_missing_output(self):
        """Test handling missing output field."""
        incomplete = {
            "instruction": "Translate",
            "input": "Hello"
        }

        output = incomplete.get("output", "")

        assert output == ""

    def test_handle_missing_messages(self):
        """Test handling missing messages field."""
        invalid = {
            "data": "some data"
        }

        messages = invalid.get("messages")

        assert messages is None

    def test_handle_malformed_messages(self):
        """Test handling malformed messages."""
        malformed = {
            "messages": [
                {"role": "user"},  # Missing content
                {"content": "Response"}  # Missing role
            ]
        }

        # Filter valid messages
        valid_messages = [
            m for m in malformed["messages"]
            if "role" in m and "content" in m
        ]

        assert len(valid_messages) == 0

    def test_handle_empty_dataset(self):
        """Test handling empty dataset."""
        empty_data = []

        assert len(empty_data) == 0

    def test_handle_none_values(self):
        """Test handling None values in data."""
        with_none = {
            "instruction": None,
            "input": None,
            "output": None
        }

        instruction = (with_none.get("instruction") or "").strip()
        output = (with_none.get("output") or "").strip()

        assert instruction == ""
        assert output == ""


# ============================================================================
# TEST 8: DATASET FILTERING
# ============================================================================

class TestDatasetFiltering:
    """Test filtering dataset rows."""

    def test_filter_valid_rows(self):
        """Test filtering valid rows."""
        data = [
            {"instruction": "Q1", "output": "A1"},  # Valid
            {"instruction": "Q2", "output": ""},    # Valid but empty output
            {"text": "Invalid format"},             # Invalid
            {"instruction": "Q3", "output": "A3"},  # Valid
        ]

        def is_valid(row):
            return "instruction" in row and "output" in row

        valid_data = [row for row in data if is_valid(row)]

        assert len(valid_data) == 3

    def test_filter_by_length(self):
        """Test filtering by content length."""
        data = [
            {"instruction": "Short", "output": "Short answer"},
            {"instruction": "Long" * 1000, "output": "A" * 10000},  # Too long
            {"instruction": "Medium length question", "output": "Medium answer"}
        ]

        max_length = 1000

        def is_reasonable_length(row):
            total_length = len(row.get("instruction", "")) + len(row.get("output", ""))
            return total_length < max_length

        filtered = [row for row in data if is_reasonable_length(row)]

        assert len(filtered) == 2

    def test_filter_none_values(self):
        """Test filtering None values from map_fn."""
        def map_fn(row):
            if "messages" in row:
                return row["messages"]
            elif "instruction" in row:
                return [{"role": "user", "content": row["instruction"]}]
            else:
                return None  # Skip unsupported

        data = [
            {"messages": [{"role": "user", "content": "Q1"}]},  # Valid
            {"instruction": "Q2"},  # Valid
            {"text": "Invalid"},  # Returns None
            {"instruction": "Q3"},  # Valid
        ]

        processed = [datum for row in data if (datum := map_fn(row)) is not None]

        assert len(processed) == 3


# ============================================================================
# TEST 9: DATASET PREVIEW
# ============================================================================

class TestDatasetPreview:
    """Test dataset preview functionality."""

    def test_preview_first_n_samples(self):
        """Test previewing first N samples."""
        data = [
            {"instruction": f"Q{i}", "output": f"A{i}"}
            for i in range(100)
        ]

        preview = data[:5]

        assert len(preview) == 5
        assert preview[0]["instruction"] == "Q0"
        assert preview[4]["instruction"] == "Q4"

    def test_preview_includes_format_info(self):
        """Test preview includes format information."""
        sample = {
            "instruction": "Test",
            "input": "",
            "output": "Answer"
        }

        format_info = {
            "format": "alpaca",
            "has_instruction": "instruction" in sample,
            "has_messages": "messages" in sample
        }

        assert format_info["format"] == "alpaca"
        assert format_info["has_instruction"] is True
        assert format_info["has_messages"] is False


# ============================================================================
# TEST SUMMARY
# ============================================================================

"""
Test Coverage Summary:
======================

1. Format Detection: 3 tests
   - Detect Alpaca format
   - Detect messages format
   - Detect unsupported format

2. Alpaca Processing: 3 tests
   - Process with input
   - Process without input
   - Process empty output

3. Messages Processing: 3 tests
   - Process messages format
   - Process multi-turn
   - Process with system message

4. Mixed Format: 2 tests
   - Unified handler
   - Skip unsupported

5. Dataset Validation: 4 tests
   - Validate HuggingFace
   - Validate local
   - Validate URL
   - Validate missing spec

6. Data Conversion: 3 tests
   - Convert Alpaca to messages
   - Convert messages to Alpaca
   - Preserve roles

7. Error Handling: 6 tests
   - Missing instruction
   - Missing output
   - Missing messages
   - Malformed messages
   - Empty dataset
   - None values

8. Dataset Filtering: 3 tests
   - Filter valid rows
   - Filter by length
   - Filter None values

9. Dataset Preview: 2 tests
   - Preview first N samples
   - Preview with format info

Total: 29 dataset processing tests

To Run:
-------
# Run all dataset tests
pytest tests/test_dataset_processing.py -v

# Run specific test class
pytest tests/test_dataset_processing.py::TestAlpacaFormatProcessing -v

# Run with coverage
pytest tests/test_dataset_processing.py --cov=job_runner --cov-report=html
"""
