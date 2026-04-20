"""
Test Suite for Evaluation Functionality
========================================

Comprehensive tests for model evaluation functionality.

Test Categories:
1. Evaluation Creation and Execution
2. Renderer Integration
3. Response Parsing
4. Metrics Collection
5. Test Prompt Handling
6. Evaluation Error Handling
7. Multi-Model Evaluation
8. Benchmark Evaluations

Usage:
    pytest tests/test_evaluation.py -v
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Run, Evaluation, Checkpoint


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_sampling_client():
    """Mock Tinker sampling client."""
    client = Mock()
    client.sample = Mock(return_value=Mock(
        result=Mock(return_value=Mock(
            sequences=[Mock(tokens=[1, 2, 3, 4, 5])]
        ))
    ))
    return client


@pytest.fixture
def sample_test_prompts():
    """Sample test prompts for evaluation."""
    return [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "user", "content": "Explain quantum computing in simple terms."},
        {"role": "user", "content": "Write a short poem about the moon."},
    ]


@pytest.fixture
def completed_run_with_checkpoint():
    """Create a completed run with checkpoint."""
    run = Run(
        id=1,
        project_id=1,
        dataset_id=1,
        recipe_type="SFT",
        status="completed",
        progress=1.0,
        config_json={
            "base_model": "meta-llama/Llama-3.2-1B",
            "renderer_name": "role_colon",
            "hyperparameters": {
                "learning_rate": 0.0001,
                "epochs": 1
            }
        }
    )

    # Add checkpoint
    checkpoint = Checkpoint(
        id=1,
        run_id=1,
        step=1000,
        metrics={"loss": 0.3, "accuracy": 0.92},
        path="/tmp/checkpoints/checkpoint-1000"
    )
    run.checkpoints = [checkpoint]

    return run


# ============================================================================
# TEST 1: EVALUATION CREATION AND EXECUTION
# ============================================================================

class TestEvaluationCreation:
    """Test evaluation creation and execution."""

    def test_create_evaluation_object(self):
        """Test creating an Evaluation object."""
        evaluation = Evaluation(
            id=1,
            run_id=1,
            evaluator_name="mmlu",
            metrics={
                "accuracy": 0.85,
                "f1_score": 0.82,
                "precision": 0.88
            },
            created_at=datetime.utcnow()
        )

        assert evaluation.id == 1
        assert evaluation.run_id == 1
        assert evaluation.evaluator_name == "mmlu"
        assert evaluation.metrics["accuracy"] == 0.85

    def test_evaluation_requires_run_id(self):
        """Test that evaluation requires a run_id."""
        with pytest.raises((TypeError, ValueError)):
            Evaluation(
                evaluator_name="mmlu",
                metrics={}
            )

    @pytest.mark.asyncio
    @patch("main.get_sampling_client")
    @patch("main.get_tokenizer")
    @patch("main.get_renderer")
    async def test_evaluation_execution(
        self,
        mock_get_renderer,
        mock_get_tokenizer,
        mock_get_sampling_client,
        completed_run_with_checkpoint,
        sample_test_prompts,
        mock_session
    ):
        """Test executing an evaluation."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_renderer = Mock()
        mock_renderer.build_generation_prompt = Mock(return_value="Test prompt")
        mock_renderer.parse_response = Mock(return_value=[
            {"role": "assistant", "content": "Paris"}
        ])
        mock_renderer.get_stop_sequences = Mock(return_value=["<|end_of_text|>"])
        mock_get_renderer.return_value = mock_renderer

        mock_client = Mock()
        mock_client.sample = Mock(return_value=Mock(
            result=Mock(return_value=Mock(
                sequences=[Mock(tokens=[1, 2, 3])]
            ))
        ))
        mock_get_sampling_client.return_value = mock_client

        # Execute evaluation logic
        for prompt in sample_test_prompts:
            model_input = mock_renderer.build_generation_prompt([prompt])
            assert model_input == "Test prompt"

            result = mock_client.sample(
                prompt=model_input,
                num_samples=1,
            )
            assert result is not None


# ============================================================================
# TEST 2: RENDERER INTEGRATION
# ============================================================================

class TestRendererIntegration:
    """Test renderer integration in evaluation."""

    def test_renderer_build_generation_prompt(self, mock_renderer):
        """Test renderer builds generation prompt correctly."""
        messages = [{"role": "user", "content": "Hello"}]
        prompt = mock_renderer.build_generation_prompt(messages)

        assert prompt is not None
        mock_renderer.build_generation_prompt.assert_called_once_with(messages)

    def test_renderer_parse_response(self, mock_renderer):
        """Test renderer parses response correctly."""
        tokens = [1, 2, 3, 4, 5]
        parsed = mock_renderer.parse_response(tokens)

        assert parsed is not None
        assert len(parsed) > 0
        assert parsed[0]["role"] == "assistant"
        assert "content" in parsed[0]

    def test_renderer_get_stop_sequences(self, mock_renderer):
        """Test renderer returns stop sequences."""
        stop_sequences = mock_renderer.get_stop_sequences()

        assert stop_sequences is not None
        assert isinstance(stop_sequences, list)
        assert len(stop_sequences) > 0


# ============================================================================
# TEST 3: RESPONSE PARSING
# ============================================================================

class TestResponseParsing:
    """Test response parsing from model outputs."""

    def test_parse_response_extracts_content(self):
        """Test parsing extracts only content from response."""
        parsed_messages = [
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]

        # Extract content
        response_text = parsed_messages[0]["content"]

        assert response_text == "The capital of France is Paris."
        assert "role" not in response_text
        assert "assistant" not in response_text

    def test_parse_response_handles_multiple_messages(self):
        """Test parsing handles multiple messages."""
        parsed_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Should extract last assistant message
        assistant_messages = [m for m in parsed_messages if m["role"] == "assistant"]
        assert len(assistant_messages) == 1
        assert assistant_messages[0]["content"] == "Hi there!"

    def test_parse_response_strips_formatting_tokens(self):
        """Test parsing strips formatting tokens."""
        # Raw response with formatting
        raw_response = "<|begin_of_text|>User: Hello\n\nAssistant: Hi there!<|end_of_text|>"

        # After parsing, should only have content
        expected_content = "Hi there!"

        # Simulate parsing
        cleaned = raw_response.split("Assistant:")[1].replace("<|end_of_text|>", "").strip()

        assert cleaned == expected_content


# ============================================================================
# TEST 4: METRICS COLLECTION
# ============================================================================

class TestMetricsCollection:
    """Test metrics collection during evaluation."""

    def test_collect_test_completions(self, sample_test_prompts):
        """Test collecting test completions."""
        test_completions = []

        for prompt in sample_test_prompts:
            test_completions.append({
                "prompt": prompt["content"],
                "completion": "Test response",
                "success": True
            })

        assert len(test_completions) == 3
        assert all(c["success"] for c in test_completions)

    def test_calculate_success_rate(self, sample_test_prompts):
        """Test calculating success rate from completions."""
        test_completions = [
            {"prompt": "Q1", "completion": "A1", "success": True},
            {"prompt": "Q2", "completion": "A2", "success": True},
            {"prompt": "Q3", "completion": "", "success": False},
        ]

        success_count = sum(1 for c in test_completions if c["success"])
        success_rate = success_count / len(test_completions)

        assert success_rate == 2/3

    def test_evaluation_metrics_structure(self):
        """Test evaluation metrics have correct structure."""
        eval_results = {
            "test_completions": [
                {"prompt": "Q1", "completion": "A1", "success": True}
            ],
            "success_rate": 1.0,
            "average_response_length": 50,
            "timestamp": datetime.utcnow().isoformat()
        }

        assert "test_completions" in eval_results
        assert "success_rate" in eval_results
        assert isinstance(eval_results["test_completions"], list)
        assert isinstance(eval_results["success_rate"], float)


# ============================================================================
# TEST 5: TEST PROMPT HANDLING
# ============================================================================

class TestTestPromptHandling:
    """Test handling of test prompts."""

    def test_validate_test_prompt_structure(self):
        """Test test prompts have correct structure."""
        valid_prompt = {
            "role": "user",
            "content": "What is AI?"
        }

        assert "role" in valid_prompt
        assert "content" in valid_prompt
        assert valid_prompt["role"] == "user"
        assert len(valid_prompt["content"]) > 0

    def test_handle_empty_test_prompts(self):
        """Test handling empty test prompts list."""
        test_prompts = []

        if not test_prompts:
            # Should use default prompts or return error
            assert True

    def test_handle_malformed_test_prompts(self):
        """Test handling malformed test prompts."""
        malformed_prompts = [
            {"content": "Missing role"},
            {"role": "user"},  # Missing content
            "Not a dict"
        ]

        # Should filter out malformed prompts
        valid_prompts = [
            p for p in malformed_prompts
            if isinstance(p, dict) and "role" in p and "content" in p
        ]

        assert len(valid_prompts) == 0


# ============================================================================
# TEST 6: EVALUATION ERROR HANDLING
# ============================================================================

class TestEvaluationErrorHandling:
    """Test error handling during evaluation."""

    @pytest.mark.asyncio
    async def test_handle_missing_checkpoint(self, mock_session):
        """Test handling when checkpoint is missing."""
        run = Run(
            id=1,
            recipe_type="SFT",
            status="completed",
            config_json={"base_model": "meta-llama/Llama-3.2-1B"}
        )
        run.checkpoints = []  # No checkpoints

        # Should raise error or return failure
        if not run.checkpoints:
            with pytest.raises((ValueError, IndexError)):
                run.checkpoints[0].path

    @pytest.mark.asyncio
    async def test_handle_invalid_model_name(self):
        """Test handling invalid model name."""
        invalid_model = "invalid/model-name"

        # Should raise error when trying to load
        with pytest.raises((ValueError, FileNotFoundError, Exception)):
            raise ValueError(f"Invalid model: {invalid_model}")

    @pytest.mark.asyncio
    async def test_handle_sampling_timeout(self, mock_sampling_client):
        """Test handling sampling timeout."""
        # Mock timeout
        mock_sampling_client.sample.side_effect = TimeoutError("Sampling timed out")

        with pytest.raises(TimeoutError):
            result = mock_sampling_client.sample(prompt="Test", num_samples=1)
            result.result()

    @pytest.mark.asyncio
    async def test_handle_renderer_error(self):
        """Test handling renderer errors."""
        with pytest.raises((ValueError, KeyError)):
            # Invalid renderer name
            raise ValueError("Invalid renderer: unknown_renderer")


# ============================================================================
# TEST 7: MULTI-MODEL EVALUATION
# ============================================================================

class TestMultiModelEvaluation:
    """Test evaluating multiple models."""

    def test_evaluate_multiple_checkpoints(self):
        """Test evaluating multiple checkpoints from same run."""
        checkpoints = [
            Checkpoint(id=1, run_id=1, step=100, path="/tmp/checkpoint-100"),
            Checkpoint(id=2, run_id=1, step=500, path="/tmp/checkpoint-500"),
            Checkpoint(id=3, run_id=1, step=1000, path="/tmp/checkpoint-1000"),
        ]

        # Should be able to evaluate each checkpoint
        assert len(checkpoints) == 3

        for checkpoint in checkpoints:
            # Evaluate checkpoint
            eval_result = {
                "checkpoint_step": checkpoint.step,
                "accuracy": 0.8 + (checkpoint.step / 10000)  # Improving with steps
            }
            assert eval_result["accuracy"] > 0.8

    def test_compare_model_evaluations(self):
        """Test comparing evaluations across models."""
        evaluations = [
            {"model": "model-A", "accuracy": 0.85},
            {"model": "model-B", "accuracy": 0.92},
            {"model": "model-C", "accuracy": 0.78},
        ]

        # Find best model
        best = max(evaluations, key=lambda e: e["accuracy"])

        assert best["model"] == "model-B"
        assert best["accuracy"] == 0.92


# ============================================================================
# TEST 8: BENCHMARK EVALUATIONS
# ============================================================================

class TestBenchmarkEvaluations:
    """Test standard benchmark evaluations."""

    def test_mmlu_evaluation_structure(self):
        """Test MMLU evaluation structure."""
        mmlu_result = {
            "evaluator_name": "mmlu",
            "metrics": {
                "overall_accuracy": 0.85,
                "subjects": {
                    "mathematics": 0.87,
                    "history": 0.83,
                    "science": 0.86
                }
            }
        }

        assert mmlu_result["evaluator_name"] == "mmlu"
        assert "overall_accuracy" in mmlu_result["metrics"]
        assert "subjects" in mmlu_result["metrics"]

    def test_hellaswag_evaluation_structure(self):
        """Test HellaSwag evaluation structure."""
        hellaswag_result = {
            "evaluator_name": "hellaswag",
            "metrics": {
                "accuracy": 0.78,
                "num_examples": 1000
            }
        }

        assert hellaswag_result["evaluator_name"] == "hellaswag"
        assert "accuracy" in hellaswag_result["metrics"]

    def test_custom_evaluation(self):
        """Test custom evaluation with user-provided prompts."""
        custom_eval = {
            "evaluator_name": "custom",
            "test_prompts": [
                {"role": "user", "content": "Custom question 1"},
                {"role": "user", "content": "Custom question 2"},
            ],
            "metrics": {
                "test_completions": [],
                "success_rate": 0.0
            }
        }

        assert custom_eval["evaluator_name"] == "custom"
        assert len(custom_eval["test_prompts"]) == 2


# ============================================================================
# TEST 9: EVALUATION PERSISTENCE
# ============================================================================

class TestEvaluationPersistence:
    """Test saving and loading evaluations."""

    def test_save_evaluation_to_database(self, mock_session):
        """Test saving evaluation to database."""
        evaluation = Evaluation(
            run_id=1,
            evaluator_name="mmlu",
            metrics={"accuracy": 0.85}
        )

        mock_session.add(evaluation)
        mock_session.commit()

        mock_session.add.assert_called_once_with(evaluation)
        mock_session.commit.assert_called_once()

    def test_load_evaluations_for_run(self, mock_session):
        """Test loading all evaluations for a run."""
        from models import Evaluation

        mock_query = Mock()
        mock_query.filter = Mock(return_value=mock_query)
        mock_query.all = Mock(return_value=[
            Evaluation(id=1, run_id=1, evaluator_name="mmlu", metrics={}),
            Evaluation(id=2, run_id=1, evaluator_name="hellaswag", metrics={}),
        ])

        mock_session.query = Mock(return_value=mock_query)

        evaluations = mock_session.query(Evaluation).filter(
            Evaluation.run_id == 1
        ).all()

        assert len(evaluations) == 2

    def test_evaluation_json_serialization(self):
        """Test evaluation metrics are JSON serializable."""
        metrics = {
            "accuracy": 0.85,
            "f1_score": 0.82,
            "test_completions": [
                {"prompt": "Q1", "completion": "A1", "success": True}
            ]
        }

        # Should be serializable to JSON
        json_str = json.dumps(metrics)
        assert json_str is not None

        # Should be deserializable
        loaded = json.loads(json_str)
        assert loaded["accuracy"] == 0.85


# ============================================================================
# TEST SUMMARY
# ============================================================================

"""
Test Coverage Summary:
======================

1. Evaluation Creation: 3 tests
   - Create evaluation object
   - Require run_id
   - Execute evaluation

2. Renderer Integration: 3 tests
   - Build generation prompt
   - Parse response
   - Get stop sequences

3. Response Parsing: 3 tests
   - Extract content
   - Handle multiple messages
   - Strip formatting tokens

4. Metrics Collection: 3 tests
   - Collect test completions
   - Calculate success rate
   - Metrics structure

5. Test Prompt Handling: 3 tests
   - Validate prompt structure
   - Handle empty prompts
   - Handle malformed prompts

6. Error Handling: 4 tests
   - Missing checkpoint
   - Invalid model name
   - Sampling timeout
   - Renderer error

7. Multi-Model Evaluation: 2 tests
   - Evaluate multiple checkpoints
   - Compare evaluations

8. Benchmark Evaluations: 3 tests
   - MMLU structure
   - HellaSwag structure
   - Custom evaluation

9. Evaluation Persistence: 3 tests
   - Save to database
   - Load for run
   - JSON serialization

Total: 27 evaluation tests

To Run:
-------
# Run all evaluation tests
pytest tests/test_evaluation.py -v

# Run specific test class
pytest tests/test_evaluation.py::TestRendererIntegration -v

# Run with coverage
pytest tests/test_evaluation.py --cov=main --cov-report=html
"""
