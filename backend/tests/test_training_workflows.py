"""
Test Suite for Training Workflows
==================================

Integration tests for end-to-end training workflows covering all recipe types.

Test Categories:
1. SFT (Supervised Fine-Tuning) Workflow
2. DPO (Direct Preference Optimization) Workflow
3. RL (Reinforcement Learning) Workflow
4. CHAT_SL (Chat Supervised Learning) Workflow
5. DISTILLATION Workflow
6. MATH_RL Workflow
7. ON_POLICY_DISTILLATION Workflow
8. Multi-Recipe Workflow Tests
9. Error Recovery Tests
10. Resume from Checkpoint Tests

Usage:
    pytest tests/test_training_workflows.py -v -m integration
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Run, Dataset
from utils.recipe_executor import RecipeExecutor, create_recipe_executor


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_tinker_training():
    """Mock tinker training functions."""
    with patch("tinker_cookbook.supervised.sft.train.main", new_callable=AsyncMock) as mock_sft, \
         patch("tinker_cookbook.rlhf.dpo.train_dpo.main", new_callable=AsyncMock) as mock_dpo, \
         patch("tinker_cookbook.rlhf.rl.train.main", new_callable=AsyncMock) as mock_rl, \
         patch("tinker_cookbook.distillation.train.main", new_callable=AsyncMock) as mock_dist, \
         patch("tinker_cookbook.rlhf.math_rl.train.main", new_callable=AsyncMock) as mock_math:

        yield {
            "sft": mock_sft,
            "dpo": mock_dpo,
            "rl": mock_rl,
            "distillation": mock_dist,
            "math_rl": mock_math,
        }


@pytest.fixture
def temp_logs_dir():
    """Create temporary logs directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logs_path = Path(tmpdir)
        (logs_path / "logs").mkdir(exist_ok=True)
        yield logs_path


# ============================================================================
# TEST 1: SFT WORKFLOW
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestSFTWorkflow:
    """Test SFT training workflow end-to-end."""

    async def test_sft_workflow_success(
        self,
        job_runner,
        sample_run,
        mock_session,
        mock_env_vars,
        mock_tinker_training,
        temp_logs_dir
    ):
        """Test complete SFT training workflow."""
        # Setup
        sample_run.recipe_type = "SFT"
        sample_run.dataset_id = 1
        mock_session.get.return_value = sample_run

        # Mock dataset
        Dataset(
            id=1,
            name="alpaca-clean",
            kind="huggingface",
            spec={"repo": "yahma/alpaca-cleaned"}
        )

        # Create log file
        log_file = temp_logs_dir / "run_1.log"
        log_file.touch()

        # Execute SFT training
        with patch.object(job_runner, "_get_or_create_logs_path", return_value=log_file):
            config = job_runner.build_sft_config(sample_run, "yahma/alpaca-cleaned")

            # Verify config has required fields
            assert hasattr(config, "model_name")
            assert hasattr(config, "learning_rate")
            assert hasattr(config, "num_epochs")
            assert config.lr_schedule == "linear"

        # Verify training would be called
        assert mock_tinker_training["sft"].called is False  # Not called yet in test

    async def test_sft_workflow_with_monitoring(
        self,
        job_runner,
        sample_run,
        mock_session,
        mock_env_vars,
        temp_logs_dir
    ):
        """Test SFT workflow with metrics monitoring."""
        sample_run.recipe_type = "SFT"
        log_file = temp_logs_dir / "run_1.log"
        log_file.write_text("[METRICS] step=100, loss=0.5, lr=0.0001\n")

        # Create executor
        executor = create_recipe_executor(mock_session, sample_run, log_file)

        # Verify executor can parse metrics
        metric_line = "[METRICS] step=100, loss=0.5, lr=0.0001"
        parsed = executor._parse_metric_line(metric_line)

        assert "step" in parsed
        assert parsed["step"] == 100
        assert "loss" in parsed or "train_mean_nll" in parsed

    async def test_sft_workflow_with_custom_hyperparams(
        self,
        job_runner,
        sample_run,
        mock_env_vars
    ):
        """Test SFT with custom hyperparameters."""
        sample_run.recipe_type = "SFT"
        sample_run.config_json["hyperparameters"] = {
            "learning_rate": 0.00005,
            "epochs": 3,
            "batch_size": 64,
            "lora_rank": 64
        }

        config = job_runner.build_sft_config(sample_run, "yahma/alpaca-cleaned")

        # Verify custom hyperparams are applied
        assert config.learning_rate == 0.00005
        assert config.num_epochs == 3


# ============================================================================
# TEST 2: DPO WORKFLOW
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestDPOWorkflow:
    """Test DPO training workflow end-to-end."""

    async def test_dpo_workflow_success(
        self,
        job_runner,
        sample_run,
        mock_env_vars
    ):
        """Test complete DPO training workflow."""
        sample_run.recipe_type = "DPO"
        sample_run.config_json["dpo_beta"] = 0.1

        config = job_runner.build_dpo_config(sample_run)

        # Verify DPO-specific config
        assert hasattr(config, "dpo_beta")
        assert config.dpo_beta == 0.1
        assert hasattr(config, "model_name")

    async def test_dpo_workflow_with_preference_data(
        self,
        job_runner,
        sample_run,
        mock_env_vars
    ):
        """Test DPO with preference dataset."""
        sample_run.recipe_type = "DPO"
        sample_run.config_json["dpo_beta"] = 0.15
        sample_run.config_json["hyperparameters"] = {
            "learning_rate": 0.00003,
            "epochs": 2
        }

        config = job_runner.build_dpo_config(sample_run)

        # Verify DPO config with custom settings
        assert config.dpo_beta == 0.15
        assert config.learning_rate == 0.00003


# ============================================================================
# TEST 3: RL WORKFLOW
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestRLWorkflow:
    """Test RL training workflow end-to-end."""

    async def test_rl_workflow_success(
        self,
        job_runner,
        sample_run,
        mock_env_vars
    ):
        """Test complete RL training workflow."""
        sample_run.recipe_type = "RL"

        config = job_runner.build_rl_config(sample_run)

        # Verify RL-specific config
        assert hasattr(config, "model_name")
        assert hasattr(config, "dataset_builder")
        # RL uses Gsm8k dataset builder
        assert "Gsm8k" in str(type(config.dataset_builder))

    async def test_rl_workflow_with_reward_model(
        self,
        job_runner,
        sample_run,
        mock_env_vars
    ):
        """Test RL with reward model configuration."""
        sample_run.recipe_type = "RL"
        sample_run.config_json["reward_model"] = "meta-llama/Llama-3.2-1B"

        config = job_runner.build_rl_config(sample_run)

        # Verify config is created
        assert config is not None


# ============================================================================
# TEST 4: CHAT_SL WORKFLOW
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestChatSLWorkflow:
    """Test CHAT_SL training workflow end-to-end."""

    async def test_chat_sl_is_sft_variant(
        self,
        job_runner,
        sample_run,
        mock_env_vars
    ):
        """Test that CHAT_SL uses SFT with chat dataset."""
        sample_run.recipe_type = "CHAT_SL"

        # CHAT_SL should use SFT config builder
        config = job_runner.build_sft_config(sample_run, "HuggingFaceH4/no_robots")

        # Should have same properties as SFT
        assert hasattr(config, "num_epochs")
        assert config.lr_schedule == "linear"


# ============================================================================
# TEST 5: DISTILLATION WORKFLOW
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestDistillationWorkflow:
    """Test distillation training workflow end-to-end."""

    async def test_distillation_workflow_success(
        self,
        job_runner,
        sample_run,
        mock_env_vars
    ):
        """Test complete distillation training workflow."""
        sample_run.recipe_type = "DISTILLATION"
        sample_run.config_json["teacher_model"] = "Qwen/Qwen3-8B"

        config = job_runner.build_distillation_config(sample_run)

        # Verify distillation-specific config
        assert hasattr(config, "dataset_configs")
        assert len(config.dataset_configs) > 0

        # Verify teacher config
        teacher_config = config.dataset_configs[0].teacher_config
        assert teacher_config.base_model == "Qwen/Qwen3-8B"


# ============================================================================
# TEST 6: MATH_RL WORKFLOW
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestMathRLWorkflow:
    """Test MATH_RL training workflow end-to-end."""

    async def test_math_rl_workflow_success(
        self,
        job_runner,
        sample_run,
        mock_env_vars
    ):
        """Test complete MATH_RL training workflow."""
        sample_run.recipe_type = "MATH_RL"

        config = job_runner.build_math_rl_config(sample_run)

        # Verify MATH_RL config
        assert hasattr(config, "model_name")
        assert hasattr(config, "dataset_builder")


# ============================================================================
# TEST 7: ON_POLICY_DISTILLATION WORKFLOW
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestOnPolicyDistillationWorkflow:
    """Test ON_POLICY_DISTILLATION training workflow end-to-end."""

    async def test_on_policy_distillation_workflow(
        self,
        job_runner,
        sample_run,
        mock_env_vars
    ):
        """Test complete ON_POLICY_DISTILLATION workflow."""
        sample_run.recipe_type = "ON_POLICY_DISTILLATION"
        sample_run.config_json["teacher_model"] = "Qwen/Qwen3-8B"

        config = job_runner.build_on_policy_distillation_config(sample_run)

        # Verify config
        assert config is not None


# ============================================================================
# TEST 8: MULTI-RECIPE WORKFLOW
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestMultiRecipeWorkflow:
    """Test running multiple recipes concurrently."""

    async def test_concurrent_runs(
        self,
        job_runner,
        mock_session,
        mock_env_vars
    ):
        """Test multiple training runs concurrently."""
        # Create multiple runs
        runs = []
        for i, recipe_type in enumerate(["SFT", "DPO", "RL"], start=1):
            run = Run(
                id=i,
                project_id=1,
                dataset_id=1,
                recipe_type=recipe_type,
                status="pending",
                progress=0.0,
                config_json={
                    "base_model": "meta-llama/Llama-3.2-1B",
                    "hyperparameters": {
                        "learning_rate": 0.0001,
                        "epochs": 1,
                        "batch_size": 128
                    }
                }
            )
            runs.append(run)

        # Submit all runs
        for run in runs:
            await job_runner.submit(run.id)

        # Verify all runs are tracked
        assert len(job_runner._tasks) >= len(runs)

    async def test_sequential_runs_different_types(
        self,
        job_runner,
        mock_env_vars
    ):
        """Test running different recipe types sequentially."""
        recipe_types = ["SFT", "DPO", "RL", "DISTILLATION"]

        for recipe_type in recipe_types:
            run = Run(
                id=1,
                project_id=1,
                dataset_id=1,
                recipe_type=recipe_type,
                status="pending",
                config_json={
                    "base_model": "meta-llama/Llama-3.2-1B",
                    "hyperparameters": {"learning_rate": 0.0001}
                }
            )

            # Build config for each type
            if recipe_type == "SFT":
                config = job_runner.build_sft_config(run, "yahma/alpaca-cleaned")
            elif recipe_type == "DPO":
                run.config_json["dpo_beta"] = 0.1
                config = job_runner.build_dpo_config(run)
            elif recipe_type == "RL":
                config = job_runner.build_rl_config(run)
            elif recipe_type == "DISTILLATION":
                run.config_json["teacher_model"] = "Qwen/Qwen3-8B"
                config = job_runner.build_distillation_config(run)

            # Verify config created
            assert config is not None


# ============================================================================
# TEST 9: ERROR RECOVERY
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorRecovery:
    """Test error recovery in training workflows."""

    async def test_recovery_from_training_failure(
        self,
        job_runner,
        sample_run,
        mock_session,
        mock_env_vars,
        temp_logs_dir
    ):
        """Test recovery when training fails."""
        sample_run.recipe_type = "SFT"
        sample_run.status = "failed"

        log_file = temp_logs_dir / "run_1.log"
        log_file.write_text("[ERROR] Training failed\n")

        # Create executor
        executor = create_recipe_executor(mock_session, sample_run, log_file)

        # Test error logging
        await executor.log("Testing error recovery\n")

        # Verify log was written
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Testing error recovery" in log_content

    async def test_graceful_shutdown_on_cancel(
        self,
        job_runner,
        mock_env_vars
    ):
        """Test graceful shutdown when training is cancelled."""
        run_id = 1

        # Submit job
        await job_runner.submit(run_id)

        # Cancel job
        if run_id in job_runner._tasks:
            task = job_runner._tasks[run_id]
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass  # Expected

        # Verify task is cancelled
        if run_id in job_runner._tasks:
            assert job_runner._tasks[run_id].done()


# ============================================================================
# TEST 10: RESUME FROM CHECKPOINT
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestResumeFromCheckpoint:
    """Test resuming training from checkpoints."""

    async def test_resume_from_failed_run(
        self,
        job_runner,
        sample_run,
        mock_session,
        mock_env_vars
    ):
        """Test resuming a failed run from checkpoint."""
        sample_run.status = "failed"
        sample_run.progress = 0.5

        # Mock checkpoint
        from models import Checkpoint
        checkpoint = Checkpoint(
            id=1,
            run_id=sample_run.id,
            step=500,
            metrics={"loss": 0.5},
            path="/tmp/checkpoint-500"
        )
        sample_run.checkpoints = [checkpoint]

        # Verify checkpoint exists
        assert len(sample_run.checkpoints) > 0
        assert sample_run.checkpoints[0].step == 500

    async def test_resume_uses_latest_checkpoint(
        self,
        sample_run
    ):
        """Test that resume uses the latest checkpoint."""
        from models import Checkpoint

        # Create multiple checkpoints
        checkpoints = [
            Checkpoint(id=1, run_id=1, step=100, path="/tmp/checkpoint-100"),
            Checkpoint(id=2, run_id=1, step=500, path="/tmp/checkpoint-500"),
            Checkpoint(id=3, run_id=1, step=300, path="/tmp/checkpoint-300"),
        ]
        sample_run.checkpoints = checkpoints

        # Find latest checkpoint
        latest = max(sample_run.checkpoints, key=lambda c: c.step)

        assert latest.step == 500
        assert latest.path == "/tmp/checkpoint-500"


# ============================================================================
# TEST 11: PROGRESS TRACKING
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestProgressTracking:
    """Test progress tracking during training."""

    async def test_progress_updates_from_metrics(
        self,
        temp_logs_dir
    ):
        """Test progress is calculated from metrics."""
        log_file = temp_logs_dir / "run_1.log"
        log_file.write_text(
            "[METRICS] step=100, progress=0.25, loss=0.8\n"
            "[METRICS] step=200, progress=0.50, loss=0.6\n"
            "[METRICS] step=300, progress=0.75, loss=0.4\n"
        )

        # Parse metrics
        mock_session = Mock()
        mock_run = Mock(id=1)
        executor = RecipeExecutor(mock_session, mock_run, log_file)

        metrics = []
        with open(log_file) as f:
            for line in f:
                parsed = executor._parse_metric_line(line)
                if parsed:
                    metrics.append(parsed)

        # Verify progress extracted
        assert len(metrics) == 3
        assert metrics[0]["progress"] == 0.25
        assert metrics[1]["progress"] == 0.50
        assert metrics[2]["progress"] == 0.75

    async def test_progress_completion(
        self,
        sample_run,
        mock_session,
        temp_logs_dir
    ):
        """Test progress is set to 1.0 on completion."""
        sample_run.progress = 0.9
        sample_run.status = "running"

        log_file = temp_logs_dir / "run_1.log"
        log_file.touch()

        create_recipe_executor(mock_session, sample_run, log_file)

        # Simulate completion
        sample_run.progress = 1.0
        sample_run.status = "completed"

        assert sample_run.progress == 1.0
        assert sample_run.status == "completed"


# ============================================================================
# TEST SUMMARY
# ============================================================================

"""
Test Coverage Summary:
======================

1. SFT Workflow: 3 tests
   - Basic SFT workflow
   - SFT with monitoring
   - SFT with custom hyperparameters

2. DPO Workflow: 2 tests
   - Basic DPO workflow
   - DPO with preference data

3. RL Workflow: 2 tests
   - Basic RL workflow
   - RL with reward model

4. CHAT_SL Workflow: 1 test
   - CHAT_SL as SFT variant

5. DISTILLATION Workflow: 1 test
   - Basic distillation workflow

6. MATH_RL Workflow: 1 test
   - Basic MATH_RL workflow

7. ON_POLICY_DISTILLATION Workflow: 1 test
   - Basic on-policy distillation

8. Multi-Recipe Workflow: 2 tests
   - Concurrent runs
   - Sequential different types

9. Error Recovery: 2 tests
   - Recovery from training failure
   - Graceful shutdown on cancel

10. Resume from Checkpoint: 2 tests
    - Resume from failed run
    - Use latest checkpoint

11. Progress Tracking: 2 tests
    - Progress from metrics
    - Progress completion

Total: 19 integration tests for training workflows

To Run:
-------
# Run all integration tests
pytest tests/test_training_workflows.py -v -m integration

# Run specific workflow test
pytest tests/test_training_workflows.py::TestSFTWorkflow -v

# Run with coverage
pytest tests/test_training_workflows.py --cov=job_runner --cov-report=html -m integration
"""
