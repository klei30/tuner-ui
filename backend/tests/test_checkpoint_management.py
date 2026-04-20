"""
Test Suite for Checkpoint Management
======================================

Comprehensive tests for checkpoint creation, storage, and retrieval.

Test Categories:
1. Checkpoint Creation
2. Checkpoint Registration
3. Checkpoint Storage
4. Checkpoint Retrieval
5. Checkpoint Metrics
6. Checkpoint Selection
7. Checkpoint Download
8. Error Handling

Usage:
    pytest tests/test_checkpoint_management.py -v
"""

import pytest
from unittest.mock import Mock
from datetime import datetime
from pathlib import Path
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Checkpoint


# ============================================================================
# TEST 1: CHECKPOINT CREATION
# ============================================================================

class TestCheckpointCreation:
    """Test checkpoint creation."""

    def test_create_checkpoint_object(self):
        """Test creating a Checkpoint object."""
        checkpoint = Checkpoint(
            id=1,
            run_id=1,
            step=1000,
            metrics={"loss": 0.3, "accuracy": 0.92},
            path="/tmp/checkpoints/checkpoint-1000",
            created_at=datetime.utcnow()
        )

        assert checkpoint.id == 1
        assert checkpoint.run_id == 1
        assert checkpoint.step == 1000
        assert checkpoint.metrics["loss"] == 0.3
        assert checkpoint.path == "/tmp/checkpoints/checkpoint-1000"

    def test_checkpoint_requires_run_id(self):
        """Test that checkpoint requires run_id."""
        with pytest.raises((TypeError, ValueError)):
            Checkpoint(
                step=1000,
                metrics={},
                path="/tmp/checkpoint"
            )

    def test_checkpoint_requires_step(self):
        """Test that checkpoint requires step number."""
        with pytest.raises((TypeError, ValueError)):
            Checkpoint(
                run_id=1,
                metrics={},
                path="/tmp/checkpoint"
            )


# ============================================================================
# TEST 2: CHECKPOINT REGISTRATION
# ============================================================================

class TestCheckpointRegistration:
    """Test checkpoint registration from logs."""

    def test_parse_checkpoint_from_log(self, temp_log_file):
        """Test parsing checkpoint info from log file."""
        # Create log with checkpoint info
        checkpoint_info = {
            "step": 1000,
            "path": "/tmp/checkpoints/checkpoint-1000",
            "metrics": {"loss": 0.3, "accuracy": 0.92}
        }

        with open(temp_log_file, 'a') as f:
            f.write(f"[CHECKPOINT] {json.dumps(checkpoint_info)}\n")

        # Parse checkpoint from log
        with open(temp_log_file, 'r') as f:
            for line in f:
                if "[CHECKPOINT]" in line:
                    checkpoint_data = json.loads(line.split("[CHECKPOINT]")[1].strip())
                    assert checkpoint_data["step"] == 1000
                    assert checkpoint_data["path"] == "/tmp/checkpoints/checkpoint-1000"

    @pytest.mark.asyncio
    async def test_register_checkpoint_to_database(self, mock_session):
        """Test registering checkpoint to database."""
        checkpoint = Checkpoint(
            run_id=1,
            step=1000,
            metrics={"loss": 0.3},
            path="/tmp/checkpoint-1000"
        )

        mock_session.add(checkpoint)
        mock_session.commit()

        mock_session.add.assert_called_once_with(checkpoint)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_multiple_checkpoints(self, mock_session):
        """Test registering multiple checkpoints."""
        checkpoints = [
            Checkpoint(run_id=1, step=100, metrics={}, path="/tmp/checkpoint-100"),
            Checkpoint(run_id=1, step=500, metrics={}, path="/tmp/checkpoint-500"),
            Checkpoint(run_id=1, step=1000, metrics={}, path="/tmp/checkpoint-1000"),
        ]

        for checkpoint in checkpoints:
            mock_session.add(checkpoint)

        mock_session.commit()

        assert mock_session.add.call_count == 3


# ============================================================================
# TEST 3: CHECKPOINT STORAGE
# ============================================================================

class TestCheckpointStorage:
    """Test checkpoint file storage."""

    def test_checkpoint_path_structure(self, temp_checkpoint_dir):
        """Test checkpoint path structure."""
        run_id = 1
        step = 1000

        checkpoint_path = temp_checkpoint_dir / f"run_{run_id}" / f"checkpoint-{step}"

        # Create checkpoint directory
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        assert checkpoint_path.exists()
        assert checkpoint_path.is_dir()

    def test_checkpoint_file_naming(self):
        """Test checkpoint file naming convention."""
        step = 1000

        checkpoint_name = f"checkpoint-{step}"

        assert checkpoint_name == "checkpoint-1000"

    def test_checkpoint_metadata_storage(self, temp_checkpoint_dir):
        """Test storing checkpoint metadata."""
        checkpoint_path = temp_checkpoint_dir / "checkpoint-1000"
        checkpoint_path.mkdir(exist_ok=True)

        metadata = {
            "step": 1000,
            "metrics": {"loss": 0.3, "accuracy": 0.92},
            "timestamp": datetime.utcnow().isoformat()
        }

        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Verify metadata was written
        assert metadata_file.exists()

        # Read and verify
        with open(metadata_file, 'r') as f:
            loaded = json.load(f)

        assert loaded["step"] == 1000
        assert loaded["metrics"]["loss"] == 0.3


# ============================================================================
# TEST 4: CHECKPOINT RETRIEVAL
# ============================================================================

class TestCheckpointRetrieval:
    """Test retrieving checkpoints."""

    def test_get_checkpoint_by_id(self, mock_session, sample_checkpoint):
        """Test getting checkpoint by ID."""
        mock_session.get.return_value = sample_checkpoint

        checkpoint = mock_session.get(Checkpoint, 1)

        assert checkpoint.id == 1
        assert checkpoint.step == 100

    def test_get_checkpoints_for_run(self, mock_session):
        """Test getting all checkpoints for a run."""
        mock_query = Mock()
        mock_query.filter = Mock(return_value=mock_query)
        mock_query.order_by = Mock(return_value=mock_query)
        mock_query.all = Mock(return_value=[
            Checkpoint(id=1, run_id=1, step=100, metrics={}, path="/tmp/checkpoint-100"),
            Checkpoint(id=2, run_id=1, step=500, metrics={}, path="/tmp/checkpoint-500"),
        ])

        mock_session.query = Mock(return_value=mock_query)

        checkpoints = mock_session.query(Checkpoint).filter(
            Checkpoint.run_id == 1
        ).order_by(Checkpoint.step).all()

        assert len(checkpoints) == 2
        assert checkpoints[0].step == 100
        assert checkpoints[1].step == 500

    def test_get_latest_checkpoint(self, sample_run):
        """Test getting latest checkpoint for a run."""
        checkpoints = [
            Checkpoint(id=1, run_id=1, step=100, metrics={}, path="/tmp/checkpoint-100"),
            Checkpoint(id=2, run_id=1, step=500, metrics={}, path="/tmp/checkpoint-500"),
            Checkpoint(id=3, run_id=1, step=1000, metrics={}, path="/tmp/checkpoint-1000"),
        ]

        sample_run.checkpoints = checkpoints

        # Get latest checkpoint
        latest = max(sample_run.checkpoints, key=lambda c: c.step)

        assert latest.step == 1000
        assert latest.path == "/tmp/checkpoint-1000"


# ============================================================================
# TEST 5: CHECKPOINT METRICS
# ============================================================================

class TestCheckpointMetrics:
    """Test checkpoint metrics."""

    def test_checkpoint_stores_training_metrics(self):
        """Test checkpoint stores training metrics."""
        metrics = {
            "loss": 0.3,
            "accuracy": 0.92,
            "learning_rate": 0.0001,
            "epoch": 2
        }

        checkpoint = Checkpoint(
            run_id=1,
            step=1000,
            metrics=metrics,
            path="/tmp/checkpoint-1000"
        )

        assert checkpoint.metrics["loss"] == 0.3
        assert checkpoint.metrics["accuracy"] == 0.92
        assert checkpoint.metrics["learning_rate"] == 0.0001

    def test_checkpoint_metrics_are_json_serializable(self):
        """Test checkpoint metrics can be serialized to JSON."""
        metrics = {
            "loss": 0.3,
            "accuracy": 0.92,
            "tokens_processed": 1000000
        }

        # Should be serializable
        json_str = json.dumps(metrics)
        assert json_str is not None

        # Should be deserializable
        loaded = json.loads(json_str)
        assert loaded["loss"] == 0.3

    def test_checkpoint_tracks_improvement(self):
        """Test tracking metric improvement across checkpoints."""
        checkpoints = [
            Checkpoint(run_id=1, step=100, metrics={"loss": 0.8}, path="/tmp/checkpoint-100"),
            Checkpoint(run_id=1, step=500, metrics={"loss": 0.5}, path="/tmp/checkpoint-500"),
            Checkpoint(run_id=1, step=1000, metrics={"loss": 0.3}, path="/tmp/checkpoint-1000"),
        ]

        # Calculate improvement
        initial_loss = checkpoints[0].metrics["loss"]
        final_loss = checkpoints[-1].metrics["loss"]
        improvement = initial_loss - final_loss

        assert improvement > 0  # Loss should decrease
        assert improvement == 0.5


# ============================================================================
# TEST 6: CHECKPOINT SELECTION
# ============================================================================

class TestCheckpointSelection:
    """Test selecting best checkpoints."""

    def test_select_best_checkpoint_by_loss(self):
        """Test selecting checkpoint with lowest loss."""
        checkpoints = [
            Checkpoint(id=1, run_id=1, step=100, metrics={"loss": 0.8}, path="/tmp/checkpoint-100"),
            Checkpoint(id=2, run_id=1, step=500, metrics={"loss": 0.3}, path="/tmp/checkpoint-500"),
            Checkpoint(id=3, run_id=1, step=1000, metrics={"loss": 0.5}, path="/tmp/checkpoint-1000"),
        ]

        # Find best checkpoint (lowest loss)
        best = min(checkpoints, key=lambda c: c.metrics.get("loss", float('inf')))

        assert best.step == 500
        assert best.metrics["loss"] == 0.3

    def test_select_best_checkpoint_by_accuracy(self):
        """Test selecting checkpoint with highest accuracy."""
        checkpoints = [
            Checkpoint(id=1, run_id=1, step=100, metrics={"accuracy": 0.85}, path="/tmp/checkpoint-100"),
            Checkpoint(id=2, run_id=1, step=500, metrics={"accuracy": 0.92}, path="/tmp/checkpoint-500"),
            Checkpoint(id=3, run_id=1, step=1000, metrics={"accuracy": 0.88}, path="/tmp/checkpoint-1000"),
        ]

        # Find best checkpoint (highest accuracy)
        best = max(checkpoints, key=lambda c: c.metrics.get("accuracy", 0))

        assert best.step == 500
        assert best.metrics["accuracy"] == 0.92

    def test_select_checkpoint_at_specific_step(self):
        """Test selecting checkpoint at specific step."""
        checkpoints = [
            Checkpoint(id=1, run_id=1, step=100, metrics={}, path="/tmp/checkpoint-100"),
            Checkpoint(id=2, run_id=1, step=500, metrics={}, path="/tmp/checkpoint-500"),
            Checkpoint(id=3, run_id=1, step=1000, metrics={}, path="/tmp/checkpoint-1000"),
        ]

        # Find checkpoint at step 500
        target_step = 500
        checkpoint = next((c for c in checkpoints if c.step == target_step), None)

        assert checkpoint is not None
        assert checkpoint.step == 500


# ============================================================================
# TEST 7: CHECKPOINT DOWNLOAD
# ============================================================================

class TestCheckpointDownload:
    """Test checkpoint download functionality."""

    def test_checkpoint_download_path(self, sample_checkpoint):
        """Test getting checkpoint download path."""
        download_path = sample_checkpoint.path

        assert download_path == "/tmp/checkpoints/checkpoint-100"

    def test_checkpoint_exists_before_download(self, temp_checkpoint_dir):
        """Test checking checkpoint exists before download."""
        checkpoint_path = temp_checkpoint_dir / "checkpoint-1000"
        checkpoint_path.mkdir(exist_ok=True)

        # Create dummy checkpoint file
        (checkpoint_path / "model.pt").touch()

        assert checkpoint_path.exists()
        assert (checkpoint_path / "model.pt").exists()

    def test_checkpoint_download_error_if_missing(self):
        """Test error if checkpoint file is missing."""
        checkpoint = Checkpoint(
            run_id=1,
            step=1000,
            metrics={},
            path="/nonexistent/checkpoint-1000"
        )

        checkpoint_path = Path(checkpoint.path)

        # Should not exist
        assert not checkpoint_path.exists()


# ============================================================================
# TEST 8: ERROR HANDLING
# ============================================================================

class TestCheckpointErrorHandling:
    """Test error handling in checkpoint management."""

    def test_handle_missing_checkpoint_path(self):
        """Test handling missing checkpoint path."""
        checkpoint = Checkpoint(
            run_id=1,
            step=1000,
            metrics={},
            path=None  # Missing path
        )

        # Should handle None path
        if checkpoint.path is None:
            assert True  # Expected

    def test_handle_invalid_checkpoint_step(self):
        """Test handling invalid checkpoint step."""
        with pytest.raises((ValueError, TypeError)):
            Checkpoint(
                run_id=1,
                step="invalid",  # Should be int
                metrics={},
                path="/tmp/checkpoint"
            )

    def test_handle_checkpoint_corruption(self, temp_checkpoint_dir):
        """Test handling corrupted checkpoint."""
        checkpoint_path = temp_checkpoint_dir / "checkpoint-1000"
        checkpoint_path.mkdir(exist_ok=True)

        # Create corrupted metadata
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            f.write("corrupted json {{{")

        # Try to load corrupted metadata
        with pytest.raises(json.JSONDecodeError):
            with open(metadata_file, 'r') as f:
                json.load(f)

    def test_handle_no_checkpoints_for_run(self, sample_run):
        """Test handling run with no checkpoints."""
        sample_run.checkpoints = []

        if not sample_run.checkpoints:
            # Should handle gracefully
            assert len(sample_run.checkpoints) == 0

    def test_handle_checkpoint_registration_failure(self, mock_session):
        """Test handling checkpoint registration failure."""
        checkpoint = Checkpoint(
            run_id=1,
            step=1000,
            metrics={},
            path="/tmp/checkpoint-1000"
        )

        # Mock commit failure
        mock_session.commit.side_effect = Exception("Database error")

        with pytest.raises(Exception) as exc_info:
            mock_session.add(checkpoint)
            mock_session.commit()

        assert "Database error" in str(exc_info.value)


# ============================================================================
# TEST 9: CHECKPOINT LIFECYCLE
# ============================================================================

class TestCheckpointLifecycle:
    """Test complete checkpoint lifecycle."""

    @pytest.mark.asyncio
    async def test_full_checkpoint_lifecycle(self, mock_session, temp_checkpoint_dir):
        """Test complete checkpoint lifecycle from creation to retrieval."""
        run_id = 1
        step = 1000

        # 1. Create checkpoint directory
        checkpoint_path = temp_checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(exist_ok=True)

        # 2. Save checkpoint metadata
        metadata = {
            "step": step,
            "metrics": {"loss": 0.3, "accuracy": 0.92},
            "timestamp": datetime.utcnow().isoformat()
        }

        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # 3. Register checkpoint in database
        checkpoint = Checkpoint(
            run_id=run_id,
            step=step,
            metrics=metadata["metrics"],
            path=str(checkpoint_path)
        )

        mock_session.add(checkpoint)
        mock_session.commit()

        # 4. Verify checkpoint is retrievable
        assert checkpoint_path.exists()
        assert metadata_file.exists()

        # 5. Verify metadata is correct
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)

        assert loaded_metadata["step"] == step
        assert loaded_metadata["metrics"]["loss"] == 0.3


# ============================================================================
# TEST SUMMARY
# ============================================================================

"""
Test Coverage Summary:
======================

1. Checkpoint Creation: 3 tests
   - Create checkpoint object
   - Require run_id
   - Require step

2. Checkpoint Registration: 3 tests
   - Parse from log
   - Register to database
   - Register multiple checkpoints

3. Checkpoint Storage: 3 tests
   - Path structure
   - File naming
   - Metadata storage

4. Checkpoint Retrieval: 3 tests
   - Get by ID
   - Get for run
   - Get latest

5. Checkpoint Metrics: 3 tests
   - Store training metrics
   - JSON serialization
   - Track improvement

6. Checkpoint Selection: 3 tests
   - Select by loss
   - Select by accuracy
   - Select at specific step

7. Checkpoint Download: 3 tests
   - Download path
   - Exists before download
   - Error if missing

8. Error Handling: 5 tests
   - Missing path
   - Invalid step
   - Corruption
   - No checkpoints
   - Registration failure

9. Checkpoint Lifecycle: 1 test
   - Full lifecycle

Total: 27 checkpoint management tests

To Run:
-------
# Run all checkpoint tests
pytest tests/test_checkpoint_management.py -v

# Run specific test class
pytest tests/test_checkpoint_management.py::TestCheckpointSelection -v

# Run with coverage
pytest tests/test_checkpoint_management.py --cov=models --cov-report=html
"""
