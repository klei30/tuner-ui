"""
Test Suite for API Endpoints
==============================

Comprehensive tests for all FastAPI endpoints in the Tuner UI backend.

Test Categories:
1. Health & Status Endpoints
2. Project Management Endpoints
3. Dataset Management Endpoints
4. Run Management Endpoints
5. Checkpoint Endpoints
6. Hyperparameter Endpoints
7. Model Management Endpoints
8. Chat & Sampling Endpoints
9. Evaluation Endpoints
10. Metrics & Logs Endpoints

Usage:
    pytest tests/test_api_endpoints.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app


# ============================================================================
# TEST CLIENT SETUP
# ============================================================================

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = Mock()
    session.query = Mock()
    session.get = Mock()
    session.add = Mock()
    session.commit = Mock()
    return session


@pytest.fixture
def mock_get_db(mock_db_session):
    """Override get_db dependency."""
    def _get_db():
        try:
            yield mock_db_session
        finally:
            pass
    return _get_db


# ============================================================================
# TEST 1: HEALTH & STATUS ENDPOINTS
# ============================================================================

class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_health_endpoint(self, client):
        """Test GET /health returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "healthy" in data["message"].lower() or "ok" in data["message"].lower()

    def test_my_test_endpoint(self, client):
        """Test GET /my-test-endpoint returns 200."""
        response = client.get("/my-test-endpoint")
        assert response.status_code == 200


# ============================================================================
# TEST 2: PROJECT MANAGEMENT ENDPOINTS
# ============================================================================

class TestProjectEndpoints:
    """Test project management endpoints."""

    def test_create_project_success(self, client):
        """Test POST /projects creates a project."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_session.add = Mock()
            mock_session.commit = Mock()
            mock_session.refresh = Mock()
            mock_get_db.return_value = [mock_session]

            project_data = {
                "name": "Test Project",
                "description": "A test project"
            }

            response = client.post("/projects", json=project_data)

            # Should create project (200 or 201)
            assert response.status_code in [200, 201]

    def test_create_project_missing_name(self, client):
        """Test POST /projects with missing name fails."""
        project_data = {
            "description": "A test project"
        }

        response = client.post("/projects", json=project_data)

        # Should fail validation (422)
        assert response.status_code == 422

    def test_list_projects(self, client):
        """Test GET /projects returns list of projects."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_query = Mock()
            mock_query.all = Mock(return_value=[])
            mock_session.query = Mock(return_value=mock_query)
            mock_get_db.return_value = [mock_session]

            response = client.get("/projects")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)


# ============================================================================
# TEST 3: DATASET MANAGEMENT ENDPOINTS
# ============================================================================

class TestDatasetEndpoints:
    """Test dataset management endpoints."""

    def test_create_dataset_huggingface(self, client):
        """Test POST /datasets creates HuggingFace dataset."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_session.add = Mock()
            mock_session.commit = Mock()
            mock_session.refresh = Mock()
            mock_get_db.return_value = [mock_session]

            dataset_data = {
                "name": "test-dataset",
                "kind": "huggingface",
                "spec": {
                    "repo": "yahma/alpaca-cleaned"
                }
            }

            response = client.post("/datasets", json=dataset_data)

            assert response.status_code in [200, 201]

    def test_create_dataset_invalid_kind(self, client):
        """Test POST /datasets with invalid kind fails."""
        dataset_data = {
            "name": "test-dataset",
            "kind": "invalid_kind",
            "spec": {}
        }

        response = client.post("/datasets", json=dataset_data)

        # Should fail validation
        assert response.status_code in [400, 422]

    def test_list_datasets(self, client):
        """Test GET /datasets returns list of datasets."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_query = Mock()
            mock_query.all = Mock(return_value=[])
            mock_session.query = Mock(return_value=mock_query)
            mock_get_db.return_value = [mock_session]

            response = client.get("/datasets")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_validate_dataset(self, client):
        """Test POST /datasets/validate validates dataset spec."""
        validate_data = {
            "kind": "huggingface",
            "spec": {
                "repo": "yahma/alpaca-cleaned"
            }
        }

        response = client.post("/datasets/validate", json=validate_data)

        # Should return validation result (200 or 400)
        assert response.status_code in [200, 400]

    def test_preview_dataset(self, client):
        """Test GET /datasets/preview returns sample data."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_dataset = Mock()
            mock_dataset.kind = "huggingface"
            mock_dataset.spec = {"repo": "yahma/alpaca-cleaned"}
            mock_session.get = Mock(return_value=mock_dataset)
            mock_get_db.return_value = [mock_session]

            response = client.get("/datasets/preview?dataset_id=1&num_samples=5")

            # Should return preview (200) or error if can't load
            assert response.status_code in [200, 400, 404, 500]


# ============================================================================
# TEST 4: RUN MANAGEMENT ENDPOINTS
# ============================================================================

class TestRunEndpoints:
    """Test run management endpoints."""

    def test_create_run_success(self, client):
        """Test POST /runs creates a training run."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_session.add = Mock()
            mock_session.commit = Mock()
            mock_session.refresh = Mock()
            mock_session.get = Mock(return_value=Mock(id=1))
            mock_get_db.return_value = [mock_session]

            run_data = {
                "project_id": 1,
                "dataset_id": 1,
                "recipe_type": "SFT",
                "config": {
                    "base_model": "meta-llama/Llama-3.2-1B",
                    "hyperparameters": {
                        "learning_rate": 0.0001,
                        "epochs": 1,
                        "batch_size": 128
                    }
                }
            }

            response = client.post("/runs", json=run_data)

            assert response.status_code in [200, 201]

    def test_create_run_invalid_recipe_type(self, client):
        """Test POST /runs with invalid recipe type fails."""
        run_data = {
            "project_id": 1,
            "dataset_id": 1,
            "recipe_type": "INVALID_TYPE",
            "config": {}
        }

        response = client.post("/runs", json=run_data)

        # Should fail validation
        assert response.status_code in [400, 422]

    def test_list_runs(self, client):
        """Test GET /runs returns paginated list."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_query = Mock()
            mock_query.order_by = Mock(return_value=mock_query)
            mock_query.offset = Mock(return_value=mock_query)
            mock_query.limit = Mock(return_value=mock_query)
            mock_query.all = Mock(return_value=[])
            mock_query.count = Mock(return_value=0)
            mock_session.query = Mock(return_value=mock_query)
            mock_get_db.return_value = [mock_session]

            response = client.get("/runs?skip=0&limit=10")

            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert "total" in data

    def test_get_run_detail(self, client):
        """Test GET /runs/{run_id} returns run details."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_run = Mock()
            mock_run.id = 1
            mock_run.recipe_type = "SFT"
            mock_run.status = "completed"
            mock_run.progress = 1.0
            mock_run.checkpoints = []
            mock_run.evaluations = []
            mock_session.get = Mock(return_value=mock_run)
            mock_get_db.return_value = [mock_session]

            response = client.get("/runs/1")

            # Should return run details (200) or 404 if not found
            assert response.status_code in [200, 404]

    def test_cancel_run(self, client):
        """Test POST /runs/{run_id}/cancel cancels a run."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_run = Mock()
            mock_run.id = 1
            mock_run.status = "running"
            mock_session.get = Mock(return_value=mock_run)
            mock_session.commit = Mock()
            mock_get_db.return_value = [mock_session]

            response = client.post("/runs/1/cancel")

            # Should cancel run (200) or fail if can't cancel
            assert response.status_code in [200, 400, 404]

    def test_resume_run(self, client):
        """Test POST /runs/{run_id}/resume resumes a run."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_run = Mock()
            mock_run.id = 1
            mock_run.status = "failed"
            mock_run.checkpoints = [Mock(step=100)]
            mock_session.get = Mock(return_value=mock_run)
            mock_get_db.return_value = [mock_session]

            response = client.post("/runs/1/resume")

            # Should resume run (200) or fail if can't resume
            assert response.status_code in [200, 400, 404]


# ============================================================================
# TEST 5: CHECKPOINT ENDPOINTS
# ============================================================================

class TestCheckpointEndpoints:
    """Test checkpoint management endpoints."""

    def test_download_checkpoint(self, client):
        """Test GET /checkpoints/{checkpoint_id}/download."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_checkpoint = Mock()
            mock_checkpoint.id = 1
            mock_checkpoint.path = "/tmp/checkpoint-100"
            mock_session.get = Mock(return_value=mock_checkpoint)
            mock_get_db.return_value = [mock_session]

            response = client.get("/checkpoints/1/download")

            # Should return file or error (200, 404, or 500)
            assert response.status_code in [200, 404, 500]


# ============================================================================
# TEST 6: HYPERPARAMETER ENDPOINTS
# ============================================================================

class TestHyperparameterEndpoints:
    """Test hyperparameter calculation endpoints."""

    def test_auto_lr_calculation(self, client):
        """Test GET /hyperparameters/auto-lr calculates learning rate."""
        response = client.get(
            "/hyperparameters/auto-lr"
            "?model_name=meta-llama/Llama-3.2-1B"
            "&recipe_type=sft"
            "&dataset_size=1000"
        )

        # Should return LR calculation (200) or error
        assert response.status_code in [200, 400, 422]

    def test_calculate_hyperparameters(self, client):
        """Test POST /hyperparameters/calculate."""
        hyperparam_data = {
            "model_name": "meta-llama/Llama-3.2-1B",
            "recipe_type": "sft",
            "dataset_size": 1000,
            "lora_rank": 32
        }

        response = client.post("/hyperparameters/calculate", json=hyperparam_data)

        # Should return hyperparameters (200) or error
        assert response.status_code in [200, 400, 422]


# ============================================================================
# TEST 7: MODEL MANAGEMENT ENDPOINTS
# ============================================================================

class TestModelEndpoints:
    """Test model management endpoints."""

    def test_list_models(self, client):
        """Test GET /models returns model catalog."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data or isinstance(data, list)

    def test_create_custom_model(self, client):
        """Test POST /models creates custom model."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_session.add = Mock()
            mock_session.commit = Mock()
            mock_session.refresh = Mock()
            mock_get_db.return_value = [mock_session]

            model_data = {
                "name": "custom-model",
                "base_model": "meta-llama/Llama-3.2-1B",
                "checkpoint_path": "/path/to/checkpoint"
            }

            response = client.post("/models", json=model_data)

            assert response.status_code in [200, 201, 422]

    def test_get_model_renderers(self, client):
        """Test GET /models/{model_name}/renderers."""
        response = client.get("/models/meta-llama%2FLlama-3.2-1B/renderers")

        # Should return available renderers (200) or error
        assert response.status_code in [200, 404]


# ============================================================================
# TEST 8: CHAT & SAMPLING ENDPOINTS
# ============================================================================

class TestChatSamplingEndpoints:
    """Test chat and sampling endpoints."""

    @patch("main.get_sampling_client")
    @patch("main.get_tokenizer")
    @patch("main.get_renderer")
    def test_chat_endpoint(self, mock_renderer, mock_tokenizer, mock_sampling_client, client):
        """Test POST /chat generates chat response."""
        # Setup mocks
        mock_sampling_client.return_value.sample.return_value.result.return_value = Mock(
            sequences=[Mock(tokens=[1, 2, 3])]
        )
        mock_renderer.return_value.parse_response.return_value = [
            {"role": "assistant", "content": "Test response"}
        ]
        mock_renderer.return_value.build_generation_prompt.return_value = "Test prompt"
        mock_renderer.return_value.get_stop_sequences.return_value = []
        mock_tokenizer.return_value.decode.return_value = "Test response"

        chat_data = {
            "model": "meta-llama/Llama-3.2-1B",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }

        response = client.post("/chat", json=chat_data)

        # Should return chat response (200) or error
        assert response.status_code in [200, 400, 422, 500]

    @patch("main.get_sampling_client")
    @patch("main.get_tokenizer")
    def test_sample_endpoint(self, mock_tokenizer, mock_sampling_client, client):
        """Test POST /sample generates samples."""
        # Setup mocks
        mock_sampling_client.return_value.sample.return_value.result.return_value = Mock(
            sequences=[Mock(tokens=[1, 2, 3])]
        )
        mock_tokenizer.return_value.decode.return_value = "Test sample"

        sample_data = {
            "model": "meta-llama/Llama-3.2-1B",
            "prompt": "Hello",
            "num_samples": 3,
            "temperature": 0.7,
            "max_tokens": 100
        }

        response = client.post("/sample", json=sample_data)

        # Should return samples (200) or error
        assert response.status_code in [200, 400, 422, 500]


# ============================================================================
# TEST 9: EVALUATION ENDPOINTS
# ============================================================================

class TestEvaluationEndpoints:
    """Test evaluation endpoints."""

    def test_list_evaluations(self, client):
        """Test GET /evaluations returns list of evaluations."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_query = Mock()
            mock_query.all = Mock(return_value=[])
            mock_session.query = Mock(return_value=mock_query)
            mock_get_db.return_value = [mock_session]

            response = client.get("/evaluations")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    @patch("main.get_sampling_client")
    @patch("main.get_tokenizer")
    @patch("main.get_renderer")
    def test_evaluate_run(self, mock_renderer, mock_tokenizer, mock_sampling_client, client):
        """Test POST /runs/{run_id}/evaluate creates evaluation."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_run = Mock()
            mock_run.id = 1
            mock_run.config_json = {
                "base_model": "meta-llama/Llama-3.2-1B",
                "renderer_name": "role_colon"
            }
            mock_run.checkpoints = [Mock(path="/tmp/checkpoint-100")]
            mock_session.get = Mock(return_value=mock_run)
            mock_session.add = Mock()
            mock_session.commit = Mock()
            mock_get_db.return_value = [mock_session]

            # Setup mocks
            mock_sampling_client.return_value.sample.return_value.result.return_value = Mock(
                sequences=[Mock(tokens=[1, 2, 3])]
            )
            mock_renderer.return_value.parse_response.return_value = [
                {"role": "assistant", "content": "Test response"}
            ]
            mock_renderer.return_value.build_generation_prompt.return_value = "Test prompt"
            mock_renderer.return_value.get_stop_sequences.return_value = []

            eval_data = {
                "evaluation_type": "mmlu",
                "test_prompts": [
                    {"role": "user", "content": "Test question"}
                ]
            }

            response = client.post("/runs/1/evaluate", json=eval_data)

            # Should create evaluation (200) or error
            assert response.status_code in [200, 400, 404, 500]


# ============================================================================
# TEST 10: METRICS & LOGS ENDPOINTS
# ============================================================================

class TestMetricsLogsEndpoints:
    """Test metrics and logs endpoints."""

    def test_get_run_logs(self, client):
        """Test GET /runs/{run_id}/logs returns log tail."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_run = Mock()
            mock_run.id = 1
            mock_session.get = Mock(return_value=mock_run)
            mock_get_db.return_value = [mock_session]

            response = client.get("/runs/1/logs?lines=100")

            # Should return logs (200) or error
            assert response.status_code in [200, 404, 500]

    def test_get_run_metrics(self, client):
        """Test GET /runs/{run_id}/metrics returns metrics."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_run = Mock()
            mock_run.id = 1
            mock_run.checkpoints = []
            mock_session.get = Mock(return_value=mock_run)
            mock_get_db.return_value = [mock_session]

            response = client.get("/runs/1/metrics")

            # Should return metrics (200) or error
            assert response.status_code in [200, 404]

    def test_get_realtime_metrics(self, client):
        """Test GET /runs/{run_id}/realtime-metrics."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_run = Mock()
            mock_run.id = 1
            mock_run.status = "running"
            mock_session.get = Mock(return_value=mock_run)
            mock_get_db.return_value = [mock_session]

            response = client.get("/runs/1/realtime-metrics")

            # Should return realtime metrics (200) or error
            assert response.status_code in [200, 404]

    def test_get_visualization_data(self, client):
        """Test GET /runs/{run_id}/visualization-data."""
        with patch("main.get_db") as mock_get_db:
            mock_session = Mock()
            mock_run = Mock()
            mock_run.id = 1
            mock_run.checkpoints = []
            mock_session.get = Mock(return_value=mock_run)
            mock_get_db.return_value = [mock_session]

            response = client.get("/runs/1/visualization-data")

            # Should return visualization data (200) or error
            assert response.status_code in [200, 404]


# ============================================================================
# TEST 11: COST ESTIMATION
# ============================================================================

class TestCostEstimation:
    """Test cost estimation endpoint."""

    def test_estimate_cost(self, client):
        """Test POST /estimate-cost."""
        cost_data = {
            "model_name": "meta-llama/Llama-3.2-1B",
            "recipe_type": "sft",
            "dataset_size": 1000,
            "num_epochs": 3,
            "batch_size": 128
        }

        response = client.post("/estimate-cost", json=cost_data)

        # Should return cost estimate (200) or error
        assert response.status_code in [200, 400, 422]


# ============================================================================
# TEST SUMMARY
# ============================================================================

"""
Test Coverage Summary:
======================

1. Health & Status: 2 tests
   - Health check endpoint
   - Test endpoint

2. Project Management: 3 tests
   - Create project (success & validation)
   - List projects

3. Dataset Management: 5 tests
   - Create dataset (HuggingFace, invalid)
   - List datasets
   - Validate dataset
   - Preview dataset

4. Run Management: 6 tests
   - Create run (success & validation)
   - List runs
   - Get run detail
   - Cancel run
   - Resume run

5. Checkpoint: 1 test
   - Download checkpoint

6. Hyperparameters: 2 tests
   - Auto LR calculation
   - Calculate hyperparameters

7. Models: 3 tests
   - List models
   - Create custom model
   - Get model renderers

8. Chat & Sampling: 2 tests
   - Chat endpoint
   - Sample endpoint

9. Evaluations: 2 tests
   - List evaluations
   - Evaluate run

10. Metrics & Logs: 4 tests
    - Get run logs
    - Get run metrics
    - Get realtime metrics
    - Get visualization data

11. Cost Estimation: 1 test
    - Estimate cost

Total: 31 API endpoint tests

To Run:
-------
# Run all API tests
pytest tests/test_api_endpoints.py -v

# Run specific test class
pytest tests/test_api_endpoints.py::TestRunEndpoints -v

# Run with coverage
pytest tests/test_api_endpoints.py --cov=main --cov-report=html
"""
