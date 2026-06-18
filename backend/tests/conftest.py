"""
Test configuration and utilities for comprehensive test suite.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock
import tempfile
import asyncio
import os

# Add backend to Python path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Test configuration
TEST_MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-0.5B",
]

TEST_RECIPES = ["sft", "dpo", "rl", "math_rl", "chat_sl", "distillation", "on_policy_distillation"]

EXPECTED_HYPERPARAMS = {
    # Model -> Recipe -> expected values
    ("meta-llama/Llama-3.2-1B", "sft"): {"batch_size": 128, "lora_rank": 32},
    ("meta-llama/Llama-3.1-8B", "sft"): {"batch_size": 64, "lora_rank": 64},
    ("meta-llama/Llama-3.1-70B", "sft"): {"batch_size": 32, "lora_rank": 128},
    ("meta-llama/Llama-3.1-8B", "dpo"): {"batch_size": 32, "lora_rank": 32},
    ("meta-llama/Llama-3.1-8B", "rl"): {"batch_size": 16, "lora_rank": 32},
}


def pytest_collection_modifyitems(session, config, items):
    if os.getenv("RUN_LIVE_E2E") == "1":
        return

    live_e2e_files = {"test_hyperparam_e2e.py", "test_hyperparam_manual.py"}
    skip_live = pytest.mark.skip(
        reason="requires a live backend server on localhost:8000; set RUN_LIVE_E2E=1"
    )
    for item in items:
        if item.path.name in live_e2e_files:
            item.add_marker(skip_live)


# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest.fixture
def mock_session():
    """Mock database session."""
    session = Mock()
    session.get = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.query = Mock()
    session.close = Mock()
    session.rollback = Mock()
    session.flush = Mock()
    return session


@pytest.fixture
def mock_db():
    """Mock database dependency for FastAPI."""
    def _get_db():
        session = Mock()
        session.query = Mock()
        session.get = Mock()
        session.add = Mock()
        session.commit = Mock()
        session.close = Mock()
        try:
            yield session
        finally:
            session.close()
    return _get_db


# ============================================================================
# MODEL FIXTURES
# ============================================================================

@pytest.fixture
def sample_user():
    """Create a sample user object."""
    from models import User
    return User(
        id=1,
        email="test@example.com",
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_project():
    """Create a sample project object."""
    from models import Project
    return Project(
        id=1,
        name="Test Project",
        user_id=1,
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_dataset():
    """Create a sample dataset object."""
    from models import Dataset
    return Dataset(
        id=1,
        name="test-dataset",
        kind="huggingface",
        spec={"repo": "yahma/alpaca-cleaned"},
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_run():
    """Create a sample run object."""
    from models import Run
    return Run(
        id=1,
        project_id=1,
        dataset_id=1,
        recipe_type="SFT",
        status="pending",
        progress=0.0,
        config_json={
            "base_model": "meta-llama/Llama-3.2-1B",
            "renderer_name": "role_colon",
            "hyperparameters": {
                "learning_rate": 0.0001,
                "epochs": 1,
                "batch_size": 128,
                "lora_rank": 32,
            }
        },
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_checkpoint():
    """Create a sample checkpoint object."""
    from models import Checkpoint
    return Checkpoint(
        id=1,
        run_id=1,
        step=100,
        metrics={"loss": 0.5, "accuracy": 0.95},
        path="/tmp/checkpoints/checkpoint-100",
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_evaluation():
    """Create a sample evaluation object."""
    from models import Evaluation
    return Evaluation(
        id=1,
        run_id=1,
        evaluator_name="mmlu",
        metrics={"accuracy": 0.85, "f1_score": 0.82},
        created_at=datetime.utcnow(),
    )


# ============================================================================
# JOB RUNNER FIXTURES
# ============================================================================

@pytest.fixture
def job_runner():
    """Create job runner instance."""
    from job_runner import JobRunner
    return JobRunner()


@pytest.fixture
def temp_log_file():
    """Create a temporary log file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        log_path = Path(f.name)
        f.write("[METRICS] step=100, loss=0.5, lr=0.0001\n")
        f.write("[METRICS] step=200, loss=0.4, lr=0.00008\n")

    yield log_path

    # Cleanup
    if log_path.exists():
        log_path.unlink()


# ============================================================================
# FASTAPI FIXTURES
# ============================================================================

@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


@pytest.fixture
def async_client():
    """Create async FastAPI test client."""
    from httpx import AsyncClient
    from main import app

    async def _client():
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    return _client


# ============================================================================
# MOCK API FIXTURES
# ============================================================================

@pytest.fixture
def mock_tinker_api():
    """Mock Tinker API for testing."""
    mock_api = Mock()
    mock_api.create_training_job = AsyncMock(return_value={"job_id": "test-job-123"})
    mock_api.get_job_status = AsyncMock(return_value={"status": "completed", "progress": 1.0})
    mock_api.sample = AsyncMock(return_value=Mock(sequences=[Mock(tokens=[1, 2, 3])]))
    return mock_api


@pytest.fixture
def mock_renderer():
    """Mock renderer for testing."""
    renderer = Mock()
    renderer.build_generation_prompt = Mock(return_value="<|begin_of_text|>User: Hello\n\nAssistant:")
    renderer.parse_response = Mock(return_value=[{"role": "assistant", "content": "Hi there!"}])
    renderer.get_stop_sequences = Mock(return_value=["<|end_of_text|>"])
    return renderer


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode = Mock(return_value=[1, 2, 3])
    tokenizer.decode = Mock(return_value="Test output")
    return tokenizer


# ============================================================================
# DATASET FIXTURES
# ============================================================================

@pytest.fixture
def alpaca_format_data():
    """Sample data in Alpaca format."""
    return [
        {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour"
        },
        {
            "instruction": "Write a poem about the moon",
            "input": "",
            "output": "The moon shines bright\nIn the darkest night"
        }
    ]


@pytest.fixture
def messages_format_data():
    """Sample data in messages format."""
    return [
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain quantum computing"},
                {"role": "assistant", "content": "Quantum computing uses quantum mechanics..."}
            ]
        }
    ]


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def sample_hyperparam_request():
    """Sample hyperparameter calculation request."""
    return {
        "model_name": "meta-llama/Llama-3.1-8B",
        "recipe_type": "sft",
        "lora_rank": 64,
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Setup mock environment variables."""
    monkeypatch.setenv("TINKER_API_KEY", "tml-test-key-1234567890123456789012345")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setenv("TESTING", "true")


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoints"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        yield checkpoint_path


# ============================================================================
# ASYNC FIXTURES
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_job_runner():
    """Create async job runner instance."""
    from job_runner import JobRunner
    runner = JobRunner()
    yield runner
    # Cleanup tasks
    for task in runner._tasks.values():
        if not task.done():
            task.cancel()


# ============================================================================
# MOCK TINKER COOKBOOK
# ============================================================================

@pytest.fixture
def mock_tinker_cookbook():
    """Mock tinker_cookbook imports for testing."""
    # This would mock the tinker_cookbook dependencies
    # to allow testing without full Tinker installation
    pass


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark e2e tests
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Mark slow tests
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
