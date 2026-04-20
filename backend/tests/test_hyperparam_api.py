"""
API tests for hyperparameter calculation endpoint.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app


class TestHyperparamAPI:
    """Test the hyperparameter API endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_calculate_hyperparameters_success(self):
        """Test successful hyperparameter calculation."""
        request_data = {
            "model_name": "meta-llama/Llama-3.1-8B",
            "recipe_type": "sft",
            "lora_rank": 64,
        }

        response = self.client.post("/hyperparameters/calculate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["model_name"] == "meta-llama/Llama-3.1-8B"
        assert data["recipe_type"] == "sft"
        assert "recommendations" in data
        assert "explanation" in data

        # Check recommendations structure
        recs = data["recommendations"]
        assert "learning_rate" in recs
        assert "batch_size" in recs
        assert "lora_rank" in recs
        assert recs["lora_rank"] == 64

        # Check explanation structure
        exp = data["explanation"]
        assert "learning_rate" in exp
        assert "batch_size" in exp
        assert "lora_rank" in exp
        assert "notes" in exp
        assert "source" in exp

    def test_calculate_hyperparameters_auto_lora_rank(self):
        """Test hyperparameter calculation with auto LoRA rank."""
        request_data = {
            "model_name": "meta-llama/Llama-3.1-8B",
            "recipe_type": "sft",
            # No lora_rank provided
        }

        response = self.client.post("/hyperparameters/calculate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["recommendations"]["lora_rank"] == 64  # Auto-calculated

    def test_calculate_hyperparameters_different_recipes(self):
        """Test hyperparameter calculation for different recipe types."""
        recipes = ["sft", "dpo", "rl"]

        for recipe in recipes:
            request_data = {
                "model_name": "meta-llama/Llama-3.1-8B",
                "recipe_type": recipe,
            }

            response = self.client.post("/hyperparameters/calculate", json=request_data)
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert data["recipe_type"] == recipe

    def test_calculate_hyperparameters_different_models(self):
        """Test hyperparameter calculation for different models."""
        models = [
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.1-8B",
            "Qwen/Qwen2.5-7B",
        ]

        for model in models:
            request_data = {"model_name": model, "recipe_type": "sft"}

            response = self.client.post("/hyperparameters/calculate", json=request_data)
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert data["model_name"] == model

    @patch("backend.main.HyperparamCalculator.get_all_recommendations")
    def test_calculate_hyperparameters_backend_error(self, mock_get_recommendations):
        """Test handling of backend calculation errors."""
        mock_get_recommendations.side_effect = ValueError("Test error")

        request_data = {"model_name": "meta-llama/Llama-3.1-8B", "recipe_type": "sft"}

        response = self.client.post("/hyperparameters/calculate", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert "Failed to calculate hyperparameters" in data["detail"]
        assert "Test error" in data["detail"]

    def test_calculate_hyperparameters_invalid_request(self):
        """Test handling of invalid request data."""
        # Missing required fields
        response = self.client.post("/hyperparameters/calculate", json={})
        assert response.status_code == 422  # Validation error

        # Invalid recipe type
        request_data = {
            "model_name": "meta-llama/Llama-3.1-8B",
            "recipe_type": "invalid_recipe",
        }
        response = self.client.post("/hyperparameters/calculate", json=request_data)
        # Should still work but with default behavior
        assert response.status_code == 200

    def test_get_auto_learning_rate(self):
        """Test the auto learning rate endpoint."""
        response = self.client.get(
            "/hyperparameters/auto-lr?model_name=meta-llama/Llama-3.1-8B&is_lora=true"
        )

        assert response.status_code == 200
        data = response.json()

        assert "model_name" in data
        assert "is_lora" in data
        assert "optimal_learning_rate" in data
        assert "explanation" in data

        assert data["model_name"] == "meta-llama/Llama-3.1-8B"
        assert data["is_lora"] is True
        assert isinstance(data["optimal_learning_rate"], float)

    def test_get_auto_learning_rate_full_finetune(self):
        """Test auto learning rate for full fine-tuning."""
        response = self.client.get(
            "/hyperparameters/auto-lr?model_name=meta-llama/Llama-3.1-8B&is_lora=false"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["is_lora"] is False
        assert isinstance(data["optimal_learning_rate"], float)

    @patch("backend.main.get_lr")
    def test_get_auto_learning_rate_error(self, mock_get_lr):
        """Test error handling in auto learning rate endpoint."""
        mock_get_lr.side_effect = Exception("Test error")

        response = self.client.get(
            "/hyperparameters/auto-lr?model_name=invalid-model&is_lora=true"
        )

        assert response.status_code == 400
        data = response.json()
        assert "Failed to calculate learning rate" in data["detail"]
