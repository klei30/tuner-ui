"""
End-to-end tests for hyperparameter calculation feature.
"""

import requests
from typing import Dict, Any


class TestHyperparamE2E:
    """End-to-end tests for hyperparameter calculation."""

    def setup_method(self):
        """Set up test environment."""
        self.base_url = "http://localhost:8000"
        self.api_key = "test-api-key"  # Would need to be configured

    def test_full_hyperparam_workflow(self):
        """Test complete hyperparameter calculation workflow."""
        # Test data
        test_cases = [
            {
                "model_name": "meta-llama/Llama-3.1-8B",
                "recipe_type": "sft",
                "expected_batch_size": 64,
                "expected_lora_rank": 64,
            },
            {
                "model_name": "meta-llama/Llama-3.2-1B",
                "recipe_type": "sft",
                "expected_batch_size": 128,
                "expected_lora_rank": 32,
            },
            {
                "model_name": "meta-llama/Llama-3.1-8B",
                "recipe_type": "dpo",
                "expected_batch_size": 32,
                "expected_lora_rank": 32,
            },
        ]

        for test_case in test_cases:
            self._test_hyperparam_calculation(test_case)

    def _test_hyperparam_calculation(self, test_case: Dict[str, Any]):
        """Test individual hyperparameter calculation."""
        payload = {
            "model_name": test_case["model_name"],
            "recipe_type": test_case["recipe_type"],
        }

        response = requests.post(
            f"{self.base_url}/hyperparameters/calculate",
            json=payload,
            headers={"X-API-Key": self.api_key},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["success"] is True
        assert data["model_name"] == test_case["model_name"]
        assert data["recipe_type"] == test_case["recipe_type"]

        # Verify recommendations
        recs = data["recommendations"]
        assert "learning_rate" in recs
        assert "batch_size" in recs
        assert "lora_rank" in recs

        # Verify expected values
        assert recs["batch_size"] == test_case["expected_batch_size"]
        assert recs["lora_rank"] == test_case["expected_lora_rank"]
        assert isinstance(recs["learning_rate"], float)
        assert recs["learning_rate"] > 0

        # Verify explanation
        exp = data["explanation"]
        assert "learning_rate" in exp
        assert "batch_size" in exp
        assert "lora_rank" in exp
        assert "notes" in exp
        assert "source" in exp

    def test_hyperparam_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with unknown model
        payload = {"model_name": "unknown-model-xyz", "recipe_type": "sft"}

        response = requests.post(
            f"{self.base_url}/hyperparameters/calculate",
            json=payload,
            headers={"X-API-Key": self.api_key},
        )

        # Should still work with fallback values
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Test with invalid recipe type
        payload = {
            "model_name": "meta-llama/Llama-3.1-8B",
            "recipe_type": "invalid_recipe",
        }

        response = requests.post(
            f"{self.base_url}/hyperparameters/calculate",
            json=payload,
            headers={"X-API-Key": self.api_key},
        )

        # Should work with default behavior
        assert response.status_code == 200

    def test_hyperparam_consistency(self):
        """Test that hyperparameter calculations are consistent."""
        payload = {
            "model_name": "meta-llama/Llama-3.1-8B",
            "recipe_type": "sft",
            "lora_rank": 64,
        }

        # Make multiple requests
        responses = []
        for _ in range(3):
            response = requests.post(
                f"{self.base_url}/hyperparameters/calculate",
                json=payload,
                headers={"X-API-Key": self.api_key},
            )
            assert response.status_code == 200
            responses.append(response.json())

        # All responses should be identical
        first_response = responses[0]
        for response in responses[1:]:
            assert response["recommendations"] == first_response["recommendations"]
            assert response["explanation"] == first_response["explanation"]

    def test_hyperparam_auto_lr_endpoint(self):
        """Test the auto learning rate endpoint."""
        params = {"model_name": "meta-llama/Llama-3.1-8B", "is_lora": "true"}

        response = requests.get(
            f"{self.base_url}/hyperparameters/auto-lr",
            params=params,
            headers={"X-API-Key": self.api_key},
        )

        assert response.status_code == 200
        data = response.json()

        assert "optimal_learning_rate" in data
        assert isinstance(data["optimal_learning_rate"], float)
        assert data["optimal_learning_rate"] > 0

    def test_hyperparam_validation(self):
        """Test input validation."""
        # Test missing required fields
        response = requests.post(
            f"{self.base_url}/hyperparameters/calculate",
            json={},
            headers={"X-API-Key": self.api_key},
        )
        assert response.status_code == 422  # Validation error

        # Test with minimal valid payload
        payload = {"model_name": "meta-llama/Llama-3.1-8B", "recipe_type": "sft"}
        response = requests.post(
            f"{self.base_url}/hyperparameters/calculate",
            json=payload,
            headers={"X-API-Key": self.api_key},
        )
        assert response.status_code == 200
