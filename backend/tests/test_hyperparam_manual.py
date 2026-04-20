"""
Test the hyperparameter calculation feature end-to-end.
"""

import sys
import requests


def test_hyperparam_calculation_e2e():
    """End-to-end test for hyperparameter calculation feature."""
    # This test assumes the backend is running
    base_url = "http://localhost:8000"

    # Test the main hyperparameter calculation endpoint
    payload = {
        "model_name": "meta-llama/Llama-3.1-8B",
        "recipe_type": "sft",
        "lora_rank": 64,
    }

    response = requests.post(
        f"{base_url}/hyperparameters/calculate", json=payload, timeout=10
    )

    assert response.status_code == 200
    data = response.json()

    # Verify the response structure
    assert data["success"] is True
    assert data["model_name"] == "meta-llama/Llama-3.1-8B"
    assert data["recipe_type"] == "sft"

    # Verify recommendations
    recs = data["recommendations"]
    assert "learning_rate" in recs
    assert "batch_size" in recs
    assert "lora_rank" in recs
    assert recs["lora_rank"] == 64
    assert recs["batch_size"] == 64  # Expected for 8B model SFT

    # Verify explanation
    exp = data["explanation"]
    assert "learning_rate" in exp
    assert "batch_size" in exp
    assert "lora_rank" in exp
    assert "notes" in exp
    assert "source" in exp

    print("✅ Hyperparameter calculation E2E test passed!")


def test_hyperparam_auto_lr_e2e():
    """Test the auto learning rate endpoint."""
    base_url = "http://localhost:8000"

    response = requests.get(
        f"{base_url}/hyperparameters/auto-lr",
        params={"model_name": "meta-llama/Llama-3.1-8B", "is_lora": "true"},
        timeout=10,
    )

    assert response.status_code == 200
    data = response.json()

    assert "optimal_learning_rate" in data
    assert isinstance(data["optimal_learning_rate"], float)
    assert data["optimal_learning_rate"] > 0

    print("✅ Auto learning rate E2E test passed!")


if __name__ == "__main__":
    print("Running hyperparameter calculation E2E tests...")
    print("Make sure the backend is running on http://localhost:8000")

    try:
        test_hyperparam_calculation_e2e()
        test_hyperparam_auto_lr_e2e()
        print("\n🎉 All E2E tests passed!")
    except Exception as e:
        print(f"\n❌ E2E test failed: {e}")
        sys.exit(1)
