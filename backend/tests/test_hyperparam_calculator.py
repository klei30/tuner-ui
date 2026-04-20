"""
Tests for hyperparameter calculation functionality.
"""

from unittest.mock import patch
from backend.utils.hyperparam_calculator import HyperparamCalculator


class TestHyperparamCalculator:
    """Test cases for hyperparameter calculation logic."""

    def test_get_hidden_size_known_models(self):
        """Test hidden size lookup for known models."""
        # Llama models
        assert HyperparamCalculator._get_hidden_size("meta-llama/Llama-3.2-1B") == 2048
        assert HyperparamCalculator._get_hidden_size("meta-llama/Llama-3.1-8B") == 4096
        assert HyperparamCalculator._get_hidden_size("meta-llama/Llama-3.1-70B") == 8192

        # Qwen models
        assert HyperparamCalculator._get_hidden_size("Qwen/Qwen2.5-1.5B") == 1536
        assert HyperparamCalculator._get_hidden_size("Qwen/Qwen2.5-7B") == 3584

    def test_get_hidden_size_unknown_model(self):
        """Test fallback for unknown models."""
        with patch("backend.utils.hyperparam_calculator.logger") as mock_logger:
            result = HyperparamCalculator._get_hidden_size("unknown-model")
            assert result == 4096  # Default fallback
            mock_logger.warning.assert_called_once()

    def test_get_model_exponent_llama(self):
        """Test model exponent for Llama models."""
        assert (
            HyperparamCalculator._get_model_exponent("meta-llama/Llama-3.1-8B") == 0.781
        )
        assert (
            HyperparamCalculator._get_model_exponent("meta-llama/Llama-3.2-1B") == 0.781
        )

    def test_get_model_exponent_qwen(self):
        """Test model exponent for Qwen models."""
        assert HyperparamCalculator._get_model_exponent("Qwen/Qwen2.5-7B") == 0.0775
        assert HyperparamCalculator._get_model_exponent("Qwen/Qwen2.5-1.5B") == 0.0775

    def test_get_model_exponent_unknown_family(self):
        """Test fallback for unknown model families."""
        with patch("backend.utils.hyperparam_calculator.logger") as mock_logger:
            result = HyperparamCalculator._get_model_exponent("unknown-model")
            assert result == 0.781  # Llama fallback
            mock_logger.warning.assert_called_once()

    def test_get_recommended_lr_lora(self):
        """Test learning rate calculation for LoRA."""
        lr = HyperparamCalculator.get_recommended_lr(
            "meta-llama/Llama-3.1-8B", is_lora=True
        )
        # LR = 5e-5 * 10 * (2000/4096)^0.781
        expected_base = 5e-5 * 10
        expected_scaling = (2000 / 4096) ** 0.781
        expected_lr = expected_base * expected_scaling
        assert abs(lr - expected_lr) < 1e-10

    def test_get_recommended_lr_full_finetune(self):
        """Test learning rate calculation for full fine-tuning."""
        lr = HyperparamCalculator.get_recommended_lr(
            "meta-llama/Llama-3.1-8B", is_lora=False
        )
        # LR = 5e-5 * (2000/4096)^0.781
        expected_base = 5e-5
        expected_scaling = (2000 / 4096) ** 0.781
        expected_lr = expected_base * expected_scaling
        assert abs(lr - expected_lr) < 1e-10

    def test_get_recommended_batch_size_sft(self):
        """Test batch size recommendations for SFT."""
        # Small model
        assert (
            HyperparamCalculator.get_recommended_batch_size(
                "meta-llama/Llama-3.2-1B", recipe_type="sft"
            )
            == 128
        )
        # Medium model
        assert (
            HyperparamCalculator.get_recommended_batch_size(
                "meta-llama/Llama-3.1-8B", recipe_type="sft"
            )
            == 64
        )
        # Large model
        assert (
            HyperparamCalculator.get_recommended_batch_size(
                "meta-llama/Llama-3.1-70B", recipe_type="sft"
            )
            == 32
        )

    def test_get_recommended_batch_size_dpo(self):
        """Test batch size recommendations for DPO."""
        assert (
            HyperparamCalculator.get_recommended_batch_size(
                "meta-llama/Llama-3.1-8B", recipe_type="dpo"
            )
            == 32
        )

    def test_get_recommended_batch_size_rl(self):
        """Test batch size recommendations for RL."""
        assert (
            HyperparamCalculator.get_recommended_batch_size(
                "meta-llama/Llama-3.1-8B", recipe_type="rl"
            )
            == 16
        )

    def test_get_recommended_lora_rank_sft(self):
        """Test LoRA rank recommendations for SFT."""
        # Small model
        assert (
            HyperparamCalculator.get_recommended_lora_rank(
                "meta-llama/Llama-3.2-1B", "sft"
            )
            == 32
        )
        # Medium model
        assert (
            HyperparamCalculator.get_recommended_lora_rank(
                "meta-llama/Llama-3.1-8B", "sft"
            )
            == 64
        )
        # Large model
        assert (
            HyperparamCalculator.get_recommended_lora_rank(
                "meta-llama/Llama-3.1-70B", "sft"
            )
            == 128
        )

    def test_get_recommended_lora_rank_rl(self):
        """Test LoRA rank recommendations for RL."""
        assert (
            HyperparamCalculator.get_recommended_lora_rank(
                "meta-llama/Llama-3.1-8B", "rl"
            )
            == 32
        )

    def test_get_all_recommendations_basic(self):
        """Test complete hyperparameter recommendations."""
        result = HyperparamCalculator.get_all_recommendations(
            model_name="meta-llama/Llama-3.1-8B", recipe_type="sft", lora_rank=64
        )

        # Check structure
        assert "learning_rate" in result
        assert "batch_size" in result
        assert "lora_rank" in result
        assert "adam_beta1" in result
        assert "adam_beta2" in result
        assert "adam_eps" in result
        assert "_metadata" in result

        # Check values
        assert result["batch_size"] == 64  # For 8B model SFT
        assert result["lora_rank"] == 64
        assert result["adam_beta1"] == 0.9
        assert result["adam_beta2"] == 0.95
        assert result["adam_eps"] == 1e-8

        # Check metadata
        metadata = result["_metadata"]
        assert "hidden_size" in metadata
        assert "exponent" in metadata
        assert "lr_formula" in metadata
        assert "notes" in metadata
        assert "source" in metadata

    def test_get_all_recommendations_auto_lora_rank(self):
        """Test that LoRA rank is auto-calculated when not provided."""
        result = HyperparamCalculator.get_all_recommendations(
            model_name="meta-llama/Llama-3.1-8B", recipe_type="sft"
        )

        assert result["lora_rank"] == 64  # Auto-calculated for 8B model

    def test_lr_independence_from_lora_rank(self):
        """Test that learning rate is independent of LoRA rank."""
        lr1 = HyperparamCalculator.get_recommended_lr(
            "meta-llama/Llama-3.1-8B", lora_rank=32
        )
        lr2 = HyperparamCalculator.get_recommended_lr(
            "meta-llama/Llama-3.1-8B", lora_rank=128
        )
        assert lr1 == lr2

    def test_constants(self):
        """Test that constants are correctly defined."""
        assert HyperparamCalculator.LR_BASE == 5e-5
        assert HyperparamCalculator.LORA_MULTIPLIER == 10.0
        assert HyperparamCalculator.MODEL_EXPONENTS == {"llama": 0.781, "qwen": 0.0775}
