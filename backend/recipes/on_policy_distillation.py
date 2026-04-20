"""
On-Policy Distillation Recipe - Learn from teacher models via KL divergence
Supports DeepMath and Tulu3 datasets for reasoning and chat distillation
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinker_cookbook import model_info
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDatasetBuilder,
    TeacherConfig,
)


async def run_on_policy_distillation(config: dict):
    """Run on-policy distillation training"""
    # Extract parameters from config
    user_config = config.get("hyperparameters", {})
    base_model = config.get("base_model", "Qwen/Qwen3-8B-Base")
    dataset_name = user_config.get("dataset", "deepmath")
    teacher_model = user_config.get("teacher_model", "Qwen/Qwen3-8B")
    learning_rate = user_config.get("learning_rate", 1e-4)
    groups_per_batch = user_config.get("groups_per_batch", 1024)
    lora_rank = user_config.get("lora_rank", 128)
    max_tokens = user_config.get("max_tokens", 4096)
    kl_penalty_coef = user_config.get("kl_penalty_coef", 1.0)
    kl_discount_factor = user_config.get("kl_discount_factor", 0.0)
    num_substeps = user_config.get("num_substeps", 1)
    eval_every = user_config.get("eval_every", 20)
    save_every = user_config.get("save_every", 20)
    wandb_project = user_config.get("wandb_project")
    wandb_name = user_config.get("wandb_name")

    # Set up logging
    log_path = f"artifacts/run_{config.get('run_id', 'unknown')}/logs"
    os.makedirs(log_path, exist_ok=True)

    # Get renderer
    renderer_name = user_config.get(
        "renderer_name"
    ) or model_info.get_recommended_renderer_name(base_model)

    # Create dataset builder
    dataset_builder = PromptOnlyDatasetBuilder(
        dataset_name=dataset_name,
        groups_per_batch=groups_per_batch,
        group_size=4,  # Default group size
        model_name_for_tokenizer=base_model,
        renderer_name=renderer_name,
    )

    # Create teacher config
    teacher_config = TeacherConfig(
        base_model=teacher_model,
        load_checkpoint_path=user_config.get("teacher_checkpoint"),
    )

    # Create dataset config
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=groups_per_batch,
    )

    # Create training config
    train_config = train_on_policy.Config(
        learning_rate=learning_rate,
        dataset_configs=[dataset_config],
        model_name=base_model,
        lora_rank=lora_rank,
        max_tokens=max_tokens,
        kl_penalty_coef=kl_penalty_coef,
        kl_discount_factor=kl_discount_factor,
        num_substeps=num_substeps,
        loss_fn="importance_sampling",
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=None,
        load_checkpoint_path=None,
        compute_post_kl=False,
        eval_every=eval_every,
        save_every=save_every,
    )

    # Run training
    await train_on_policy.main(train_config)
