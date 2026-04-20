"""
HuggingFace Hub integration service for model deployment.
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class HuggingFaceService:
    """Service for HuggingFace Hub integration."""

    def __init__(self, token: str):
        """
        Initialize HuggingFace service.

        Args:
            token: HuggingFace API token
        """
        if not token:
            raise ValueError("HuggingFace token is required")

        self.api = HfApi(token=token)
        self.token = token

    def verify_token(self) -> Dict[str, Any]:
        """
        Verify token and get user info.

        Returns:
            Dictionary with validation result and user info
        """
        try:
            user_info = self.api.whoami()
            return {
                "valid": True,
                "username": user_info["name"],
                "type": user_info.get("type", "user"),
                "email": user_info.get("email")
            }
        except Exception as e:
            logger.error(f"Failed to verify HuggingFace token: {e}")
            return {
                "valid": False,
                "error": str(e)
            }

    def create_model_repo(
        self,
        repo_name: str,
        private: bool = False,
        exist_ok: bool = True
    ) -> str:
        """
        Create model repository on HuggingFace Hub.

        Args:
            repo_name: Repository name (username/model-name)
            private: Whether to create private repository
            exist_ok: If True, don't raise error if repo exists

        Returns:
            Repository URL

        Raises:
            Exception if repo creation fails
        """
        try:
            repo_url = create_repo(
                repo_id=repo_name,
                token=self.token,
                private=private,
                exist_ok=exist_ok,
                repo_type="model"
            )
            logger.info(f"Created HuggingFace repo: {repo_url}")
            return str(repo_url)
        except Exception as e:
            logger.error(f"Failed to create repo {repo_name}: {e}")
            raise Exception(f"Failed to create repository: {str(e)}")

    def upload_checkpoint(
        self,
        checkpoint_path: Path,
        repo_name: str,
        commit_message: str = "Upload model checkpoint from Tinker"
    ) -> str:
        """
        Upload checkpoint to HuggingFace Hub.

        Args:
            checkpoint_path: Path to checkpoint directory
            repo_name: Repository name (username/model-name)
            commit_message: Git commit message

        Returns:
            Repository URL

        Raises:
            Exception if upload fails
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

        try:
            logger.info(f"Uploading checkpoint from {checkpoint_path} to {repo_name}")

            self.api.upload_folder(
                folder_path=str(checkpoint_path),
                repo_id=repo_name,
                repo_type="model",
                commit_message=commit_message,
                token=self.token
            )

            repo_url = f"https://huggingface.co/{repo_name}"
            logger.info(f"Successfully uploaded checkpoint to {repo_url}")
            return repo_url

        except Exception as e:
            logger.error(f"Failed to upload checkpoint: {e}")
            raise Exception(f"Failed to upload checkpoint: {str(e)}")

    def create_model_card(
        self,
        repo_name: str,
        base_model: str,
        training_config: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create README.md model card with metadata.

        Args:
            repo_name: Repository name
            base_model: Base model name
            training_config: Training configuration dictionary
            metrics: Optional training metrics
        """
        # Extract config details
        recipe_type = training_config.get('recipe_type', 'SFT')
        lora_rank = training_config.get('hyperparameters', {}).get('lora_rank', 32)
        learning_rate = training_config.get('hyperparameters', {}).get('learning_rate', 'N/A')
        batch_size = training_config.get('hyperparameters', {}).get('batch_size', 'N/A')
        dataset_name = training_config.get('dataset_name', 'custom')

        model_card = f"""---
language: en
license: apache-2.0
tags:
- tinker
- lora
- fine-tuned
- {recipe_type.lower()}
base_model: {base_model}
datasets:
- {dataset_name}
library_name: transformers
pipeline_tag: text-generation
---

# {repo_name.split('/')[-1]}

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) using [Tinker](https://thinkingmachines.ai/tinker/).

## Model Details

- **Base Model:** {base_model}
- **Training Type:** {recipe_type}
- **LoRA Rank:** {lora_rank}
- **Learning Rate:** {learning_rate}
- **Batch Size:** {batch_size}
- **Framework:** Tinker API
- **Training Platform:** [Tinker](https://thinkingmachines.ai/tinker/)

## Training Data

- **Dataset:** {dataset_name}

{self._format_metrics_section(metrics) if metrics else ""}

## Usage

### Using Transformers with PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{base_model}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_name}")

# Generate
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### Using HuggingFace Inference API

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/{repo_name}"
headers = {{"Authorization": "Bearer YOUR_HF_TOKEN"}}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({{"inputs": "Your prompt here"}})
print(output)
```

## Training Details

This model was trained using the Tinker platform, which provides:
- Distributed training on high-performance GPUs
- Support for multiple training recipes (SFT, DPO, RL, etc.)
- Automatic hyperparameter optimization
- Real-time training metrics and monitoring

Learn more at [thinkingmachines.ai/tinker](https://thinkingmachines.ai/tinker/)

## Limitations and Biases

This model inherits any limitations and biases from the base model [{base_model}](https://huggingface.co/{base_model}).
Users should be aware of these limitations when deploying the model.

## Citation

If you use this model, please cite:

```bibtex
@misc{{{repo_name.replace('/', '_').replace('-', '_')}}},
  title = {{{repo_name}}},
  author = {{Tinker Platform User}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{repo_name}}}}},
  note = {{Fine-tuned using Tinker Platform}}
}}
```

## Acknowledgments

- Base model: [{base_model}](https://huggingface.co/{base_model})
- Training platform: [Tinker](https://thinkingmachines.ai/tinker/)
- Framework: [HuggingFace Transformers](https://huggingface.co/transformers/)

---

**Note:** This model was automatically deployed from Tinker. For questions or issues, please refer to the [Tinker documentation](https://tinker-docs.thinkingmachines.ai/).
"""

        try:
            # Upload README
            self.api.upload_file(
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                repo_id=repo_name,
                repo_type="model",
                token=self.token
            )
            logger.info(f"Created model card for {repo_name}")
        except Exception as e:
            logger.error(f"Failed to create model card: {e}")
            # Don't raise - model card creation is not critical

    def _format_metrics_section(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for model card."""
        lines = ["## Performance Metrics\n"]

        for key, value in metrics.items():
            # Format the key nicely
            formatted_key = key.replace('_', ' ').title()
            lines.append(f"- **{formatted_key}:** {value}")

        return "\n".join(lines)

    def delete_repo(self, repo_name: str) -> None:
        """
        Delete a repository from HuggingFace Hub.

        Args:
            repo_name: Repository name to delete
        """
        try:
            self.api.delete_repo(repo_id=repo_name, token=self.token)
            logger.info(f"Deleted repo {repo_name}")
        except Exception as e:
            logger.error(f"Failed to delete repo {repo_name}: {e}")
            raise

    def repo_exists(self, repo_name: str) -> bool:
        """
        Check if repository exists.

        Args:
            repo_name: Repository name to check

        Returns:
            True if repo exists, False otherwise
        """
        try:
            self.api.repo_info(repo_id=repo_name, repo_type="model")
            return True
        except Exception:
            return False
