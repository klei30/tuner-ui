"""
Utilities for working with tokenizers. Create new types to avoid needing to import AutoTokenizer and PreTrainedTokenizer.


Avoid importing AutoTokenizer and PreTrainedTokenizer until runtime, because they're slow imports.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeAlias, Optional
import os

if TYPE_CHECKING:
    # this import takes a few seconds, so avoid it on the module import when possible
    from transformers.tokenization_utils import PreTrainedTokenizer

    Tokenizer: TypeAlias = PreTrainedTokenizer
else:
    # make it importable from other files as a type in runtime
    Tokenizer: TypeAlias = Any


def get_tokenizer(model_name: str, hf_token: Optional[str] = None) -> Tokenizer:
    """
    Load a tokenizer for the given model.

    Args:
        model_name: HuggingFace model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
        hf_token: Optional HuggingFace token for gated models. If not provided,
                  will try to use HF_TOKEN environment variable or cached login.

    Returns:
        The loaded tokenizer

    Raises:
        RuntimeError: If tokenizer loading fails (e.g., for gated models without auth)
    """
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    # Use provided token, or fall back to environment variable
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)
    except Exception as e:
        error_msg = str(e)
        # If it's a gated repo error, provide helpful message
        if "gated" in error_msg.lower() or "403" in error_msg or "forbidden" in error_msg.lower():
            raise RuntimeError(
                f"Failed to load tokenizer for {model_name}. "
                f"This model requires HuggingFace authentication. "
                f"Please either:\n"
                f"  1. Add your HuggingFace token in Settings → HuggingFace Integration, or\n"
                f"  2. Run 'huggingface-cli login' in terminal, or\n"
                f"  3. Set HF_TOKEN environment variable\n"
                f"Also ensure you've accepted the model license at https://huggingface.co/{model_name}"
            ) from e
        raise


# Cached version for when token doesn't change
@lru_cache(maxsize=10)
def get_tokenizer_cached(model_name: str) -> Tokenizer:
    """Cached version of get_tokenizer for models that don't need auth."""
    return get_tokenizer(model_name, hf_token=None)
