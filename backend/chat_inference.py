"""
Chat Inference Module for Tuner UI
Handles chat inference with both base models and fine-tuned models.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Import Tinker API
try:
    import tinker
    from tinker import types
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook import renderers

    TINKER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Tinker not available: {e}")
    TINKER_AVAILABLE = False

# Import database models
try:
    from models import Run, Checkpoint, ModelRegistry
except ImportError:
    from .models import Run, Checkpoint, ModelRegistry


def resolve_model_path(
    run_id: Optional[int] = None,
    model_id: Optional[int] = None,
    checkpoint_id: Optional[int] = None,
    default_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    db_session: Optional[Session] = None,
) -> Tuple[Optional[str], str]:
    """Resolve model path from run_id, model_id, or checkpoint_id"""
    if not db_session:
        logger.error("No database session provided")
        return None, default_model

    # Priority 1: Direct checkpoint selection
    if checkpoint_id is not None:
        checkpoint = db_session.get(Checkpoint, checkpoint_id)
        if checkpoint:
            run = db_session.get(Run, checkpoint.run_id)
            base_model = default_model
            if run and run.config_json:
                base_model = run.config_json.get("base_model", default_model)
            logger.info(
                f"Resolved checkpoint {checkpoint_id}: path={checkpoint.tinker_path}, base={base_model}"
            )
            return checkpoint.tinker_path, base_model

    # Priority 2: Run ID - get latest checkpoint from training run
    if run_id is not None:
        run = db_session.get(Run, run_id)
        if not run:
            logger.warning(f"Run {run_id} not found")
            return None, default_model

        # Get base model from run config
        base_model = default_model
        if run.config_json:
            base_model = run.config_json.get("base_model", default_model)

        # Check database checkpoints
        checkpoints = (
            db_session.query(Checkpoint)
            .filter(Checkpoint.run_id == run_id)
            .order_by(Checkpoint.step.desc())
            .all()
        )

        if checkpoints:
            latest_checkpoint = checkpoints[0]
            logger.info(
                f"Resolved run {run_id}: checkpoint_path={latest_checkpoint.tinker_path}, base={base_model}"
            )
            return latest_checkpoint.tinker_path, base_model

        # Fallback: check checkpoints.jsonl file
        if run.log_path:
            log_dir = Path(run.log_path).parent
            checkpoints_file = log_dir / "checkpoints.jsonl"

            if checkpoints_file.exists():
                try:
                    import json

                    with open(checkpoints_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            last_checkpoint = json.loads(lines[-1])
                            if "sampler_path" in last_checkpoint:
                                model_path = last_checkpoint["sampler_path"]
                                logger.info(
                                    f"Resolved run {run_id} from checkpoints.jsonl: path={model_path}, base={base_model}"
                                )
                                return model_path, base_model
                except Exception as e:
                    logger.error(f"Error reading checkpoints file: {e}")

        logger.warning(f"No checkpoints found for run {run_id}")
        return None, base_model

    # Priority 3: Model ID - registered model from registry
    if model_id is not None:
        model_entry = db_session.get(ModelRegistry, model_id)
        if not model_entry:
            logger.warning(f"Model {model_id} not found in registry")
            return None, default_model

        base_model = model_entry.base_model
        model_path = model_entry.tinker_path

        logger.info(
            f"Resolved model_id {model_id}: path={model_path}, base={base_model}"
        )
        return model_path, base_model

    # Priority 4: Default - use base model
    logger.info(f"Using default base model: {default_model}")
    return None, default_model


class ChatInferenceClient:
    """Client for performing chat inference with Tinker models"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        renderer_name: Optional[str] = None,
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.renderer_name = renderer_name
        self.service_client = None
        self.sampling_client = None
        self.tokenizer = None
        self.renderer = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Tinker service client and sampling client"""
        if self._initialized:
            return

        if not TINKER_AVAILABLE:
            raise RuntimeError("Tinker library is not available")

        try:
            # Initialize service client
            self.service_client = tinker.ServiceClient()
            logger.info("Service client initialized")

            # Get tokenizer
            self.tokenizer = get_tokenizer(self.base_model)
            logger.info(f"Tokenizer loaded for {self.base_model}")

            # Auto-detect renderer if not specified
            if not self.renderer_name:
                if (
                    "llama-3" in self.base_model.lower()
                    or "llama3" in self.base_model.lower()
                ):
                    self.renderer_name = "llama3"
                elif "qwen" in self.base_model.lower():
                    self.renderer_name = "qwen3"
                elif "deepseek" in self.base_model.lower():
                    self.renderer_name = "deepseekv3"
                elif "mistral" in self.base_model.lower():
                    self.renderer_name = "role_colon"
                else:
                    self.renderer_name = "role_colon"

            # Get renderer
            self.renderer = renderers.get_renderer(
                name=self.renderer_name, tokenizer=self.tokenizer
            )
            logger.info(f"Renderer initialized: {self.renderer_name}")

            # Create sampling client - use sync version then await if needed
            if self.model_path:
                logger.info(f"Loading fine-tuned model from: {self.model_path}")
                # Use synchronous create_sampling_client
                self.sampling_client = self.service_client.create_sampling_client(
                    model_path=self.model_path
                )
            else:
                logger.info(f"Loading base model: {self.base_model}")
                self.sampling_client = self.service_client.create_sampling_client(
                    base_model=self.base_model
                )

            self._initialized = True
            logger.info("Chat inference client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize chat client: {e}", exc_info=True)
            raise

    async def chat(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a chat response"""
        if not self._initialized:
            await self.initialize()

        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Build model input using renderer
            model_input = self.renderer.build_generation_prompt(messages)

            # Define sampling parameters
            sampling_params = types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=self.renderer.get_stop_sequences(),
            )

            logger.info(f"Generating response with {len(messages)} messages")

            # Generate response - use synchronous sample
            response = self.sampling_client.sample(
                prompt=model_input, num_samples=1, sampling_params=sampling_params
            ).result()

            # Extract generated tokens
            if response and response.sequences:
                tokens = response.sequences[0].tokens

                # parse_response returns (message_dict, parse_success)
                parsed_message, parse_success = self.renderer.parse_response(tokens)

                if parse_success and parsed_message:
                    response_text = parsed_message["content"]
                    logger.info(f"Generated response: {response_text[:100]}...")
                    return response_text
                else:
                    # Fallback: decode tokens directly
                    response_text = self.tokenizer.decode(
                        tokens, skip_special_tokens=True
                    )
                    logger.info(
                        f"Generated response (fallback decode): {response_text[:100]}..."
                    )
                    return response_text
            else:
                logger.warning("Empty response from sampling client")
                return ""

        except Exception as e:
            logger.error(f"Chat generation failed: {e}", exc_info=True)
            raise

    async def cleanup(self) -> None:
        """Clean up resources"""
        self._initialized = False
        logger.info("Chat inference client cleaned up")
