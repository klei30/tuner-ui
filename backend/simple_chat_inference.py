"""
Simplified chat inference that actually works.
Based on successful test pattern from Tinker docs.
"""
import os
from typing import Optional
import tinker
from tinker.types import SamplingParams, ModelInput
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers as _renderers
from tinker_cookbook.model_info import get_recommended_renderer_name

class SimpleChatClient:
    """Simple synchronous chat client that works"""

    def __init__(self, base_model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.base_model = base_model
        self.service_client = None
        self.sampling_client = None
        self.tokenizer = None

    def initialize(self):
        """Initialize the client (synchronous)"""
        print(f"[SimpleChatClient] Initializing with model: {self.base_model}", flush=True)

        # Create service client
        self.service_client = tinker.ServiceClient()
        print(f"[SimpleChatClient] ServiceClient created", flush=True)

        # Create sampling client for base model
        self.sampling_client = self.service_client.create_sampling_client(
            base_model=self.base_model
        )
        print(f"[SimpleChatClient] SamplingClient created", flush=True)

        # Get tokenizer
        self.tokenizer = get_tokenizer(self.base_model)
        print(f"[SimpleChatClient] Tokenizer loaded", flush=True)

    def chat(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a chat response (synchronous)"""
        if not self.tokenizer:
            self.initialize()

        # Build prompt using cookbook renderer for correct chat template
        renderer_name = get_recommended_renderer_name(self.base_model)
        renderer = _renderers.get_renderer(renderer_name, self.tokenizer)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        full_prompt = renderer.build_generation_prompt(messages)

        print(f"[SimpleChatClient] Encoding prompt...", flush=True)
        prompt_tokens = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        model_input = ModelInput.from_ints(prompt_tokens)

        # Create sampling params
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\nUser:", "\nUser:", "Assistant:"]
        )

        print(f"[SimpleChatClient] Sampling from model...", flush=True)
        # Sample (synchronous - blocks until done)
        result = self.sampling_client.sample(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=1
        ).result()

        print(f"[SimpleChatClient] Decoding response...", flush=True)
        response_tokens = result.sequences[0].tokens
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Clean up response
        response_text = response_text.strip()

        print(f"[SimpleChatClient] Response generated: {len(response_text)} chars", flush=True)
        return response_text


def test_simple_chat():
    """Test the simple chat client"""
    print("\n" + "="*60)
    print("Testing SimpleChatClient")
    print("="*60)

    client = SimpleChatClient()
    client.initialize()

    response = client.chat("Tell me a joke about Python programming")

    print("\n" + "="*60)
    print("Chat Test Result")
    print("="*60)
    print(f"Response: {response}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Set API key from environment
    if "TINKER_API_KEY" not in os.environ:
        raise ValueError(
            "TINKER_API_KEY environment variable must be set. "
            "Please set it before running this test script."
        )
    test_simple_chat()
