### Install Tinker Cookbook (Python)

Source: https://tinker-docs.thinkingmachines.ai/install

Clones the Tinker Cookbook repository and installs it in editable mode. This is recommended for users who want to browse and modify the training code and experiment tools.

```bash
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
# Switch to your virtual environment
pip install -e .
```

--------------------------------

### Install Tinker SDK (Python)

Source: https://tinker-docs.thinkingmachines.ai/install

Installs the Tinker SDK using pip. This provides access to the Python SDK for low-level operations and the Tinker CLI for management functionalities.

```bash
pip install tinker
```

--------------------------------

### Build Supervised Example and Format Output (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This snippet demonstrates how to use `build_supervised_example` to get model inputs and per-token loss weights for supervised learning. It then uses `format_colorized` to visualize the model input, highlighting prompt and completion tokens with corresponding loss weights.

```python
from tinker_cookbook.utils.format_colorized import format_colorized

model_input, weights = renderer.build_supervised_example(messages)

print(format_colorized(model_input.to_ints(), weights, tokenizer))
```

--------------------------------

### Sampling from an Image (Complete Example)

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

A comprehensive example demonstrating how to set up a training client, save weights for sampling, and then perform sampling from an image by asking a question.

```APIDOC
## Putting it together: Sampling from an image

Here's a complete example that creates a training client, saves weights for sampling, and asks a question about an image. You can copy-paste it into an iPython notebook:

### Method
POST (Implied by `service_client.create_lora_training_client_async` and `training_client.save_weights_and_get_sampling_client_async`)

### Endpoint
`/create_lora_training_client` and `/save_weights_and_get_sampling_client` (Implied)

### Parameters
#### Request Body (for creating client)
- **base_model** (str) - Required - The name of the base model to use.
- **rank** (int) - Required - The rank for LoRA training.

#### Request Body (for sampling)
- **prompt** (tinker.ModelInput) - Required - The input, including image data and text.
- **num_samples** (int) - Optional - Number of samples to generate.
- **sampling_params** (tinker.types.SamplingParams) - Required - Sampling parameters, e.g., `max_tokens`.

### Request Example
```python
import requests
import tinker
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

service_client = tinker.ServiceClient()
training_client = await service_client.create_lora_training_client_async(base_model=model_name, rank=32)
sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="sampler")

# Grab an image and ask a question
image_data = requests.get("https://thinkingmachines.ai/blog/on-policy-distillation/images/chess.png").content
model_input = tinker.ModelInput(chunks=[
    tinker.types.EncodedTextChunk(tokens=tokenizer.encode("<|im_start|>user\n<|vision_start|>")),
    tinker.types.ImageChunk(data=image_data, format="png"),
    tinker.types.EncodedTextChunk(tokens=tokenizer.encode("<|vision_end|>What is this?<|im_end|>\n<|im_start|>assistant\n"))
])

result = await sampling_client.sample_async(prompt=model_input, num_samples=1, sampling_params=tinker.types.SamplingParams(max_tokens=100))
print(tokenizer.decode(result.sequences[0].tokens))
```

### Response
#### Success Response (200)
- **sequences** (list) - A list containing the generated sequences.
  - **tokens** (list) - The token IDs of the generated sequence.

#### Response Example
```json
{
  "sequences": [
    {
      "tokens": [1, 2, 3, 4, 5, ...]
    }
  ]
}
```

For higher-level abstractions that handle special tokens automatically, see Rendering: Vision Inputs.
```

--------------------------------

### Full Example: Sampling from an Image with Tinker

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

A complete Python example demonstrating how to set up a Tinker training client, save weights for sampling, and perform a visual question answering task on an image. This includes loading a tokenizer, creating clients, preparing a multimodal input with an image, and generating a response.

```python
import requests
import tinker
from transformers import AutoTokenizer
 
model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
 
service_client = tinker.ServiceClient()
training_client = await service_client.create_lora_training_client_async(base_model=model_name, rank=32)
sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="sampler")
 
# Grab an image and ask a question
image_data = requests.get("https://thinkingmachines.ai/blog/on-policy-distillation/images/chess.png").content
model_input = tinker.ModelInput(chunks=[
    tinker.types.EncodedTextChunk(tokens=tokenizer.encode("<|im_start|>user\n<|vision_start|>")),
    tinker.types.ImageChunk(data=image_data, format="png"),
    tinker.types.EncodedTextChunk(tokens=tokenizer.encode("<|vision_end|>What is this?<|im_end|>\n<|im_start|>assistant\n")),
])
 
result = await sampling_client.sample_async(prompt=model_input, num_samples=1, sampling_params=tinker.types.SamplingParams(max_tokens=100))
print(tokenizer.decode(result.sequences[0].tokens))
```

--------------------------------

### Install Tinker Cookbook locally

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Instructions for a local editable installation of the tinker-cookbook. This is recommended for users who plan to modify or extensively browse the cookbook's code, which is built on top of Tinker.

```bash
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
# Switch to your virtual environment
pip install -e .

```

--------------------------------

### Initialize Tinker Service Client and List Models

Source: https://tinker-docs.thinkingmachines.ai/quickstart.py

This snippet demonstrates how to initialize a Tinker Service Client and retrieve a list of available supported models from the server. It requires the 'tinker' library to be installed. The output is a list of model names printed to the console.

```python
#!/usr/bin/env python3
"""
Auto-generated Python script from markdown code blocks
"""

# --- Code Block 1 ---
import tinker
service_client = tinker.ServiceClient()
print("Available models:")
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)
```

--------------------------------

### Sampling from an Image

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This example demonstrates how to set up a training client, save weights for sampling, and query an image using the sampling client.

```APIDOC
## Example: Sampling from an image

### Description
This comprehensive example illustrates the process of setting up a training client, saving model weights for sampling, and then using the sampling client to ask questions about an image.

### Code
```python
import requests
import tinker
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

service_client = tinker.ServiceClient()
training_client = await service_client.create_lora_training_client_async(base_model=model_name, rank=32)
sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="sampler")

# Grab an image and ask a question
image_data = requests.get("https://thinkingmachines.ai/blog/on-policy-distillation/images/chess.png").content
model_input = tinker.ModelInput(chunks=[
    tinker.types.EncodedTextChunk(tokens=tokenizer.encode("<|im_start|>user\n<|vision_start|>")),
    tinker.types.ImageChunk(data=image_data, format="png"),
    tinker.types.EncodedTextChunk(tokens=tokenizer.encode("<|vision_end|>What is this?<|im_end|>\n<|im_start|>assistant\n"))
])

result = await sampling_client.sample_async(prompt=model_input, num_samples=1, sampling_params=tinker.types.SamplingParams(max_tokens=100))
print(tokenizer.decode(result.sequences[0].tokens))
```

### Notes
For higher-level abstractions that automatically handle special tokens, refer to the "Rendering: Vision Inputs" section [here](/rendering#vision-inputs).
```

--------------------------------

### Visualize Example Data - Python

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

Visualizes the first processed example, displaying input, target, and weight for each token. This is useful for debugging and understanding data processing.

```python
datum0 = processed_examples[0]
print(f"{'Input':<20} {'Target':<20} {'Weight':<10}")
print("-" * 50)
for i, (inp, tgt, wgt) in enumerate(zip(datum0.model_input.to_ints(), datum0.loss_fn_inputs['target_tokens'].tolist(), datum0.loss_fn_inputs['weights'].tolist())):
    print(f"{repr(tokenizer.decode([inp])):<20} {repr(tokenizer.decode([tgt])):<20} {wgt:<10}")
```

--------------------------------

### Prepare Training Data for LoRA

Source: https://tinker-docs.thinkingmachines.ai/quickstart.py

This snippet focuses on preparing training data for a LoRA model. It defines a list of input-output examples, tokenizes them using a tokenizer obtained from the training client, and formats them into `tinker.types.Datum` objects suitable for the loss function. The function `process_example` handles the tokenization and formatting, including creating input and target token sequences and associated weights. It requires the 'tinker' library and a `training_client` instance.

```python
# --- Code Block 3 ---
# Create some training examples
examples = [
    {
        "input": "banana split",
        "output": "anana-bay plit-say"
    },
    {
        "input": "quantum physics",
        "output": "uantum-qay ysics-phay"
    },
    {
        "input": "donut shop",
        "output": "onut-day op-shay"
    },
    {
        "input": "pickle jar",
        "output": "ickle-pay ar-jay"
    },
    {
        "input": "space exploration",
        "output": "ace-spay exploration-way"
    },
    {
        "input": "rubber duck",
        "output": "ubber-ray uck-day"
    },
    {
        "input": "coding wizard",
        "output": "oding-cay izard-way"
    },
]

# Convert examples into the format expected by the training client
from tinker import types

# Get the tokenizer from the training client
tokenizer = training_client.get_tokenizer()

def process_example(example: dict, tokenizer) -> types.Datum:
    # Format the input with Input/Output template
    # For most real use cases, you'll want to use a renderer / chat template,
    # (see later docs) but here, we'll keep it simple.
    prompt = f"English: {example['input']}\nPig Latin:"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:] # We're predicting the next token, so targets need to be shifted.
    weights = weights[1:]

    # A datum is a single training example for the loss function.
    # It has model_input, which is the input sequence that'll be passed into the LLM,
    # loss_fn_inputs, which is a dictionary of extra inputs used by the loss function.
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )

processed_examples = [process_example(ex, tokenizer) for ex in examples]

# Visualize the first example for debugging purposes
datum0 = processed_examples[0]
print(f"{'Input':<20} {'Target':<20} {'Weight':<10}")
print("-" * 50)
for i, (inp, tgt, wgt) in enumerate(zip(datum0.model_input.to_ints(), datum0.loss_fn_inputs['target_tokens'].tolist(), datum0.loss_fn_inputs['weights'].tolist())):
    print(f"{repr(tokenizer.decode([inp])):<20} {repr(tokenizer.decode([tgt])):<20} {wgt:<10}")
```

--------------------------------

### Build Supervised Example and Format Output (Python)

Source: https://tinker-docs.thinkingmachines.ai/rendering

This snippet demonstrates how to use `build_supervised_example` to generate a `ModelInput` and token loss weights for supervised learning. It then formats the output for visualization, highlighting prompt and completion tokens with different colors.

```Python
model_input, weights = renderer.build_supervised_example(messages)

from tinker_cookbook.utils.format_colorized import format_colorized
print(format_colorized(model_input.to_ints(), weights, tokenizer))
```

--------------------------------

### Compute Prompt Logprobs (Python)

Source: https://tinker-docs.thinkingmachines.ai/quickstart.py

This example demonstrates how to compute the log probabilities for a given prompt using the Tinker sampling client. It assumes that the tokenizer and sampling client have already been set up. The result is the log probabilities for each token in the prompt.

```python
sampling_client.compute_logprobs(prompt).result()
```

--------------------------------

### Run Twenty Questions RL Training - Shell

Source: https://tinker-docs.thinkingmachines.ai/rl/rl-envs

Command to execute the training script for the Twenty Questions RL example. This script demonstrates a multi-step environment where an agent learns to ask questions to guess a hidden word, utilizing Llama-3.1-8B-Instruct models.

```shell
python -m tinker_cookbook.recipes.twenty_questions.train
```

--------------------------------

### Create LoRA Training Client Asynchronously

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Offers an asynchronous version for creating a LoRA TrainingClient. This enables initiating training client setup without blocking the main execution thread, suitable for high-concurrency environments.

```python
training_client = await service_client.create_lora_training_client_async(
    base_model="Qwen/Qwen3-8B",
    rank=16
)
```

--------------------------------

### Set TINKER_API_KEY Environment Variable

Source: https://tinker-docs.thinkingmachines.ai/install

Sets the TINKER_API_KEY environment variable to your newly generated API key. This is required for authentication with the Tinker service.

```bash
export TINKER_API_KEY="your_api_key"
```

--------------------------------

### Save Weights and Get SamplingClient Asynchronously (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Offers an asynchronous version of saving current model weights and obtaining a SamplingClient. This function facilitates non-blocking saving and subsequent inference setup, suitable for applications requiring high concurrency.

```python
async def save_weights_and_get_sampling_client_async(
        name: str | None = None,
        retry_config: RetryConfig | None = None) -> SamplingClient:
    """
    Async version of save_weights_and_get_sampling_client.
    """
    pass
```

--------------------------------

### Get Server Capabilities (Async)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Asynchronously retrieves the server's supported features and capabilities.

```APIDOC
## Get Server Capabilities (Async)

### Description
Provides an asynchronous version of `get_server_capabilities` to query the server's supported features and capabilities without blocking the event loop.

### Method
`get_server_capabilities_async`

### Endpoint
N/A (Client-side method)

### Parameters
None

### Request Example
```python
async def get_caps():
    capabilities = await service_client.get_server_capabilities_async()
    print(f"Supported models: {capabilities.supported_models}")
```

### Response
#### Success Response (200)
- **supported_models** (list[str]) - A list of model names supported by the server.
- **max_batch_size** (int) - The maximum batch size allowed for requests.

#### Response Example
```json
{
  "supported_models": ["Qwen/Qwen3-8B", "meta-llama/Llama-2-7b-hf"],
  "max_batch_size": 64
}
```
```

--------------------------------

### Get Recommended Learning Rate for Llama/Qwen Models

Source: https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams

Calculates the recommended learning rate for Llama or Qwen models based on their architecture. This function utilizes predefined constants for base learning rate, LoRA multiplier, and model-specific exponents to determine an optimal learning rate, aiming to minimize loss and maximize evaluation performance.

```python
from tinker_cookbook.hyperparam_utils import get_lr
model_name = "meta-llama/Llama-3.2-1B"
recommended_lr = get_lr(model_name)
print(f"Recommended LR: {recommended_lr}")
```

--------------------------------

### Prepare Training Data for Tinker

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

Processes a list of English to Pig Latin translation examples into the format required by the Tinker training client. It uses the client's tokenizer to encode prompts and completions, creating Datum objects for the loss function.

```python
# Create some training examples
examples = [
    {
        "input": "banana split",
        "output": "anana-bay plit-say"
    },
    {
        "input": "quantum physics",
        "output": "uantum-qay ysics-phay"
    },
    {
        "input": "donut shop",
        "output": "onut-day op-shay"
    },
    {
        "input": "pickle jar",
        "output": "ickle-pay ar-jay"
    },
    {
        "input": "space exploration",
        "output": "ace-spay exploration-way"
    },
    {
        "input": "rubber duck",
        "output": "ubber-ray uck-day"
    },
    {
        "input": "coding wizard",
        "output": "oding-cay izard-way"
    },
]
 
# Convert examples into the format expected by the training client
from tinker import types
 
# Get the tokenizer from the training client
tokenizer = training_client.get_tokenizer()

def process_example(example: dict, tokenizer) -> types.Datum:
    # Format the input with Input/Output template
    # For most real use cases, you'll want to use a renderer / chat template,
    # (see later docs) but here, we'll keep it simple.
    prompt = f"English: {example['input']}\nPig Latin:"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:] # We're predicting the next token, so targets need to be shifted.
    weights = weights[1:]

    # A datum is a single training example for the loss function.
    # It has model_input, which is the input sequence that'll be passed into the LLM,
    # loss_fn_inputs, which is a dictionary of extra inputs used by the loss function.
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )
 
processed_examples = [process_example(ex, tokenizer) for ex in examples]
```

--------------------------------

### Save Weights and Get SamplingClient (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Saves the current model weights and immediately creates a SamplingClient for inference. An optional name can be provided for the saved weights, though it's currently ignored for ephemeral saves. This is useful after training to quickly start inference.

```python
def save_weights_and_get_sampling_client(
        name: str | None = None,
        retry_config: RetryConfig | None = None) -> SamplingClient:
    """
    Save current weights and create a SamplingClient for inference.
    Args:
      * `name`: Optional name for the saved weights (currently ignored for ephemeral saves)
      * `retry_config`: Optional configuration for retrying failed requests
    Returns:
      * `SamplingClient` configured with the current model weights
    """
    pass

# After training, create a sampling client directly
sampling_client = training_client.save_weights_and_get_sampling_client()

# Now use it for inference
prompt = types.ModelInput.from_ints(tokenizer.encode("Hello"))
params = types.SamplingParams(max_tokens=20)
result = sampling_client.sample(prompt, 1, params).result()
```

--------------------------------

### Get Server Capabilities

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Retrieves the server's supported features, capabilities, and available models.

```APIDOC
## Get Server Capabilities

### Description
Queries the Tinker API server to retrieve information about its supported features, capabilities, and available models.

### Method
`get_server_capabilities`

### Endpoint
N/A (Client-side method)

### Parameters
None

### Request Example
```python
capabilities = service_client.get_server_capabilities()
print(f"Supported models: {capabilities.supported_models}")
print(f"Max batch size: {capabilities.max_batch_size}")
```

### Response
#### Success Response (200)
- **supported_models** (list[str]) - A list of model names supported by the server.
- **max_batch_size** (int) - The maximum batch size allowed for requests.

#### Response Example
```json
{
  "supported_models": ["Qwen/Qwen3-8B", "meta-llama/Llama-2-7b-hf"],
  "max_batch_size": 64
}
```
```

--------------------------------

### Get Sampler Information (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Retrieves detailed information about a specific sampler using its sampler ID. This function returns an APIFuture containing a GetSamplerResponse object with sampler details such as the base model and model path. Both synchronous and asynchronous usage patterns are demonstrated in the example.

```python
def get_sampler(sampler_id: str) -> APIFuture[types.GetSamplerResponse]

Get sampler information.
Args:
  * `sampler_id`: The sampler ID (sampling_session_id) to get information for


Returns:
  * An `APIFuture` containing the `GetSamplerResponse` with sampler details


Example:
```
# Sync usage
future = rest_client.get_sampler("session-id:sample:0")
response = future.result()
print(f"Base model: {response.base_model}")
print(f"Model path: {response.model_path}")
 
# Async usage
response = await rest_client.get_sampler("session-id:sample:0")
print(f"Base model: {response.base_model}")
```
```

--------------------------------

### Create LoRA Training Client with Base Model

Source: https://tinker-docs.thinkingmachines.ai/quickstart.py

This code initializes a LoRA (Low-Rank Adaptation) training client using a specified base model. It depends on an existing `ServiceClient` instance. The `base_model` parameter is a string representing the model to be fine-tuned.

```python
# --- Code Block 2 ---
base_model = "Qwen/Qwen3-VL-30B-A3B-Instruct"
training_client = service_client.create_lora_training_client(
    base_model=base_model
)
```

--------------------------------

### Command Line: Generate Distillation Data

Source: https://tinker-docs.thinkingmachines.ai/supervised-learning/prompt-distillation

This command initiates the process of creating prompt distillation data using the Tinker Cookbook's `create_data` script. It specifies the output file path for the generated dataset. The script will use the configured teacher model to produce examples.

```bash
python -m tinker_cookbook.recipes.prompt_distillation.create_data \
  output_file=/tmp/tinker-datasets/prompt_distillation_lang.jsonl
```

--------------------------------

### SessionStartEvent Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Represents an event signaling the start of a session.

```APIDOC
## SessionStartEvent Objects
```
class SessionStartEvent(BaseModel)
```

#### `event`
Telemetry event type
#### `severity`
Log severity level
```

--------------------------------

### Tinker Forward-Backward Loss Function Examples (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Demonstrates how to use Tinker's `forward_backward_async` function with different loss functions, such as 'importance_sampling' (REINFORCE) and 'ppo' (PPO with clipping). It shows the expected input format for training data, including model inputs and loss function-specific inputs like target tokens, logprobs, and advantages. The output is a `ForwardBackwardOutput` object containing results from the forward and backward passes.

```python
import tinker
import torch
from tinker import TensorData

# Create training data with required inputs
datum = tinker.Datum(
    model_input=input_tokens,
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
        "logprobs": TensorData.from_torch(torch.tensor(sampling_logprobs)),  # Reference logprobs
        "advantages": TensorData.from_torch(torch.tensor(advantages)),
    }
)

# Option 1: Use importance sampling REINFORCE
fwd_bwd_result = await training_client.forward_backward_async(
    [datum], loss_fn="importance_sampling"
)

# Option 2: Use PPO with clipping
fwd_bwd_result = await training_client.forward_backward_async(
    [datum], loss_fn="ppo"
)
```

--------------------------------

### Multimodal Inference with Image and Text (Python)

Source: https://tinker-docs.thinkingmachines.ai/quickstart.py

This snippet demonstrates how to perform multimodal inference using the Tinker client. It involves initializing a LoRA training client, setting up a sampling client, and then feeding both an image and text prompt to the model. The output is the decoded text response from the model.

```python
import requests
import tinker
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

service_client = tinker.ServiceClient()
training_client = await service_client.create_lora_training_client_async(base_model=model_name, rank=32)
sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="sampler")

# Grab an image and ask a question
image_data = requests.get("https://thinkingmachines.ai/blog/on-policy-distillation/images/chess.png").content
model_input = tinker.ModelInput(chunks=[
    tinker.types.EncodedTextChunk(tokens=tokenizer.encode("<|im_start|>user\n<|vision_start|>")),
    tinker.types.ImageChunk(data=image_data, format="png"),
    tinker.types.EncodedTextChunk(tokens=tokenizer.encode("<|vision_end|>What is this?<|im_end|>\n<|im_start|>assistant\n"))
])

result = await sampling_client.sample_async(prompt=model_input, num_samples=1, sampling_params=tinker.types.SamplingParams(max_tokens=100))
print(tokenizer.decode(result.sequences[0].tokens))
```

--------------------------------

### Save Weights and Get Sampling Client Async

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Asynchronously saves the current model weights and creates a SamplingClient for inference.

```APIDOC
## POST /sampling_clients/save_and_create/async

### Description
Asynchronously saves the current model weights and creates a SamplingClient for inference.

### Method
POST

### Endpoint
/sampling_clients/save_and_create/async

### Parameters
#### Query Parameters
- **name** (str | None) - Optional - A name for the saved weights (currently ignored for ephemeral saves).
- **retry_config** (RetryConfig | None) - Optional - Configuration for retrying failed requests.

### Request Example
```json
{
  "name": "my-inference-weights",
  "retry_config": null
}
```

### Response
#### Success Response (200)
- **sampling_client** (SamplingClient) - An asynchronously usable SamplingClient configured with the current model weights.

#### Response Example
```json
{
  "sampling_client": { ... SamplingClient object ... }
}
```
```

--------------------------------

### Shortcut to Save Weights and Get Sampling Client using Python

Source: https://tinker-docs.thinkingmachines.ai/save-load

A convenience method that combines saving weights for sampling and creating a sampling client in a single step. It requires a `name` for the checkpoint.

```python
sampling_client = training_client.save_weights_and_get_sampling_client(name="0000")
```

--------------------------------

### Get Model Information

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Retrieves information about the current model, including its configuration and metadata. Returns a `GetInfoResponse` object. An asynchronous version is available.

```python
def get_info() -> types.GetInfoResponse:
    """
    Get information about the current model.
    Returns:
      * `GetInfoResponse` with model configuration and metadata
    """
    pass

# Example:
# info = training_client.get_info()
# print(f"Model ID: {info.model_data.model_id}")
# print(f"Base model: {info.model_data.model_name}")
# print(f"LoRA rank: {info.model_data.lora_rank}")
```

```python
async def get_info_async() -> types.GetInfoResponse:
    """
    Async version of get_info.
    """
    pass
```

--------------------------------

### Compute Log Probabilities for Prompt Tokens (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/samplingclient

Illustrates how to compute the log probabilities for each token in a given prompt using the SamplingClient. This is useful for analyzing the likelihood of specific token sequences. The example shows encoding the prompt, calling the `compute_logprobs` method, and then iterating through the results to print the log probabilities. The method returns `None` for tokens where probabilities could not be computed.

```python
from tinker import types

# Assuming sampling_client and tokenizer are already initialized
prompt = types.ModelInput.from_ints(tokenizer.encode("Hello world"))

# Synchronous call
# future = sampling_client.compute_logprobs(prompt)
# logprobs = future.result()

# Asynchronous call example
logprobs = await sampling_client.compute_logprobs_async(prompt)

for i, logprob in enumerate(logprobs):
    if logprob is not None:
        print(f"Token {i}: logprob = {logprob:.4f}")
```

--------------------------------

### Generate Text with Top-K Prompt Logprobs (Python)

Source: https://tinker-docs.thinkingmachines.ai/quickstart.py

This snippet illustrates generating text while also capturing the top-k log probabilities for each token in the prompt. It requires a tokenizer and sampling client. The output includes the generated text and a detailed breakdown of the top-k token probabilities for the prompt.

```python
sample_response = sampling_client.sample(
    prompt=prompt,
    num_samples=1,
    sampling_params=tinker.SamplingParams(max_tokens=1),
    include_prompt_logprobs=True,
    topk_prompt_logprobs=5,
).result()

# example:
# [None,
#  [(14924, -1.17005), (755, -2.23255), (2, -2.73255), (791, -3.67005), (16309, -4.29505)],
#  [(25, -1.64629), (3137, -2.39629), (11630, -2.89629), (21460, -3.83379), (14881, -4.02129)],
#  [(41, -3.49866), (42, -3.49866), (49, -4.24866), (38, -4.37366), (54, -4.49866)],
#  [(311, -1.00217), (656, -2.25217), (2057, -2.75217), (649, -3.25217), (10470, -3.37717)],
#  ...]
sample_response.topk_prompt_logprobs
```

--------------------------------

### Generate Text with Prompt Logprobs (Python)

Source: https://tinker-docs.thinkingmachines.ai/quickstart.py

This snippet shows how to generate text using the Tinker sampling client and include prompt log probabilities in the response. It requires a tokenizer and a sampling client to be initialized. The output includes the generated text and its associated log probabilities.

```python
prompt = types.ModelInput.from_ints(tokenizer.encode("How many r's are in the word strawberry?"))
sample_response = sampling_client.sample(
    prompt=prompt,
    num_samples=1,
    sampling_params=tinker.SamplingParams(max_tokens=1),  # Must be at least 1 token, represents prefill step
    include_prompt_logprobs=True,
).result()

# example: [None, -9.54505, -1.64629, -8.81116, -3.50217, -8.25927, ...]
print(sample_response.prompt_logprobs)
```

--------------------------------

### Get Server Capabilities

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Retrieves the server's supported features, models, and operational limits using the get_server_capabilities method. This is useful for understanding the environment's capacity before initiating operations.

```python
capabilities = service_client.get_server_capabilities()
print(f"Supported models: {capabilities.supported_models}")
print(f"Max batch size: {capabilities.max_batch_size}")
```

--------------------------------

### Get Sampler API

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Retrieves detailed information about a specific sampler, identified by its sampler ID.

```APIDOC
## GET /get_sampler

### Description
Get sampler information.

### Method
GET

### Endpoint
/get_sampler

### Parameters
#### Path Parameters
None

#### Query Parameters
- **sampler_id** (str) - Required - The sampler ID (sampling_session_id) to get information for

### Request Example
(No request body for GET requests)

### Response
#### Success Response (200)
- **base_model** (str) - The base model used by the sampler.
- **model_path** (str) - The path to the model files.

#### Response Example
```json
{
  "base_model": "tinker-model-v1",
  "model_path": "/path/to/model/files"
}
```
```

--------------------------------

### Define an RL Environment Interface - Python

Source: https://tinker-docs.thinkingmachines.ai/rl/rl-envs

Implements the core interface for a stateful RL environment. It requires implementing `initial_observation` to provide the starting state and `step` to process agent actions and return results. This environment operates on tokens for compatibility with training code.

```python
class Env:
    """
    Stateful environment that a single agent interacts with.
    Discard after running for one episode.
    """

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        raise NotImplementedError

    async def step(self, action: Action) -> StepResult:
        raise NotImplementedError
```

--------------------------------

### Get Session API

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Retrieves detailed information about a specific session, including all associated training runs and samplers.

```APIDOC
## GET /get_session

### Description
Get session information including all training runs and samplers.

### Method
GET

### Endpoint
/get_session

### Parameters
#### Path Parameters
None

#### Query Parameters
- **session_id** (str) - Required - The session ID to get information for

### Request Example
(No request body for GET requests)

### Response
#### Success Response (200)
- **training_run_ids** (array) - A list of training run IDs associated with the session.
- **sampler_ids** (array) - A list of sampler IDs associated with the session.

#### Response Example
```json
{
  "training_run_ids": ["run-id-1", "run-id-2"],
  "sampler_ids": ["session-id:sample:0", "session-id:sample:1"]
}
```
```

--------------------------------

### Save Weights and Get Sampling Client

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Saves the current model weights and then creates a SamplingClient for inference using these newly saved weights.

```APIDOC
## POST /sampling_clients/save_and_create

### Description
Saves the current model weights and creates a SamplingClient for inference using these weights.

### Method
POST

### Endpoint
/sampling_clients/save_and_create

### Parameters
#### Query Parameters
- **name** (str | None) - Optional - A name for the saved weights (currently ignored for ephemeral saves).
- **retry_config** (RetryConfig | None) - Optional - Configuration for retrying failed requests.

### Request Example
```json
{
  "name": "my-inference-weights",
  "retry_config": null
}
```

### Response
#### Success Response (200)
- **sampling_client** (SamplingClient) - A SamplingClient configured with the current model weights.

#### Response Example
```json
{
  "sampling_client": { ... SamplingClient object ... }
}
```
```

--------------------------------

### Get Server Capabilities Asynchronously

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Provides an asynchronous method to query the server's supported features and capabilities. This allows for non-blocking calls, improving performance in applications requiring concurrent operations.

```python
capabilities = await service_client.get_server_capabilities_async()
```

--------------------------------

### Get Checkpoint Archive URL API

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Retrieves a signed URL to download a checkpoint archive. Supports both synchronous and asynchronous operations.

```APIDOC
## GET CHECKPOINT ARCHIVE URL

### Description
Gets a signed URL to download a specific checkpoint archive for a given training run.

### Method
GET

### Endpoint
`/training_runs/{training_run_id}/checkpoints/{checkpoint_id}/archive/url`

### Parameters
#### Path Parameters
- **training_run_id** (string) - Required - The ID of the training run.
- **checkpoint_id** (string) - Required - The ID of the checkpoint to download.

### Request Example
```python
# Synchronous
future = rest_client.get_checkpoint_archive_url("run-id", "checkpoint-123")
response = future.result()

# Asynchronous
response = await rest_client.get_checkpoint_archive_url_async("run-id", "checkpoint-123")

print(f"Download URL: {response.url}")
print(f"Expires at: {response.expires_at}")
# Use the URL to download the archive with your preferred HTTP client
```

### Response
#### Success Response (200)
- **url** (string) - The signed URL for downloading the checkpoint archive.
- **expires_at** (string) - The expiration timestamp of the signed URL.

#### Response Example
```json
{
  "url": "https://example.com/signed/url/to/checkpoint.zip?expires=...&signature=...",
  "expires_at": "2023-10-27T10:00:00Z"
}
```
```

--------------------------------

### Compute Forward and Backward Pass for Gradients

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Shows how to compute both the forward and backward passes to calculate gradients for training. This method is crucial for model optimization. It returns an APIFuture with outputs, loss, and gradients. An example of subsequent parameter update using optim_step is also included.

```python
data = [types.Datum(
    model_input=types.ModelInput.from_ints(tokenizer.encode("Hello")),
    loss_fn_inputs={"target_tokens": types.ModelInput.from_ints(tokenizer.encode("world"))}
)]

# Compute gradients
fwdbwd_future = training_client.forward_backward(data, "cross_entropy")

# Update parameters
optim_future = training_client.optim_step(
    types.AdamParams(learning_rate=1e-4)
)

fwdbwd_result = await fwdbwd_future
print(f"Loss: {fwdbwd_result.loss}")
```

--------------------------------

### Get Checkpoint Archive URL Async (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

An asynchronous version of `get_checkpoint_archive_url`. This function directly returns the `CheckpointArchiveUrlResponse`, simplifying the asynchronous retrieval of checkpoint download URLs. Dependencies: `types` for type hinting.

```python
async def get_checkpoint_archive_url_async(
        training_run_id: types.ModelID,
        checkpoint_id: str) -> types.CheckpointArchiveUrlResponse:
    # ... implementation details ...
    pass
```

--------------------------------

### Process Image Input for Model

Source: https://tinker-docs.thinkingmachines.ai/quickstart.py

This snippet demonstrates how to prepare image data for a Tinker model. It fetches an image using the `requests` library, encodes it, and then constructs a `tinker.ModelInput` object containing both encoded text and image chunks. This is typically used for multimodal models. It requires the `requests` and `tinker` libraries.

```python
# --- Code Block 4 ---
import requests
import tinker
from tinker import types

image_data = requests.get("https://thinkingmachines.ai/blog/on-policy-distillation/images/chess.png").content
model_input = tinker.ModelInput(chunks=[
  types.EncodedTextChunk(tokens=tokenizer.encode("<|im_start|>user\n<|vision_start|>")),
  types.ImageChunk(data=image_data, format="png"),
  types.EncodedTextChunk(tokens=tokenizer.encode("<|vision_end|>What is this?<|im_end|>\n<|im_start|>assistant\n"))
])
```

--------------------------------

### Call Custom Loss Function with forward_backward_custom in Python

Source: https://tinker-docs.thinkingmachines.ai/llms

Demonstrates how to invoke a custom loss function, such as `logprob_squared_loss`, using Tinker's `forward_backward_custom` method. This example shows the integration pattern for applying user-defined loss calculations within the training process.

```python
loss, metrics = training_client.forward_backward_custom(data, logprob_squared_loss)
```

--------------------------------

### List Training Runs with Pagination using RestClient

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Allows listing training runs with support for pagination. The `list_training_runs` method includes `limit` and `offset` parameters for controlling the number of results and the starting point, returning a `ConcurrentFuture` containing the training runs and cursor information. An asynchronous version (`list_training_runs_async`) is also available.

```python
future = rest_client.list_training_runs(limit=50)
response = future.result()
print(f"Found {len(response.training_runs)} training runs")
print(f"Total: {response.cursor.total_count}")
# Get next page
next_page = rest_client.list_training_runs(limit=50, offset=50)
```

```python
async def list_runs(rest_client, limit=20, offset=0):
    response = await rest_client.list_training_runs_async(limit=limit, offset=offset)
    print(f"Found {len(response.training_runs)} training runs")
    print(f"Total: {response.cursor.total_count}")
```

--------------------------------

### Initialize ServiceClient and Create Clients

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Demonstrates the initialization of the ServiceClient and the creation of specialized clients for training, sampling, and REST API operations. The ServiceClient acts as the main entry point, and different clients are generated based on the required tasks.

```python
client = ServiceClient()

training_client = client.create_lora_training_client(base_model="Qwen/Qwen3-8B")

sampling_client = client.create_sampling_client(base_model="Qwen/Qwen3-8B")

rest_client = client.create_rest_client()
```

--------------------------------

### Get Default Learning Rate Recommendation (Python)

Source: https://tinker-docs.thinkingmachines.ai/supervised-learning/sweep-case-study

Retrieves the recommended default learning rate for a given model. This is useful for establishing a baseline for hyperparameter sweeps. It takes a model identifier string as input and outputs a float representing the learning rate.

```python
from tinker_cookbook.hyperparam_utils import get_lr
print(get_lr("meta-llama/Llama-3.1-8B"))
```

--------------------------------

### Get ModelInput length

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

This method returns the total context length of a ModelInput object. It is an instance method and returns an integer representing the length.

```python
def length() -> int

```

--------------------------------

### Get Training Run by Tinker Path using RestClient

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Enables retrieval of training run information using a Tinker path. This functionality is available in both synchronous (`get_training_run_by_tinker_path`) and asynchronous (`get_training_run_by_tinker_path_async`) forms, allowing flexibility in how the data is accessed.

```python
future = rest_client.get_training_run_by_tinker_path("tinker://run-id/weights/checkpoint-001")
response = future.result()
print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
```

```python
async def get_run_by_path(rest_client, tinker_path):
    response = await rest_client.get_training_run_by_tinker_path_async(tinker_path)
    print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
```

--------------------------------

### Get Model Tokenizer

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Provides access to the tokenizer associated with the current model. Returns a `PreTrainedTokenizer` object compatible with the model, allowing for text encoding and decoding. An asynchronous version is not explicitly shown but implied by the pattern.

```python
def get_tokenizer() -> PreTrainedTokenizer:
    """
    Get the tokenizer for the current model.
    Returns:
      * `PreTrainedTokenizer` compatible with the model
    """
    pass

# Example:
# tokenizer = training_client.get_tokenizer()
# tokens = tokenizer.encode("Hello world")
# text = tokenizer.decode(tokens)
```

--------------------------------

### Get Checkpoint Archive URL from Tinker Path API

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Retrieves a signed URL to download a checkpoint archive using its Tinker path. Supports both synchronous and asynchronous operations.

```APIDOC
## GET CHECKPOINT ARCHIVE URL FROM TINKER PATH

### Description
Gets a signed URL to download a checkpoint archive identified by its Tinker path.

### Method
GET

### Endpoint
`/checkpoints/tinker/{tinker_path}/archive/url`

### Parameters
#### Path Parameters
- **tinker_path** (string) - Required - The Tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001").

### Request Example
```python
# Synchronous
future = rest_client.get_checkpoint_archive_url_from_tinker_path("tinker://run-id/weights/0001")
response = future.result()

# Asynchronous
response = await rest_client.get_checkpoint_archive_url_from_tinker_path_async("tinker://run-id/weights/0001")

print(f"Download URL: {response.url}")
print(f"Expires at: {response.expires_at}")
```

### Response
#### Success Response (200)
- **url** (string) - The signed URL for downloading the checkpoint archive.
- **expires_at** (string) - The expiration timestamp of the signed URL.

#### Response Example
```json
{
  "url": "https://example.com/signed/url/to/checkpoint.zip?expires=...&signature=...",
  "expires_at": "2023-10-27T10:00:00Z"
}
```
```

--------------------------------

### Save LoRA Weights and Sample from Model

Source: https://tinker-docs.thinkingmachines.ai/quickstart.py

This snippet shows how to save the weights of a trained LoRA model and obtain a sampling client to generate text. It then demonstrates how to use the sampling client to generate responses based on a given prompt, with specified sampling parameters like `max_tokens`, `temperature`, and `stop` tokens. The generated sequences are printed to the console. It requires a `training_client` instance and the `tinker` library.

```python
# --- Code Block 6 ---
import tinker
from tinker import types

# First, create a sampling client. We need to transfer weights
sampling_client = training_client.save_weights_and_get_sampling_client(name='pig-latin-model')

# Now, we can sample from the model.
prompt = types.ModelInput.from_ints(tokenizer.encode("English: coffee break\nPig Latin:"))
params = types.SamplingParams(max_tokens=20, temperature=0.0, stop=["\n"]) # Greedy sampling
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
result = future.result()
print("Responses:")
for i, seq in enumerate(result.sequences):
    print(f"{i}: {repr(tokenizer.decode(seq.tokens))}")
```

--------------------------------

### Process Training Data for Tinker (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This function processes raw training examples into the `tinker.types.Datum` format required by the training client. It tokenizes the input and output, creates prompt and completion tokens, and defines target tokens and weights for the loss function. It requires a tokenizer obtained from the `TrainingClient`.

```Python
# Create some training examples
examples = [
    {
        "input": "banana split",
        "output": "anana-bay plit-say"
    },
    {
        "input": "quantum physics",
        "output": "uantum-qay ysics-phay"
    },
    {
        "input": "donut shop",
        "output": "onut-day op-shay"
    },
    {
        "input": "pickle jar",
        "output": "ickle-pay ar-jay"
    },
    {
        "input": "space exploration",
        "output": "ace-spay exploration-way"
    },
    {
        "input": "rubber duck",
        "output": "ubber-ray uck-day"
    },
    {
        "input": "coding wizard",
        "output": "oding-cay izard-way"
    },
]

# Convert examples into the format expected by the training client
from tinker import types

# Get the tokenizer from the training client
tokenizer = training_client.get_tokenizer()

def process_example(example: dict, tokenizer) -> types.Datum:
    # Format the input with Input/Output template
    # For most real use cases, you'll want to use a renderer / chat template,
    # (see later docs) but here, we'll keep it simple.
    prompt = f"English: {example['input']}\nPig Latin:"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:] # We're predicting the next token, so targets need to be shifted.
    weights = weights[1:]

    # A datum is a single training example for the loss function.
    # It has model_input, which is the input sequence that'll be passed into the LLM,
    # loss_fn_inputs, which is a dictionary of extra inputs used by the loss function.
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )

processed_examples = [process_example(ex, tokenizer) for ex in examples]
```

--------------------------------

### Get Session Information (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Retrieves session information, including all associated training runs and samplers, for a given session ID. Returns a Future containing a GetSessionResponse object with lists of training run IDs and sampler IDs.

```python
def get_session(session_id: str) -> ConcurrentFuture[types.GetSessionResponse]

Get session information including all training runs and samplers.
Args:
  * `session_id`: The session ID to get information for


Returns:
  * A `Future` containing the `GetSessionResponse` with training_run_ids and sampler_ids


Example:
```
future = rest_client.get_session("session-id")
response = future.result()
print(f"Training runs: {len(response.training_run_ids)}")
print(f"Samplers: {len(response.sampler_ids)}")
```
```

--------------------------------

### Naive vs. Better Forward/Backward and OptimStep Submission (Python)

Source: https://tinker-docs.thinkingmachines.ai/under-the-hood

Demonstrates two Python code patterns for submitting `forward_backward` and `optim_step` asynchronous requests. The naive approach uses three clock cycles by awaiting each call sequentially, while the better approach submits both requests before awaiting, utilizing a single clock cycle for both operations. This requires the `ServiceClient` and its asynchronous methods.

```python
# ❌ Naive implementation (uses 3 clock cycles):
# Submit forward_backward, gets queued for clock cycle N
fwd_bwd_future = await client.forward_backward_async(batch, loss_fn)

# Wait for it to complete, and for client to receive the result
# Due to communication latency, this happens a little after cycle N+1 started
fwd_bwd_result = await fwd_bwd_future

# Submit optim_step, gets queued for clock cycle N+2
optim_future = await client.optim_step_async(adam_params)

# Wait for it to complete, and for client to receive the result
# This happens a little after cycle N+2 finishes
optim_result = await optim_future

# Total: forward_backward on cycle N, optim_step on cycle N+2
# This takes 3 clock cycles (plus the time we waited before cycle N started)

# ✓ Better implementation (uses 1 clock cycle):
# Submit both requests immediately. They'll both be slotted into the same clock cycle N
fwd_bwd_future = await client.forward_backward_async(batch, loss_fn)
optim_future = await client.optim_step_async(adam_params)

# Now wait for results - both operations happen on cycle N
fwd_bwd_result = await fwd_bwd_future
optim_result = await optim_future

# Total: both operations on cycle N
# This takes 1 clock cycle
```

--------------------------------

### Initialize and Use TrainingClient

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Demonstrates how to create a TrainingClient instance using a service client and then perform training operations such as forward/backward passes and optimization steps. It also shows how to save the trained model for inference.

```python
training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-8B")
fwdbwd_future = training_client.forward_backward(training_data, "cross_entropy")
optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
fwdbwd_result = fwdbwd_future.result()  # Wait for gradients
optim_result = optim_future.result()    # Wait for parameter update
sampling_client = training_client.save_weights_and_get_sampling_client("my-model")
```

--------------------------------

### Run Basic Supervised Learning Training

Source: https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-basic

Execute the basic supervised learning script from the command line. This command initiates the fine-tuning process for a specified model on a dataset.

```bash
python -m tinker_cookbook.recipes.sl_basic
```

--------------------------------

### Create LLM-as-a-Judge Evaluation Task (Python)

Source: https://tinker-docs.thinkingmachines.ai/evals

This Python code defines an example task for creating an evaluation using an LLM-as-a-judge approach. It leverages Inspect AI and Tinker's sampling client to set up a grading system where a model grades question-answering pairs. The grader model can be specified separately or default to the model being evaluated.

```python
import tinker
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

QA_DATASET = MemoryDataset(
    name="qa_dataset",
    samples=[
        Sample(
            input="What is the capital of France?",
            target="Paris",
        ),
        Sample(
            input="What is the capital of Italy?",
            target="Rome",
        ),
    ],
)

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(
    base_model="meta-llama/Llama-3.1-8B-Instruct"
)

api = InspectAPIFromTinkerSampling(
    renderer_name="llama3",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    sampling_client=sampling_client,
    verbose=False,
)

GRADER_MODEL = InspectAIModel(api=api, config=InspectAIGenerateConfig())


@task
def example_lm_as_judge() -> Task:
    """
    Example task using LLM-as-a-judge scoring.

    Note: The grader model defaults to the model being evaluated.
    To use a different grader model, specify it with --model-grader when using inspect directly.
    """
    return Task(
        name="llm_as_judge",
        dataset=QA_DATASET,
        solver=generate(),
        scorer=model_graded_qa(
            instructions="Grade strictly against the target text as general answer key and rubric. "
            "Respond 'GRADE: C' if correct or 'GRADE: I' otherwise.",
            partial_credit=False,
            # model parameter is optional - if not specified, uses the model being evaluated
            model=GRADER_MODEL,
        ),
    )

```

--------------------------------

### Get Weights Info by Tinker Path using RestClient

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Fetches checkpoint information using a specified Tinker path. The `get_weights_info_by_tinker_path` method returns an `APIFuture` which can be awaited or have its result retrieved directly, providing details like base model and LoRA rank.

```python
future = rest_client.get_weights_info_by_tinker_path("tinker://run-id/weights/checkpoint-001")
response = future.result()  # or await future
print(f"Base Model: {response.base_model}, LoRA Rank: {response.lora_rank}")
```

--------------------------------

### Perform LoRA Training Forward-Backward and Optimization

Source: https://tinker-docs.thinkingmachines.ai/quickstart.py

This code block executes a single training step for a LoRA model. It performs a forward-backward pass using `training_client.forward_backward` with a specified loss function ('cross_entropy') and then updates the model weights via `training_client.optim_step` using the Adam optimizer with a given learning rate. The process is repeated multiple times, and the average loss per token is printed after each step. It requires `tinker`, `numpy`, and pre-processed training data.

```python
# --- Code Block 5 ---
import numpy as np
import tinker

for _ in range(6):
    fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
    optim_future = training_client.optim_step(tinker.types.AdamParams(learning_rate=1e-4))

    # Wait for the results
    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    # fwdbwd_result contains the logprobs of all the tokens we put in. Now we can compute the weighted
    # average log loss per token.
    logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
    print(f"Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}")
```

--------------------------------

### TensorData Utility Functions: Dtype Conversion

Source: https://tinker-docs.thinkingmachines.ai/llms-full

These utility functions handle the conversion of data types between TensorDtype, NumPy dtypes, and PyTorch dtypes. They are essential for interoperability between different tensor representations. PyTorch conversion requires PyTorch to be installed.

```python
def _convert_tensor_dtype_to_numpy(dtype: TensorDtype) -> npt.DTypeLike:
    """Convert TensorDtype to numpy dtype-like."""
    if dtype == "float32":
        return np.float32
    elif dtype == "int64":
        return np.int64
    else:
        raise ValueError(f"Unsupported TensorDtype: {dtype}")

def _convert_tensor_dtype_to_torch(dtype: TensorDtype) -> "torch.dtype":
    """Convert TensorDtype to torch dtype."""
    if not _HAVE_TORCH:
        raise ImportError("PyTorch is not installed. Cannot convert to torch dtype.")
    import torch

    if dtype == "float32":
        return torch.float32
    elif dtype == "int64":
        return torch.int64
    else:
        raise ValueError(f"Unsupported TensorDtype: {dtype}")


def _convert_numpy_dtype_to_tensor(dtype: np.dtype[Any]) -> TensorDtype:
    """Convert numpy dtype to TensorDtype."""
    if dtype.kind == "f":
        return "float32"
    elif dtype.kind == "i":
        return "int64"
    else:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")

def _convert_torch_dtype_to_tensor(dtype: "torch.dtype") -> TensorDtype:
    """Convert torch dtype to TensorDtype."""
    # torch.dtype objects have .is_floating_point
    if getattr(dtype, "is_floating_point", False):
        return "float32"
    else:
        return "int64"
```

--------------------------------

### Get Checkpoint Archive URL from Tinker Path Async (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

An asynchronous version of `get_checkpoint_archive_url_from_tinker_path`. This function allows for non-blocking retrieval of checkpoint archive URLs using Tinker paths, returning the `CheckpointArchiveUrlResponse` directly. Dependencies: `types` for type hinting.

```python
async def get_checkpoint_archive_url_from_tinker_path_async(
        tinker_path: str) -> types.CheckpointArchiveUrlResponse:
    # ... implementation details ...
    pass
```

--------------------------------

### Create Training Client From State With Optimizer Async

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Asynchronous version of `create_training_client_from_state_with_optimizer`. Resumes training from a checkpoint with optimizer state.

```APIDOC
## `create_training_client_from_state_with_optimizer_async`

### Description
Asynchronous version of `create_training_client_from_state_with_optimizer`. Resumes training from a specified checkpoint, including the optimizer's state.

### Method
POST (Assumed, as it creates a client from a state)

### Endpoint
`/training/resume/async` (Assumed)

### Parameters
#### Path Parameters
- **path** (string) - Required - The path to the checkpoint file (e.g., "tinker://run-id/weights/checkpoint-001").
- **user_metadata** (dict[str, str]) - Optional - User-defined metadata to associate with the training session.

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
async def resume_training():
    training_client = await service_client.create_training_client_from_state_with_optimizer_async(
        "tinker://run-id/weights/checkpoint-001"
    )
    # ... continue training ...
```

### Response
#### Success Response (200)
- **TrainingClient** (object) - An instance of the TrainingClient configured to resume training.

#### Response Example
```json
{
  "message": "Async training client created successfully from checkpoint state."
}
```
```

--------------------------------

### Get Training Run Information using RestClient

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Provides methods to retrieve information about a specific training run by its ID. Includes both synchronous (`get_training_run`) and asynchronous (`get_training_run_async`) versions. The synchronous method returns a `ConcurrentFuture`, while the asynchronous one directly returns the training run data.

```python
future = rest_client.get_training_run("run-id")
response = future.result()
print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
```

```python
async def get_run_info(rest_client, run_id):
    response = await rest_client.get_training_run_async(run_id)
    print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
```

--------------------------------

### Naive Forward-Backward and Optimizer Step (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Demonstrates a naive implementation of submitting forward-backward and optimizer step requests asynchronously, which results in increased clock cycles. This approach leads to less efficient training due to suboptimal resource utilization.

```python
fwd_bwd_future = await client.forward_backward_async(batch, loss_fn)

# Wait for it to complete, and for client to receive the result

```

--------------------------------

### Instantiate and Use SamplingClient for Text Generation (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/samplingclient

Demonstrates how to create a SamplingClient using a base model and then use it to generate text completions. It shows the process of encoding a prompt, setting sampling parameters, making the asynchronous call, and decoding the generated text. This client is essential for interacting with trained or base language models.

```python
from tinker import types

# Assuming service_client and tokenizer are already initialized
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")
prompt = types.ModelInput.from_ints(tokenizer.encode("The weather today is"))
params = types.SamplingParams(max_tokens=20, temperature=0.7)

# Synchronous call (example shown in main description, but async is more common)
# future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
# result = future.result()

# Asynchronous call example
result = await sampling_client.sample_async(prompt=prompt, sampling_params=params, num_samples=1)

for sample in result.samples:
    print(tokenizer.decode(sample.tokens))
```

--------------------------------

### Create Tinker Service Client and List Models

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

Initializes a Tinker ServiceClient to discover available base models for fine-tuning. It retrieves and prints the names of supported models, such as Llama3 and Qwen3 series.

```python
import tinker
service_client = tinker.ServiceClient()
print("Available models:")
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)
```

--------------------------------

### Handling Futures in Sync and Async Python

Source: https://tinker-docs.thinkingmachines.ai/async

Illustrates how to obtain and process results from Tinker API calls using Future objects. In synchronous Python, '.result()' is called on the Future. In asynchronous Python, Futures are awaited twice: once to ensure submission and again to get the result.

```python
# Sync Python example
# future = client.forward_backward(data, loss_fn)
# result = future.result() # Blocks until complete

# Async Python example
# future = await client.forward_backward_async(data, loss_fn)
# result = await future
```

--------------------------------

### Create Training Client From Saved Weights and Optimizer State

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Initializes a TrainingClient by restoring both model weights and optimizer state from a specified path. This allows for exact resumption of training, including momentum and other optimizer-specific parameters.

```python
training_client = service_client.create_training_client_from_state_with_optimizer(
    "tinker://run-id/weights/checkpoint-001"
)
```

--------------------------------

### Python: EncodedTextChunk Type Definition

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the `EncodedTextChunk` class, inheriting from `StrictBase`. It represents a chunk of encoded text, containing an array of `tokens` (token IDs) and a `type` field set to 'encoded_text'. It provides a `length` property to get the number of tokens.

```python
class EncodedTextChunk(StrictBase):
    tokens: Sequence[int]
    """Array of token IDs"""

    type: Literal["encoded_text"] = "encoded_text"

    @property
    def length(self) -> int:
        return len(self.tokens)
```

--------------------------------

### Configuring PPO Loss with Tinker Client

Source: https://tinker-docs.thinkingmachines.ai/llms

Demonstrates how to configure and apply the PPO loss function using the Tinker training client. This asynchronous method allows specifying custom clipping thresholds for the PPO loss calculation. It takes the data, specifies 'ppo' as the loss function, and provides a configuration dictionary for `clip_low_threshold` and `clip_high_threshold`.

```python
fwd_bwd_result = await training_client.forward_backward_async(
    data=data,
    loss_fn="ppo",
    loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}
)
```

--------------------------------

### Plot RL Training Rewards - Python

Source: https://tinker-docs.thinkingmachines.ai/rl/rl-loops

Generates a plot of the total reward over training steps using the metrics saved by the RL training loop. It reads a JSONL file, processes it with pandas, and visualizes the reward curve with matplotlib. Ensure that pandas and matplotlib are installed.

```python
import pandas
import matplotlib.pyplot as plt

metrics_path = "/tmp/tinker-examples/rl-loop/metrics.jsonl"
df = pandas.read_json(metrics_path, lines=True)
plt.plot(df["reward/total"], label="reward/total")
plt.legend()
plt.show()
```

--------------------------------

### Python: GetInfoResponse Type Definition

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the `GetInfoResponse` class, inheriting from `BaseModel`. This class is used for responses to get information about a model, including `model_data`, `model_id`, and optional fields like `is_lora`, `lora_rank`, and `model_name`. It includes Pydantic V2 configuration for protected namespaces.

```python
class GetInfoResponse(BaseModel):
    type: Optional[Literal["get_info"]] = None

    model_data: ModelData

    model_id: ModelID

    is_lora: Optional[bool] = None

    lora_rank: Optional[int] = None

    model_name: Optional[str] = None

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
```

--------------------------------

### Run DPO Training with CLI

Source: https://tinker-docs.thinkingmachines.ai/preferences/dpo-guide

Execute Direct Preference Optimization (DPO) training using the `tinker_cookbook` from the command line. This command initiates the training process, specifying the log path, model name, dataset, renderer, learning rate, and DPO beta parameter.

```bash
python -m tinker_cookbook.recipes.preference.train \
    log_path=/tmp/dpo-hhh-experiment \
    model_name=meta-llama/Llama-3.2-1B \
    dataset=hhh \
    renderer_name=role_colon \
    learning_rate=1e-5 \
    dpo_beta=0.1
```

--------------------------------

### Create Training Client From State (Async)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Asynchronously creates a `TrainingClient` by loading model weights from a specified path.

```APIDOC
## Create Training Client From State (Async)

### Description
Provides an asynchronous version of `create_training_client_from_state` for loading model weights from a Tinker path without blocking the event loop.

### Method
`create_training_client_from_state_async`

### Endpoint
N/A (Client-side method)

### Parameters
#### Path Parameters
- **path** (str) - Required - The Tinker path to the saved weights (e.g., "tinker://run-id/weights/checkpoint-001").

#### Query Parameters
None

#### Request Body
- **user_metadata** (dict[str, str] | None) - Optional - Metadata to attach to the new training run.

### Request Example
```python
async def create_client_from_state():
    training_client = await service_client.create_training_client_from_state_async(
        "tinker://run-id/weights/checkpoint-001"
    )
    # Use training_client asynchronously
```

### Response
#### Success Response (200)
- **TrainingClient** (TrainingClient) - A `TrainingClient` instance loaded with the specified weights.
```

--------------------------------

### Command Line: Train Student Model

Source: https://tinker-docs.thinkingmachines.ai/supervised-learning/prompt-distillation

This command executes the training script from the Tinker Cookbook to fine-tune a student model on the previously generated distillation dataset. The script automatically loads the data and applies the training configurations.

```bash
python -m tinker_cookbook.recipes.prompt_distillation.train
```

--------------------------------

### Define Variance Loss Function in Python

Source: https://tinker-docs.thinkingmachines.ai/llms

Implements a custom loss function to compute the variance across multiple log probability sequences. This function concatenates the input log probabilities, calculates their variance, and returns the result along with a metric. It serves as an example for losses operating on sequences.

```python
def variance_loss(data: list[Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    flat_logprobs = torch.cat(logprobs)
    variance = torch.var(flat_logprobs)
    return variance, {"variance_loss": variance.item()}
```

--------------------------------

### Computing Logprobs for a Sequence

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

This section explains how to compute log probabilities for a given sequence using the sampler, including the prompt logprobs which are returned from the prefill step.

```APIDOC
## Computing logprobs for a sequence

We can use the sampler to compute logprobs for a given sequence as well. This uses the prefill step and is returned as _prompt logprobs_.

### Method
POST (Implied by `sampling_client.sample`)

### Endpoint
`/sample` (Implied)

### Parameters
#### Request Body
- **prompt** (tinker.ModelInput) - Required - The input sequence for which to compute logprobs.
- **num_samples** (int) - Optional - The number of samples to generate. Defaults to 1.
- **sampling_params** (tinker.SamplingParams) - Required - Sampling parameters, including `max_tokens` which must be at least 1 for the prefill step.
- **include_prompt_logprobs** (bool) - Required - Set to `true` to include prompt logprobs.

### Request Example
```python
# Assuming tokenizer and sampling_client are initialized
prompt = types.ModelInput.from_ints(tokenizer.encode("How many r's are in the word strawberry?"))
sample_response = sampling_client.sample(
    prompt=prompt,
    num_samples=1,
    sampling_params=tinker.SamplingParams(max_tokens=1),  # Must be at least 1 token, represents prefill step
    include_prompt_logprobs=True,
).result()
```

### Response
#### Success Response (200)
- **prompt_logprobs** (list) - A list of logprobs for each token in the prompt. The first entry is `None`.

#### Response Example
```json
[null, -9.54505, -1.64629, -8.81116, -3.50217, -8.25927, ...]
```

### Helper Function
The sampling client also has a helper function:
```python
# Assuming prompt and sampling_client are initialized
sampling_client.compute_logprobs(prompt).result()
```
```

--------------------------------

### Get Checkpoint Archive URL (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Retrieves a signed URL to download a checkpoint archive. This function requires both the training run ID and the specific checkpoint ID. It returns a Future object that resolves to a response containing the download URL and its expiration time. Dependencies: `types` for type hinting, `ConcurrentFuture` for asynchronous operations.

```python
def get_checkpoint_archive_url(
    training_run_id: types.ModelID,
    checkpoint_id: str
) -> ConcurrentFuture[types.CheckpointArchiveUrlResponse]:
    # ... implementation details ...
    pass

# Example Usage:
# future = rest_client.get_checkpoint_archive_url("run-id", "checkpoint-123")
# response = future.result()
# print(f"Download URL: {response.url}")
# print(f"Expires at: {response.expires_at}")
```

--------------------------------

### Create Training Client From State With Optimizer

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Creates a `TrainingClient` by loading both model weights and optimizer state from a specified path.

```APIDOC
## Create Training Client From State With Optimizer

### Description
Creates a `TrainingClient` instance by loading both the model weights and the optimizer state (e.g., Adam momentum) from a given Tinker path. This is ideal for resuming training exactly from where it was last saved.

### Method
`create_training_client_from_state_with_optimizer`

### Endpoint
N/A (Client-side method)

### Parameters
#### Path Parameters
- **path** (str) - Required - The Tinker path to the saved weights and optimizer state (e.g., "tinker://run-id/weights/checkpoint-001").

#### Query Parameters
None

#### Request Body
- **user_metadata** (dict[str, str] | None) - Optional - Metadata to attach to the new training run.

### Request Example
```python
# Resume training exactly from the last saved point
training_client = service_client.create_training_client_from_state_with_optimizer(
    "tinker://run-id/weights/checkpoint-001"
)
# Continue training from the loaded state, including optimizer state
```

### Response
#### Success Response (200)
- **TrainingClient** (TrainingClient) - A `TrainingClient` instance loaded with the specified weights and optimizer state.
```

--------------------------------

### Python StrictBase for Image Chunk with Base64 Handling

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the ImageChunk type using StrictBase. This class handles image data as bytes, supporting PNG and JPEG formats. It includes validators and serializers to convert between bytes and base64 strings, and a property to get the length based on expected tokens.

```python
class ImageChunk(StrictBase):
    data: bytes
    """Image data as bytes"""

    format: Literal["png", "jpeg"]
    """Image format"""

    expected_tokens: int | None = None
    """Expected number of tokens this image represents.
    This is only advisory: the tinker backend will compute the number of tokens
    from the image, and we can fail requests quickly if the tokens does not
    match expected_tokens."""

    type: Literal["image"] = "image"

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: Union[bytes, str]) -> bytes:
        """Deserialize base64 string to bytes if needed."""
        if isinstance(value, str):
            return base64.b64decode(value)
        return value

    @field_serializer("data")
    def serialize_data(self, value: bytes) -> str:
        """Serialize bytes to base64 string for JSON."""
        return base64.b64encode(value).decode("utf-8")

    @property
    def length(self) -> int:
        if self.expected_tokens is None:
            raise ValueError("ImageChunk expected_tokens needs to be set in order to compute the length")
        return self.expected_tokens
```

--------------------------------

### Use Qwen3VLRenderer for VLM Prompts (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This snippet illustrates how to initialize and use the `Qwen3VLRenderer` for building generation prompts with multimodal messages. It sets up the tokenizer and image processor for a specific Qwen model and then constructs the prompt from a list of messages containing image and text content.

```python
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.image_processing_utils import get_image_processor

model_name = "Qwen/Qwen3-VL-235B-A22B-Instruct"
tokenizer = tokenizer_utils.get_tokenizer(model_name)
image_processor = get_image_processor(model_name)

renderer = renderers.Qwen3VLRenderer(tokenizer, image_processor)

messages = [
    {
        'role': 'user',
        'content': [
            {'type': 'image', 'image': 'https://thinkingmachines.ai/blog/on-policy-distillation/images/chess.png'},
            {'type': 'text', 'text': 'What is in this image?'},
        ]
    }
]

prompt = renderer.build_generation_prompt(messages)
```

--------------------------------

### Create LoRA Training Client (Async)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Asynchronously creates a `TrainingClient` instance for LoRA fine-tuning.

```APIDOC
## Create LoRA Training Client (Async)

### Description
Provides an asynchronous version of `create_lora_training_client` for creating a `TrainingClient` for LoRA fine-tuning without blocking the event loop.

### Method
`create_lora_training_client_async`

### Endpoint
N/A (Client-side method)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **base_model** (str) - Required - The name of the base model to fine-tune (e.g., "Qwen/Qwen3-8B").
- **rank** (int) - Optional - The LoRA rank, which controls the size of the adaptation matrices. Defaults to 32.
- **seed** (int | None) - Optional - A random seed for initialization. If `None`, a random seed is used.
- **train_mlp** (bool) - Optional - Whether to train the MLP layers. Defaults to `True`.
- **train_attn** (bool) - Optional - Whether to train the attention layers. Defaults to `True`.
- **train_unembed** (bool) - Optional - Whether to train the unembedding layers. Defaults to `True`.
- **user_metadata** (dict[str, str] | None) - Optional - Metadata to attach to the training run.

### Request Example
```python
async def create_lora_client():
    training_client = await service_client.create_lora_training_client_async(
        base_model="Qwen/Qwen3-8B",
        rank=16
    )
    # Use training_client asynchronously
```

### Response
#### Success Response (200)
- **TrainingClient** (TrainingClient) - An instance of `TrainingClient` configured for LoRA training.
```

--------------------------------

### Get Checkpoint Archive URL from Tinker Path (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Retrieves a signed URL for downloading a checkpoint archive using its Tinker path. This is useful when the checkpoint is referenced by a Tinker-specific path rather than traditional IDs. It returns a Future object containing the URL and expiration details. Dependencies: `types` for type hinting, `ConcurrentFuture` for asynchronous operations.

```python
def get_checkpoint_archive_url_from_tinker_path(
        tinker_path: str
) -> ConcurrentFuture[types.CheckpointArchiveUrlResponse]:
    # ... implementation details ...
    pass
```

--------------------------------

### Async Resume Training from Checkpoint with Optimizer State (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Provides an asynchronous method to create a TrainingClient, enabling training resumption from a checkpoint with optimizer state. This is suitable for non-blocking operations in asynchronous applications.

```python
async def create_training_client_from_state_with_optimizer_async(
        path: str,
        user_metadata: dict[str, str] | None = None) -> TrainingClient:
    # Implementation details here
```

--------------------------------

### Create Training Client From State With Optimizer

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Resumes training from a specified checkpoint, including the optimizer's state, to continue training with restored momentum.

```APIDOC
## `create_training_client_from_state_with_optimizer`

### Description
Resumes training from a checkpoint, restoring the optimizer's state to continue training with momentum.

### Method
POST (Assumed, as it creates a client from a state)

### Endpoint
`/training/resume` (Assumed)

### Parameters
#### Path Parameters
- **checkpoint_path** (string) - Required - The path to the checkpoint file (e.g., "tinker://run-id/weights/checkpoint-001").

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
training_client = service_client.create_training_client_from_state_with_optimizer(
    "tinker://run-id/weights/checkpoint-001"
)
```

### Response
#### Success Response (200)
- **TrainingClient** (object) - An instance of the TrainingClient configured to resume training.

#### Response Example
```json
{
  "message": "Training client created successfully from checkpoint state."
}
```
```

--------------------------------

### Train Student Model for Prompt Distillation (Python)

Source: https://tinker-docs.thinkingmachines.ai/supervised-learning/prompt-distillation

This script fine-tunes a student model using a pre-generated distillation dataset. It loads the data and applies optimized training configurations. The script assumes the distillation data is accessible and configured within the Tinker Cookbook environment.

```python
from tinker_cookbook.recipes.prompt_distillation import train

def main():
    # Example usage: trains the student model
    train.main()
    print("Student model training complete.")

if __name__ == "__main__":
    main()
```

--------------------------------

### Create Dummy Training Data

Source: https://tinker-docs.thinkingmachines.ai/clock_cycles.py

Generates a minimal training datum for demonstration purposes. It requires numpy and tinker.types. The datum includes model input and loss function inputs.

```python
import tinker
from tinker import types
import numpy as np

def create_dummy_datum():
    """Create a minimal training datum for demonstration purposes."""
    n = 16000
    return types.Datum(
        model_input=types.ModelInput.from_ints(np.arange(n).tolist()),
        loss_fn_inputs=dict(
            weights=np.ones(16000, dtype=np.float32),
            target_tokens=np.arange(16000, dtype=np.int64),
        ),
    )
```

--------------------------------

### Create Tinker Training Client

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

Creates a Tinker TrainingClient for a specified base model. This client is used for fine-tuning and sampling operations. Ensure the TINKER_API_KEY environment variable is set.

```python
base_model = "Qwen/Qwen3-VL-30B-A3B-Instruct"
training_client = service_client.create_lora_training_client(
    base_model=base_model
)
```

--------------------------------

### Async Create Sampling Client for Text Generation (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Offers an asynchronous version of the SamplingClient creation. This allows for non-blocking text generation requests, improving performance in concurrent applications.

```python
async def create_sampling_client_async(
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None) -> SamplingClient:
    # Implementation details here
```

--------------------------------

### RL Hyperparameter Overview (Markdown)

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Provides an overview of key hyperparameters for reinforcement learning training. It discusses core settings like learning rate and batch/group sizes, along with advanced configurations such as multiple updates per sampling iteration.

```markdown
# RL Hyperparameters

This guide covers the key hyperparameters for reinforcement learning training, from core settings to advanced configurations.

## Core Hyperparameters

### Learning Rate

Similar to the [supervised learning setting](../supervised-learning/sl-hyperparams), the learning rate is the most critical hyperparameter choice. We recommend using the guidance presented there as a starting point for RL experiments as well.


### Batch and Group Sizes

As described in our [RL environments](../rl/rl-envs.mdx) documentation, we use two key parameters:

- **`batch_size`**: The number of unique environments or problems used for training
- **`group_size`**: The number of rollouts performed per unique environment

If you have limited environments or problems available for training, increase the `group_size` to generate more training data. While the total number of rollouts depends on both parameters, we recommend scaling learning rates proportionally to $\text{LR} \propto \sqrt{\text{batch\_size}}$.

## Multiple Updates per Sampling Iteration

The `num_substeps` parameter controls how many policy weight updates are performed on data sampled from the last policy iteration, similar to PPO and GRPO.
```

--------------------------------

### Create LoRA Training Client

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Creates a `TrainingClient` instance specifically configured for LoRA fine-tuning.

```APIDOC
## Create LoRA Training Client

### Description
Creates a `TrainingClient` instance for performing LoRA (Low-Rank Adaptation) fine-tuning on a specified base model. This client is tailored for the training workflow.

### Method
`create_lora_training_client`

### Endpoint
N/A (Client-side method)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **base_model** (str) - Required - The name of the base model to fine-tune (e.g., "Qwen/Qwen3-8B").
- **rank** (int) - Optional - The LoRA rank, which controls the size of the adaptation matrices. Defaults to 32.
- **seed** (int | None) - Optional - A random seed for initialization. If `None`, a random seed is used.
- **train_mlp** (bool) - Optional - Whether to train the MLP layers. Defaults to `True`.
- **train_attn** (bool) - Optional - Whether to train the attention layers. Defaults to `True`.
- **train_unembed** (bool) - Optional - Whether to train the unembedding layers. Defaults to `True`.
- **user_metadata** (dict[str, str] | None) - Optional - Metadata to attach to the training run.

### Request Example
```python
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
    rank=16,
    train_mlp=True,
    train_attn=True
)
# Now use training_client.forward_backward() to train
```

### Response
#### Success Response (200)
- **TrainingClient** (TrainingClient) - An instance of `TrainingClient` configured for LoRA training.
```

--------------------------------

### Initialize and Use RestClient for Tinker API

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Demonstrates how to initialize the RestClient and perform basic operations like retrieving training run details and listing checkpoints. The RestClient is typically obtained via `service_client.create_rest_client()` and requires an internal client for managing HTTP connections.

```python
rest_client = service_client.create_rest_client()
training_run = rest_client.get_training_run("run-id").result()
print(f"Training Run: {training_run.training_run_id}, LoRA: {training_run.is_lora}")
checkpoints = rest_client.list_checkpoints("run-id").result()
print(f"Found {len(checkpoints.checkpoints)} checkpoints")
for checkpoint in checkpoints.checkpoints:
    print(f"  {checkpoint.checkpoint_type}: {checkpoint.checkpoint_id}")
```

--------------------------------

### Create Sampling Client Async

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Asynchronous version of `create_sampling_client`. Creates a SamplingClient for text generation.

```APIDOC
## `create_sampling_client_async`

### Description
Asynchronous version of `create_sampling_client`. Creates a `SamplingClient` for text generation tasks.

### Method
POST (Assumed, as it creates a client)

### Endpoint
`/sampling/create/async` (Assumed)

### Parameters
#### Path Parameters
None

#### Query Parameters
- **model_path** (string) - Optional - Path to saved model weights (e.g., "tinker://run-id/weights/checkpoint-001").
- **base_model** (string) - Optional - Name of base model to use (e.g., "Qwen/Qwen3-8B").
- **retry_config** (object) - Optional - Configuration for retrying failed requests.

#### Request Body
None

### Request Example
```python
async def setup_sampling():
    sampling_client = await service_client.create_sampling_client_async(
        base_model="Qwen/Qwen3-8B"
    )
    # ... use sampling client ...
```

### Response
#### Success Response (200)
- **SamplingClient** (object) - A `SamplingClient` instance configured for text generation.

#### Response Example
```json
{
  "message": "Async sampling client created successfully.",
  "client_id": "async-sampling-xyz789"
}
```
```

--------------------------------

### Run RL Training Loop - Python

Source: https://tinker-docs.thinkingmachines.ai/rl/rl-loops

Executes the Reinforcement Learning training loop from the command line. This command initiates the training process using the default configuration, saving metrics and results to a specified directory.

```shell
python -m tinker_cookbook.recipes.rl_loop
```

--------------------------------

### Create Training Client From Saved Weights Asynchronously

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Provides an asynchronous method to create a TrainingClient from saved model weights. This non-blocking approach is beneficial for applications that need to load model states efficiently without pausing execution.

```python
training_client = await service_client.create_training_client_from_state_async(
    "tinker://run-id/weights/checkpoint-001"
)
```

--------------------------------

### Create Sampling Client for Text Generation (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Creates a SamplingClient for text generation tasks. It can be configured using a model path to saved weights or by specifying a base model name. Optional retry configurations can be provided.

```python
def create_sampling_client(
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None) -> SamplingClient:
    # Implementation details here
```

```python
# Use a base model
sampling_client = service_client.create_sampling_client(
    base_model="Qwen/Qwen3-8B"
)

# Or use saved weights
sampling_client = service_client.create_sampling_client(
    model_path="tinker://run-id/weights/checkpoint-001"
)
```

--------------------------------

### Construct Generation Prompt with Qwen3VLRenderer (Python)

Source: https://tinker-docs.thinkingmachines.ai/rendering

This snippet shows how to initialize and use the `Qwen3VLRenderer` for vision-language models. It processes a list of messages, including an image and text, to build the appropriate prompt for the model, automatically handling special vision tokens.

```Python
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.image_processing_utils import get_image_processor

model_name = "Qwen/Qwen3-VL-235B-A22B-Instruct"
tokenzier = tokenizer_utils.get_tokenizer(model_name)
image_processor = get_image_processor(model_name)

renderer = renderers.Qwen3VLRenderer(tokenizer, image_processor)

messages = [
    {
        'role': 'user',
        'content': [
            {'type': 'image', 'image': 'https://thinkingmachines.ai/blog/on-policy-distillation/images/chess.png'},
            {'type': 'text', 'text': 'What is in this image?'},
        ]
    }
]

prompt = renderer.build_generation_prompt(messages)
```

--------------------------------

### Run Supervised Learning Loop with Specific Learning Rate (Bash)

Source: https://tinker-docs.thinkingmachines.ai/supervised-learning/sweep-case-study

Launches supervised learning experiments in parallel, each with a different learning rate. This command-line interface allows for easy execution of multiple training runs for hyperparameter tuning. It requires the learning rate and a log path for each experiment.

```bash
python -m tinker_cookbook.recipes.sl_loop learning_rate=0.003 log_path=/tmp/sft-lr-sweep/lr-0.003
python -m tinker_cookbook.recipes.sl_loop learning_rate=0.001 log_path=/tmp/sft-lr-sweep/lr-0.001
python -m tinker_cookbook.recipes.sl_loop learning_rate=0.0003 log_path=/tmp/sft-lr-sweep/lr-0.0003
python -m tinker_cookbook.recipes.sl_loop learning_rate=0.0001 log_path=/tmp/sft-lr-sweep/lr-0.0001
python -m tinker_cookbook.recipes.sl_loop learning_rate=0.00003 log_path=/tmp/sft-lr-sweep/lr-0.00003
python -m tinker_cookbook.recipes.sl_loop learning_rate=0.00001 log_path=/tmp/sft-lr-sweep/lr-0.00001
```

--------------------------------

### Create LoRA Training Client

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Initializes a TrainingClient specifically configured for LoRA fine-tuning. It allows customization of parameters such as base model, LoRA rank, and which layers (MLP, attention, unembedding) to train.

```python
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
    rank=16,
    train_mlp=True,
    train_attn=True
)
```

--------------------------------

### Computing Top-k Logprobs for a Sequence

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

This section details how to compute top-k log probabilities for each token in a sequence, which can provide insight into alternative token choices the model might have made.

```APIDOC
## Top-k logprobs

For distillation, it may be especially useful to compute _top-k logprobs_ for each token as well, which can get you a sense for what the model "would have said" after each prefix instead of the actual prompt.

### Method
POST (Implied by `sampling_client.sample`)

### Endpoint
`/sample` (Implied)

### Parameters
#### Request Body
- **prompt** (tinker.ModelInput) - Required - The input sequence.
- **num_samples** (int) - Optional - Number of samples.
- **sampling_params** (tinker.SamplingParams) - Required - Sampling parameters.
- **include_prompt_logprobs** (bool) - Required - Set to `true`.
- **topk_prompt_logprobs** (int) - Required - The number of top-k log probabilities to compute for each token.

### Request Example
```python
# Assuming prompt, tokenizer, and sampling_client are initialized
sample_response = sampling_client.sample(
    prompt=prompt,
    num_samples=1,
    sampling_params=tinker.SamplingParams(max_tokens=1),
    include_prompt_logprobs=True,
    topk_prompt_logprobs=5,
).result()
```

### Response
#### Success Response (200)
- **topk_prompt_logprobs** (list) - For each position in the response, this returns a list of `(token_id, logprob)` pairs for the top-k most likely tokens at that position.

#### Response Example
```json
[
 None,
 [(14924, -1.17005), (755, -2.23255), (2, -2.73255), (791, -3.67005), (16309, -4.29505)],
 [(25, -1.64629), (3137, -2.39629), (11630, -2.89629), (21460, -3.83379), (14881, -4.02129)],
 # ... more entries
]
```
```

--------------------------------

### Tinker Forward Backward Function

Source: https://tinker-docs.thinkingmachines.ai/llms

Documentation for the `forward_backward` and `forward_backward_custom` functions, explaining how to use built-in loss functions or provide custom differentiable losses.

```APIDOC
## POST /tinker/forward_backward

### Description
This endpoint handles the forward and backward pass of the training process. It can use built-in loss functions specified by a string or accept custom differentiable loss functions via `forward_backward_custom`.

### Method
POST

### Endpoint
`/tinker/forward_backward`

### Parameters

#### Request Body
- **data** (array[Datum]) - Required - The training data and loss function inputs.
- **loss_fn** (string) - Optional - Identifier for a built-in loss function (e.g., "importance_sampling", "ppo").

### Request Example (Importance Sampling)
```json
{
  "data": [
    {
      "model_input": [101, 2000, 234, 567, 102],
      "loss_fn_inputs": {
        "target_tokens": {"torch_tensor": [101, 2000, 234, 567, 102]},
        "logprobs": {"torch_tensor": [-0.1, -0.5, -0.2, -0.8, -0.05]},
        "advantages": {"torch_tensor": [0.5, 1.2, 0.8, -0.1, 0.6]}
      }
    }
  ],
  "loss_fn": "importance_sampling"
}
```

### Response
#### Success Response (200)
- **fwd_bwd_result** (object) - Contains output tensors from the loss function calculation.
  - **loss_fn_outputs** (object) - A dictionary mapping string identifiers to output tensors.

#### Response Example
```json
{
  "fwd_bwd_result": {
    "loss_fn_outputs": {
      "policy_loss": {"torch_tensor": [0.0123]},
      "value_loss": {"torch_tensor": [0.0045]}
    }
  }
}
```

### Notes
- When using `forward_backward_custom`, the `loss_fn_inputs` should contain all necessary tensors for the custom loss calculation.
- The `Datum` object expects specific inputs depending on the chosen `loss_fn`.
```

--------------------------------

### Run RLHF Pipeline

Source: https://tinker-docs.thinkingmachines.ai/preferences/rlhf-example

Executes the standard Reinforcement Learning from Human Feedback (RLHF) pipeline using the provided script. This command initiates the entire RLHF process as defined in rlhf_pipeline.py.

```shell
python -m recipes.preference.rlhf.rlhf_pipeline
```

--------------------------------

### Configuring DRO Loss with Tinker Client

Source: https://tinker-docs.thinkingmachines.ai/llms

Illustrates how to set up and use the DRO loss function via the Tinker training client. The `forward_backward_async` method is used here, specifying 'dro' as the loss function and providing the `beta` hyper-parameter within the `loss_fn_config` dictionary.

```python
fwd_bwd_result = await training_client.forward_backward_async(
    data=data,
    loss_fn="dro",
    loss_fn_config={"beta": 0.05}
)
```

--------------------------------

### Create Training Client From State

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Creates a `TrainingClient` by loading model weights from a specified path.

```APIDOC
## Create Training Client From State

### Description
Creates a `TrainingClient` instance by loading only the model weights from a given Tinker path. This is useful for resuming training from a checkpoint where optimizer state is not needed or is being reset.

### Method
`create_training_client_from_state`

### Endpoint
N/A (Client-side method)

### Parameters
#### Path Parameters
- **path** (str) - Required - The Tinker path to the saved weights (e.g., "tinker://run-id/weights/checkpoint-001").

#### Query Parameters
None

#### Request Body
- **user_metadata** (dict[str, str] | None) - Optional - Metadata to attach to the new training run.

### Request Example
```python
# Resume training from a checkpoint (weights only, optimizer resets)
training_client = service_client.create_training_client_from_state(
    "tinker://run-id/weights/checkpoint-001"
)
# Continue training from the loaded state
```

### Response
#### Success Response (200)
- **TrainingClient** (TrainingClient) - A `TrainingClient` instance loaded with the specified weights.
```

--------------------------------

### ServiceClient Initialization

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Initializes the ServiceClient, which is the main entry point for the Tinker API. It supports advanced options for the underlying HTTP client.

```APIDOC
## ServiceClient Initialization

### Description
Initializes the ServiceClient, the primary interface for interacting with the Tinker API. It can accept advanced options for the underlying HTTP client, such as API keys, headers, and connection settings.

### Method
`__init__`

### Parameters
#### Keyword Arguments
- **kwargs** (dict) - Optional - Advanced options passed to the underlying HTTP client, including API keys, headers, and connection settings.

### Request Example
```python
# Near instant initialization
client = ServiceClient()

# Initialization with advanced options (e.g., API key)
# client = ServiceClient(api_key="YOUR_API_KEY")
```

### Response
This method does not return a value directly, but initializes the ServiceClient object.
```

--------------------------------

### Visualize Training Metrics with Pandas and Matplotlib

Source: https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-basic

Load and plot training metrics from a JSONL file using pandas and matplotlib. This allows for visualization of training and test loss over time.

```python
import pandas
import matplotlib.pyplot as plt
df = pandas.read_json("/tmp/tinker-examples/sl_basic/metrics.jsonl", lines=True)
plt.plot(df['train_mean_nll'], label='train_loss')
plt.plot(df['test/nll'].dropna(), label='test_loss')
plt.legend()
plt.show()
```

--------------------------------

### Asynchronous LoRA Training Loop

Source: https://tinker-docs.thinkingmachines.ai/clock_cycles.py

This Python code snippet outlines the core asynchronous training loop for LoRA models. It demonstrates how to create a training client, submit forward-backward and optimizer steps concurrently, and manage pending futures to achieve pipelining. It also includes logic for extracting and printing performance metrics.

```python
training_client = await service_client.create_lora_training_client_async(
        base_model="meta-llama/Llama-3.2-1B"
    )

    num_steps = 5
    last_completion_time = None
    last_clock_cycle = None
    first_clock_cycle = None
    pending_futures = None
    start_time = time.time()

    for step in range(num_steps):
        submit_start = time.time()
        print(f"\n[Step {step}]")

        # Submit the next batch immediately (before waiting for previous results)
        print(f"[Step {step}] Submitting forward_backward...")
        fwdbwd_future = await training_client.forward_backward_async(
            [create_dummy_datum()],
            loss_fn="cross_entropy",
        )

        print(f"[Step {step}] Submitting optim_step...")
        optim_future = await training_client.optim_step_async(
            adam_params=types.AdamParams(learning_rate=1e-4)
        )

        # Now wait for the PREVIOUS batch to complete (if there is one)
        if pending_futures is not None:
            prev_step = step - 1
            print(f"[Step {step}] Waiting for step {prev_step} to complete...")
            prev_fwdbwd, prev_optim = pending_futures

            fwdbwd_result = await prev_fwdbwd
            optim_result = await prev_optim

            # Extract metrics (clock cycle is in fwdbwd_result.metrics)
            fwdbwd_metrics = fwdbwd_result.metrics
            current_clock_cycle = fwdbwd_metrics["clock_cycle:unique"]
            loss = fwdbwd_metrics["loss:sum"]

            if first_clock_cycle is None:
                first_clock_cycle = current_clock_cycle

            completion_time = time.time()
            prev_duration = completion_time - prev_submit_start

            print(f"[Step {step}] ✓ Step {prev_step} completed in {prev_duration:.2f}s")
            print(f"[Step {step}]   Loss: {loss:.4f}")
            print(f"[Step {step}]   Clock cycle: {current_clock_cycle}")

            # Calculate and display clock cycles elapsed and time since last completion
            if last_clock_cycle is not None:
                cycles_elapsed = current_clock_cycle - last_clock_cycle
                time_since_last = completion_time - last_completion_time
                print(
                    f"[Step {step}]   Clock cycles elapsed since last step: {cycles_elapsed}"
                )
                print(
                    f"[Step {step}]   Time since last completion: {time_since_last:.2f}s"
                )
                if cycles_elapsed == 1:
                    print(f"[Step {step}]   ✓ Perfect! Only 1 clock cycle used.")
                else:
                    print(
                        f"[Step {step}]   ⚠️  Used {cycles_elapsed} clock cycles (should be 1)!"
                    )

            last_completion_time = completion_time
            last_clock_cycle = current_clock_cycle

        # Store current futures for next iteration
        pending_futures = (fwdbwd_future, optim_future)
        prev_submit_start = submit_start

    # Wait for the last batch to complete
    if pending_futures is not None:
        print(f"\n[Final] Waiting for step {num_steps - 1} to complete...")
        fwdbwd_result = await pending_futures[0]
        optim_result = await pending_futures[1]

        # Extract metrics (clock cycle is in fwdbwd_result.metrics)
        fwdbwd_metrics = fwdbwd_result.metrics
        current_clock_cycle = fwdbwd_metrics["clock_cycle:unique"]
        loss = fwdbwd_metrics["loss:sum"]

        if first_clock_cycle is None:
            first_clock_cycle = current_clock_cycle

        completion_time = time.time()
        final_duration = completion_time - prev_submit_start

        print(f"[Final] ✓ Step {num_steps - 1} completed in {final_duration:.2f}s")
        print(f"[Final]   Loss: {loss:.4f}")
        print(f"[Final]   Clock cycle: {current_clock_cycle}")

        # Calculate and display clock cycles elapsed and time since last completion
        if last_clock_cycle is not None:
            cycles_elapsed = current_clock_cycle - last_clock_cycle
            time_since_last = completion_time - last_completion_time
            print(f"[Final]   Clock cycles elapsed since last step: {cycles_elapsed}")
            print(f"[Final]   Time since last completion: {time_since_last:.2f}s")
            if cycles_elapsed == 1:
                print("[Final]   ✓ Perfect! Only 1 clock cycle used.")
            else:
                print(f"[Final]   ⚠️  Used {cycles_elapsed} clock cycles (should be 1)!")

    total_time = time.time() - start_time
    total_clock_cycles = current_clock_cycle - first_clock_cycle

    print("\n" + "=" * 70)
    print(f"Pipelined training completed: {num_steps} steps")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total clock cycles used: {total_clock_cycles}")
    print("=" * 70 + "\n")

    return {"total_time": total_time, "total_clock_cycles": total_clock_cycles}


if __name__ == "__main__":
    print("\n" + "=" * 70)

```

--------------------------------

### Using Forward-Backward with Predefined Loss Functions (Python)

Source: https://tinker-docs.thinkingmachines.ai/losses

Demonstrates how to use the `forward_backward` function with predefined loss functions like 'importance_sampling' and 'ppo'. It shows the structure of the input `datum` which includes `model_input` and `loss_fn_inputs` such as target tokens, log probabilities, and advantages.

```python
import tinker
import torch
from tinker import TensorData

# Assuming input_tokens, target_tokens, sampling_logprobs, and advantages are defined elsewhere

# Create training data with required inputs
datum = tinker.Datum(
    model_input=input_tokens,
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
        "logprobs": TensorData.from_torch(torch.tensor(sampling_logprobs)),  # Reference logprobs
        "advantages": TensorData.from_torch(torch.tensor(advantages)),
    }
)

# Option 1: Use importance sampling REINFORCE
fwd_bwd_result = await training_client.forward_backward_async(
    [datum], loss_fn="importance_sampling"
)

# Option 2: Use PPO with clipping
fwd_bwd_result = await training_client.forward_backward_async(
    [datum], loss_fn="ppo"
)
```

--------------------------------

### Configuring CISPO Loss with Tinker Client

Source: https://tinker-docs.thinkingmachines.ai/llms

Shows how to configure and use the CISPO loss function with the Tinker training client. This asynchronous call allows for custom `clip_low_threshold` and `clip_high_threshold` values to be passed in the `loss_fn_config` dictionary for the 'cispo' loss.

```python
fwd_bwd_result = await training_client.forward_backward_async(
    data=data,
    loss_fn="cispo",
    loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
)
```

--------------------------------

### get_info

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Retrieves information about the current model, including configuration and metadata. This is a synchronous operation.

```APIDOC
## get_info

### Description
Get information about the current model.

### Method
GET

### Endpoint
/info

### Response
#### Success Response (200)
- **GetInfoResponse** (types.GetInfoResponse) - An object containing model configuration and metadata.

#### Response Example
```json
{
  "model_data": {
    "model_id": "model-abc",
    "model_name": "llama-2-7b",
    "lora_rank": 8,
    "fine_tuned_at": "2023-01-01T10:00:00Z"
  },
  "training_config": {
    "optimizer": "adamw",
    "learning_rate": 1e-4
  }
}
```
```

--------------------------------

### Create Sampling Client

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Creates a SamplingClient for text generation, configurable via a model path or a base model name.

```APIDOC
## `create_sampling_client`

### Description
Creates a `SamplingClient` used for text generation tasks. You can specify either a path to saved model weights or the name of a base model.

### Method
POST (Assumed, as it creates a client)

### Endpoint
`/sampling/create` (Assumed)

### Parameters
#### Path Parameters
None

#### Query Parameters
- **model_path** (string) - Optional - Path to saved model weights (e.g., "tinker://run-id/weights/checkpoint-001").
- **base_model** (string) - Optional - Name of base model to use (e.g., "Qwen/Qwen3-8B").
- **retry_config** (object) - Optional - Configuration for retrying failed requests.

#### Request Body
None

### Request Example
```python
# Use a base model
sampling_client_base = service_client.create_sampling_client(
    base_model="Qwen/Qwen3-8B"
)

# Or use saved weights
sampling_client_weights = service_client.create_sampling_client(
    model_path="tinker://run-id/weights/checkpoint-001"
)
```

### Response
#### Success Response (200)
- **SamplingClient** (object) - A `SamplingClient` instance configured for text generation.

#### Response Example
```json
{
  "message": "Sampling client created successfully.",
  "client_id": "sampling-abc123"
}
```

### Raises
- **ValueError**: If neither `model_path` nor `base_model` is provided.
```

--------------------------------

### load_state_with_optimizer_async

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Asynchronously loads both model weights and optimizer state from a specified checkpoint path.

```APIDOC
## load_state_with_optimizer_async

### Description
Asynchronous version of `load_state_with_optimizer`. Loads model weights and optimizer state from a checkpoint.

### Method
POST

### Endpoint
/load_state_with_optimizer_async

### Parameters
#### Request Body
- **path** (string) - Required - The Tinker path to the saved weights (e.g., "tinker://run-id/weights/checkpoint-001").

### Request Example
```json
{
  "path": "tinker://run-id/weights/checkpoint-001"
}
```

### Response
#### Success Response (200)
- **APIFuture** (APIFuture[types.LoadWeightsResponse]) - Contains the load response.

#### Response Example
```json
{
  "future_id": "load_state_opt_async_15161"
}
```
```

--------------------------------

### Download Weights via SDK (Python)

Source: https://tinker-docs.thinkingmachines.ai/download-weights

This Python code snippet demonstrates downloading checkpoint weights using the Tinker SDK. It generates a signed URL for the checkpoint archive and then downloads it using `urllib.request`. Replace `<unique_id>` with your specific Training Run ID. The downloaded file will be named `archive.tar`.

```python
import tinker
import urllib.request

sc = tinker.ServiceClient()
rc = sc.create_rest_client()
future = rc.get_checkpoint_archive_url_from_tinker_path("tinker://<unique_id>/sampler_weights/final")
checkpoint_archive_url_response = future.result()

# `checkpoint_archive_url_response.url` is a signed URL that can be downloaded
# until checkpoint_archive_url_response.expires
urllib.request.urlretrieve(checkpoint_archive_url_response.url, "archive.tar")
```

--------------------------------

### Generate Prompt Distillation Data (Python)

Source: https://tinker-docs.thinkingmachines.ai/supervised-learning/prompt-distillation

This script uses a teacher model to generate distilled training data for prompt distillation. It takes an output file path as an argument where the dataset will be saved in JSON Lines format. The generated data is suitable for fine-tuning a student model.

```python
import sys
from tinker_cookbook.recipes.prompt_distillation import create_data

def main():
    # Example usage: saves distilled data to a file
    output_file = "/tmp/tinker-datasets/prompt_distillation_lang.jsonl"
    create_data.main(output_file=output_file)
    print(f"Distilled data saved to {output_file}")

if __name__ == "__main__":
    main()
```

--------------------------------

### Policy Optimization Loss Functions

Source: https://tinker-docs.thinkingmachines.ai/llms

This section covers Policy Optimization methods like PPO and CISPO, detailing their objective functions and how to configure clipping thresholds.

```APIDOC
## Policy Optimization Loss Functions

This section covers Policy Optimization methods like PPO and CISPO, detailing their objective functions and how to configure clipping thresholds.

### Proximal Policy Optimization (PPO)

PPO is a policy gradient method that uses a clipped objective function to prevent large policy updates. The objective is to maximize the expected advantage, but with a clipping mechanism to stay close to the old policy.

**Implementation Details:**
```python
# Compute probability ratio
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
# Apply clipping
clipped_ratio = torch.clamp(prob_ratio, clip_low_threshold, clip_high_threshold)
# Compute both objectives
unclipped_objective = prob_ratio * advantages
clipped_objective = clipped_ratio * advantages
# Take minimum (most conservative)
ppo_objective = torch.min(unclipped_objective, clipped_objective)
# PPO loss is negative of objective
loss = -ppo_objective.sum()
```

**Example with custom clipping thresholds:**
```python
fwd_bwd_result = await training_client.forward_backward_async(
    data=data,
    loss_fn="ppo",
    loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}
)
```

### Clipped Importance Sampling Policy Optimization (CISPO)

CISPO is a policy gradient method that uses a clipped importance ratio to weight the policy gradient. Unlike PPO, it clips the ratio directly before multiplying it with the log probability.

**Mathematical Objective:**
$$ \mathcal{L}_{\text{CISPO}}(\theta) = \mathbb{E}_{x \sim q}\left[\textbf{sg}\left( \text{clip}\left(\frac{p_\theta(x)}{q(x)}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}\right) \right) \cdot \log p_\theta(x) \cdot A(x)\right] $$

**Implementation Details:**
```python
# Compute probability ratio
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
# Apply clipping
clipped_ratio = torch.clamp(prob_ratio, clip_low_threshold, clip_high_threshold)
# Compute CISPO objective (detach the clipped ratio)
cispo_objective = clipped_ratio.detach() * target_logprobs * advantages
# CISPO loss is negative of objective
loss = -cispo_objective.sum()
```

**Example with custom clipping thresholds:**
```python
fwd_bwd_result = await training_client.forward_backward_async(
    data=data,
    loss_fn="cispo",
    loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
)
```

### Additional Notes on Loss Formulation

- The loss formulations are general, allowing user organization of data generation and advantage estimation.
- The functional implementations of REINFORCE and PPO do not use an additional KL term like the original GRPO work, which has been noted to be mathematically inconsistent.
- It is possible to include a KL regularization term as part of the reward.
- For all objectives, token-level losses are summed over the sequence length, unlike some other loss implementations. Different aggregation schemes can be explored in advantage tensor computation.
```

--------------------------------

### Create Sampling Client

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Creates a SamplingClient instance from specified model weights. This client can then be used for inference.

```APIDOC
## POST /sampling_clients

### Description
Creates a SamplingClient from saved weights. This client is used for performing sampling operations.

### Method
POST

### Endpoint
/sampling_clients

### Parameters
#### Query Parameters
- **model_path** (str) - Required - Tinker path to saved weights.
- **retry_config** (RetryConfig | None) - Optional - Configuration for retrying failed requests.

### Request Example
```json
{
  "model_path": "tinker://run-id/weights/checkpoint-001",
  "retry_config": null
}
```

### Response
#### Success Response (200)
- **sampling_client** (SamplingClient) - A SamplingClient configured with the specified weights.

#### Response Example
```json
{
  "sampling_client": { ... SamplingClient object ... }
}
```
```

--------------------------------

### Build Generation Prompt for Alternative Response (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This snippet demonstrates how to build a generation prompt from a conversation history using the `build_generation_prompt` method. It's useful for sampling alternative assistant responses by excluding the last assistant message. It requires the `tinker_cookbook` library and a tokenizer.

```python
from tinker_cookbook import renderers, tokenizer_utils
tokenizer = tokenizer_utils.get_tokenizer('Qwen/Qwen3-30B-A3B')
renderer = renderers.get_renderer('qwen3', tokenizer)
prompt = renderer.build_generation_prompt(messages[:-1])
print(prompt)
print('-'*10)
print(tokenizer.decode(prompt.to_ints()))
```

--------------------------------

### Python OpenAI Client for Tinker Inference

Source: https://tinker-docs.thinkingmachines.ai/compatible-apis/openai

This snippet demonstrates how to configure and use the OpenAI Python client to interact with Tinker's compatible inference endpoint. It requires setting the base URL, providing a Tinker sampler weight path as the model name, and authenticating with a Tinker API key. The arguments mirror those of the OpenAI Completions API.

```python
from os import getenv
from openai import OpenAI

BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
MODEL_PATH = "tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080"

api_key = getenv("TINKER_API_KEY")

client = OpenAI(
    base_url=BASE_URL,
    api_key=api_key,
)

response = client.completions.create(
    model=MODEL_PATH,
    prompt="The capital of France is",
    max_tokens=50,
    temperature=0.7,
    top_p=0.9,
)

print(f"{response.choices[0].text}")
```

--------------------------------

### Policy Optimization Loss Functions

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This section details the implementation of Policy Optimization loss functions, including Proximal Policy Optimization (PPO) and Clipped Importance Sampling Policy Optimization (CISPO). It covers their mathematical formulations, Python code implementations, and how to configure them using `loss_fn_config`.

```APIDOC
## Proximal Policy Optimization (PPO)

### Description
PPO is a policy gradient method that aims to improve stability by clipping the objective function. It ensures that policy updates do not deviate too far from the previous policy.

### Method
This is a conceptual description of the PPO loss calculation, not a direct API endpoint.

### Endpoint
N/A

### Parameters
N/A

### Request Example
```python
fwd_bwd_result = await training_client.forward_backward_async(
    data=data,
    loss_fn="ppo",
    loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}
)
```

### Response
N/A

## Clipped Importance Sampling Policy Optimization: `cispo`

### Description
CISPO is a policy gradient method that clips the importance ratio. Unlike PPO, CISPO clips the ratio itself and uses it to weight the log probability, offering a different approach to policy update stability.

### Method
This is a conceptual description of the CISPO loss calculation, not a direct API endpoint.

### Endpoint
N/A

### Parameters
N/A

### Request Example
```python
fwd_bwd_result = await training_client.forward_backward_async(
    data=data,
    loss_fn="cispo",
    loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
)
```

### Response
N/A
```

--------------------------------

### load_state_with_optimizer

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Loads both model weights and optimizer state from a specified checkpoint path. This is a synchronous operation.

```APIDOC
## load_state_with_optimizer

### Description
Load model weights and optimizer state from a checkpoint.

### Method
POST

### Endpoint
/load_state_with_optimizer

### Parameters
#### Request Body
- **path** (string) - Required - The Tinker path to the saved weights (e.g., "tinker://run-id/weights/checkpoint-001").

### Request Example
```json
{
  "path": "tinker://run-id/weights/checkpoint-001"
}
```

### Response
#### Success Response (200)
- **APIFuture** (APIFuture[types.LoadWeightsResponse]) - Contains the load response.

#### Response Example
```json
{
  "future_id": "load_state_opt_13141"
}
```
```

--------------------------------

### Sample from Model - Python

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

Samples output from a trained model to perform a translation task. It first obtains a sampling client, then defines sampling parameters and prompts the model for translation, printing the generated responses.

```python
# First, create a sampling client. We need to transfer weights
sampling_client = training_client.save_weights_and_get_sampling_client(name='pig-latin-model')
 
# Now, we can sample from the model.
prompt = types.ModelInput.from_ints(tokenizer.encode("English: coffee break\nPig Latin:"))
params = types.SamplingParams(max_tokens=20, temperature=0.0, stop=["\n"]) # Greedy sampling
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
result = future.result()
print("Responses:")
for i, seq in enumerate(result.sequences):
    print(f"{i}: {repr(tokenizer.decode(seq.tokens))}")
```

--------------------------------

### Flexible Loss Functions: `forward_backward_custom`

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Introduces the `forward_backward_custom` and `forward_backward_custom_async` methods for implementing more general and flexible loss functions when the predefined options are insufficient.

```APIDOC
## Flexible loss functions: `forward_backward_custom`

### Description
For use cases outside of the standard PPO, CISPO, and DRO, the `forward_backward_custom` and `forward_backward_custom_async` methods offer greater flexibility in defining and computing loss functions. These methods are generally slower but allow for more complex custom loss formulations.

### Method
This describes the availability of custom forward/backward functions, not a specific API endpoint.

### Endpoint
N/A

### Parameters
N/A

### Request Example
N/A

### Response
N/A
```

--------------------------------

### Run Standard Evaluations with Inspect AI (Shell)

Source: https://tinker-docs.thinkingmachines.ai/evals

This script demonstrates how to run standard cited evaluations using the Inspect AI library with Tinker's internal sampling functionality. It requires specifying the model path, model name, tasks to evaluate, and a renderer name.

```shell
MODEL_PATH=tinker://FIXME # YOUR MODEL PATH HERE
python -m tinker_cookbook.eval.run_inspect_evals \
    model_path=$MODEL_PATH \
    model_name=MODEL_NAME # YOUR MODEL_NAME HERE \
    tasks=inspect_evals/ifeval,inspect_evals/mmlu_0_shot \
    renderer_name=RENDERER_NAME # YOUR RENDERER_NAME HERE
```

--------------------------------

### Create Training Client From Saved Weights

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Restores a TrainingClient from previously saved model weights, without loading the optimizer state. This is useful for resuming training when only the model parameters need to be reloaded.

```python
training_client = service_client.create_training_client_from_state(
    "tinker://run-id/weights/checkpoint-001"
)
```

--------------------------------

### Sample from Model for Translation Task

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This snippet demonstrates how to sample from a trained model to perform a translation task. It first obtains a sampling client by saving the model weights, then defines the input prompt and sampling parameters (greedy sampling with a stop token). The output sequences are decoded back into human-readable text.

```python
# First, create a sampling client. We need to transfer weights
sampling_client = training_client.save_weights_and_get_sampling_client(name='pig-latin-model')

# Now, we can sample from the model.
prompt = types.ModelInput.from_ints(tokenizer.encode("English: coffee break\nPig Latin:"))
params = types.SamplingParams(max_tokens=20, temperature=0.0, stop=["\n"]) # Greedy sampling
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
result = future.result()
print("Responses:")
for i, seq in enumerate(result.sequences):
    print(f"{i}: {repr(tokenizer.decode(seq.tokens))}")
```

--------------------------------

### Load Model Weights and Optimizer State

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Loads both model weights and optimizer state from a checkpoint. This is useful for resuming training from a saved point. It requires the path to the saved checkpoint. An asynchronous version is available.

```python
def load_state_with_optimizer(
        path: str) -> APIFuture[types.LoadWeightsResponse]:
    """
    Load model weights and optimizer state from a checkpoint.
    Args:
      * `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")


    Returns:
      * `APIFuture` containing the load response
    """
    pass

# Example:
# Resume training with optimizer state
# load_future = training_client.load_state_with_optimizer(
#     "tinker://run-id/weights/checkpoint-001"
# )
# await load_future
# # Continue training with restored optimizer momentum
```

```python
async def load_state_with_optimizer_async(
        path: str) -> APIFuture[types.LoadWeightsResponse]:
    """
    Async version of load_state_with_optimizer.
    """
    pass
```

--------------------------------

### Create SamplingClient Asynchronously (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Provides an asynchronous method to create a SamplingClient from model weights. Similar to its synchronous counterpart, it allows for optional retry configurations and is intended for non-blocking inference operations.

```python
async def create_sampling_client_async(
        model_path: str,
        retry_config: RetryConfig | None = None) -> SamplingClient:
    """
    Async version of create_sampling_client.
    """
    pass
```

--------------------------------

### get_info_async

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Asynchronously retrieves information about the current model.

```APIDOC
## get_info_async

### Description
Asynchronous version of `get_info`. Retrieves information about the current model.

### Method
GET

### Endpoint
/info_async

### Response
#### Success Response (200)
- **APIFuture** (APIFuture[types.GetInfoResponse]) - Contains the GetInfoResponse object.

#### Response Example
```json
{
  "future_id": "get_info_async_21222"
}
```
```

--------------------------------

### Demonstrate Pipelining Impact on Training Efficiency (Python)

Source: https://tinker-docs.thinkingmachines.ai/clock_cycles.py

This Python script compares the performance of pipelined and non-pipelined training processes. It measures and prints the total time and clock cycles used for both methods, then calculates and displays the percentage improvement achieved by pipelining. Dependencies include the 'asyncio' module for asynchronous operations.

```python
print("CLOCK CYCLES DEMONSTRATION")
print("=" * 70)
print("\nThis script demonstrates the impact of pipelining on training efficiency.")
print("Watch the 'Clock cycles elapsed' to see the difference!")
print(
    "- Pipelined: Should typically use 1 clock cycle per step when server is lightly loaded"
)
print("- Non-pipelined: May use 2+ clock cycles per step (wasting time)\n")

# Run non-pipelined version
non_pipelined_stats = main_non_pipelined()

# Run pipelined version
pipelined_stats = asyncio.run(main_pipelined())

# Show comparison
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
print(
    "\n┌─────────────────────────────┬──────────────┬──────────────┬─────────────┐"
)
print("│ Metric                      │ Non-Pipelined│ Pipelined    │ Improvement │")
print("├─────────────────────────────┼──────────────┼──────────────┼─────────────┤")

np_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_time = non_pipelined_stats["total_time"]
p_
```

--------------------------------

### Create Sampling Client Async

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Asynchronously creates a SamplingClient instance from specified model weights. This is useful for non-blocking operations in asynchronous applications.

```APIDOC
## POST /sampling_clients/async

### Description
Asynchronously creates a SamplingClient from saved weights.

### Method
POST

### Endpoint
/sampling_clients/async

### Parameters
#### Query Parameters
- **model_path** (str) - Required - Tinker path to saved weights.
- **retry_config** (RetryConfig | None) - Optional - Configuration for retrying failed requests.

### Request Example
```json
{
  "model_path": "tinker://run-id/weights/checkpoint-001",
  "retry_config": null
}
```

### Response
#### Success Response (200)
- **sampling_client** (SamplingClient) - An asynchronously usable SamplingClient configured with the specified weights.

#### Response Example
```json
{
  "sampling_client": { ... SamplingClient object ... }
}
```
```

--------------------------------

### TrainingRunsResponse Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Response containing a list of training runs and pagination cursor.

```APIDOC
## TrainingRunsResponse Objects
```
class TrainingRunsResponse(BaseModel)
```

#### `training_runs`
List of training runs
#### `cursor`
Pagination cursor information
```

--------------------------------

### Download Weights via CLI

Source: https://tinker-docs.thinkingmachines.ai/download-weights

This command downloads checkpoint weights using the Tinker CLI. Ensure the TINKER_CHECKPOINT_PATH environment variable is set. Refer to `tinker checkpoint download --help` for additional options.

```bash
tinker checkpoint download $TINKER_CHECKPOINT_PATH
```

--------------------------------

### Not Found Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

HTTP 404: Indicates the requested resource was not found.

```APIDOC
## NotFoundError Objects

### Description
HTTP 404: The requested resource was not found.

### Class Definition
```python
class NotFoundError(APIStatusError)
```
```

--------------------------------

### Sample and Parse Response with Renderer - Python

Source: https://tinker-docs.thinkingmachines.ai/rendering

Samples a response from a model using a generated prompt and then parses the tokenized output back into a human-readable message. It employs `tinker.ServiceClient` for sampling and `Renderer.parse_response` for message reconstruction. This method ensures the model's output is in a structured message format.

```python
import tinker
from tinker.types import SamplingParams

# Assuming 'renderer' and 'prompt' are defined from the previous step
# tokenizer = tokenizer_utils.get_tokenizer('Qwen/Qwen3-30B-A3B')
# renderer = renderers.get_renderer('qwen3', tokenizer)
# prompt = renderer.build_generation_prompt(messages[:-1])

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model='Qwen/Qwen3-30B-A3B')
stop_sequences = renderer.get_stop_sequences()
print(f"Stop sequences: {stop_sequences}")
sampling_params = SamplingParams(max_tokens=100, temperature=0.5, stop=stop_sequences)

output = sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1).result()
print(f"Sampled tokens: {output.sequences[0].tokens}")

sampled_message, parse_success = renderer.parse_response(output.sequences[0].tokens)
print(f"Sampled message: {sampled_message}")
print(f"Parse success: {parse_success}")
```

--------------------------------

### Perform Training Update - Python

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

Performs a training update using the `training_client`. It executes forward and backward passes, followed by an optimizer step. Futures are used for asynchronous execution, and the loss per token is calculated and printed.

```python
import numpy as np
for _ in range(6):
    fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
    optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
 
    # Wait for the results
    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()
 
    # fwdbwd_result contains the logprobs of all the tokens we put in. Now we can compute the weighted
    # average log loss per token.
    logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
    print(f"Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}")
```

--------------------------------

### Evaluate DPO Models with Inspect Framework

Source: https://tinker-docs.thinkingmachines.ai/preferences/dpo-guide

Evaluate a trained DPO model using the `inspect` evaluation framework. This command-line execution specifies the path to the trained model, the base model name, the evaluation tasks, and the renderer name.

```bash
MODEL_PATH=tinker://YOUR_MODEL_PATH_HERE
python -m tinker_cookbook.eval.run_inspect_evals \
    model_path=$MODEL_PATH \
    model_name=meta-llama/Llama-3.2-1B \
    tasks=inspect_evals/ifeval \
    renderer_name=role_colon
```

--------------------------------

### load_state_async

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Asynchronously loads model weights from a specified checkpoint path.

```APIDOC
## load_state_async

### Description
Asynchronous version of `load_state`. Loads model weights from a saved checkpoint.

### Method
POST

### Endpoint
/load_state_async

### Parameters
#### Request Body
- **path** (string) - Required - The Tinker path to the saved weights (e.g., "tinker://run-id/weights/checkpoint-001").

### Request Example
```json
{
  "path": "tinker://run-id/weights/checkpoint-001"
}
```

### Response
#### Success Response (200)
- **APIFuture** (APIFuture[types.LoadWeightsResponse]) - Contains the load response.

#### Response Example
```json
{
  "future_id": "load_state_async_10112"
}
```
```

--------------------------------

### Compute Prompt Logprobs for a Sequence

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

Computes the log-probabilities for each token in a given prompt sequence. This is useful for understanding the model's confidence at each step of the input. The result includes a list where the first element is None and subsequent elements are the logprobs for each token.

```python
prompt = types.ModelInput.from_ints(tokenizer.encode("How many r's are in the word strawberry?"))
sample_response = sampling_client.sample(
    prompt=prompt,
    num_samples=1,
    sampling_params=tinker.SamplingParams(max_tokens=1),  # Must be at least 1 token, represents prefill step
    include_prompt_logprobs=True,
).result()
 
# example: [None, -9.54505, -1.64629, -8.81116, -3.50217, -8.25927, ...]
print(sample_response.prompt_logprobs)
```

--------------------------------

### Configure PPO Loss with Custom Clipping Thresholds

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Demonstrates how to configure custom clipping thresholds for the PPO loss function when using the `forward_backward_async` method. This allows fine-tuning of the policy update behavior. The `loss_fn_config` dictionary accepts `clip_low_threshold` and `clip_high_threshold`.

```python
fwd_bwd_result = await training_client.forward_backward_async(
    data=data,
    loss_fn="ppo",
    loss_fn_config={\"clip_low_threshold\": 0.9, \"clip_high_threshold\": 1.1}
)
```

--------------------------------

### Permission Denied Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

HTTP 403: Indicates insufficient permissions to access the resource.

```APIDOC
## PermissionDeniedError Objects

### Description
HTTP 403: Insufficient permissions to access the resource.

### Class Definition
```python
class PermissionDeniedError(APIStatusError)
```
```

--------------------------------

### Create Rest Client

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Creates a RestClient for performing REST API operations, including querying model information, checkpoints, and managing visibility.

```APIDOC
## `create_rest_client`

### Description
Creates a `RestClient` instance, which provides access to various REST API endpoints for managing and querying training runs, checkpoints, and sessions.

### Method
POST (Assumed, as it creates a client)

### Endpoint
`/rest/create` (Assumed)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```python
rest_client = service_client.create_rest_client()

# Example Usage:
# List checkpoints for a training run
# checkpoints = rest_client.list_checkpoints("run-id").result()

# Get training run info
# training_run = rest_client.get_training_run("run-id").result()

# Publish a checkpoint
# rest_client.publish_checkpoint_from_tinker_path(
#     "tinker://run-id/weights/checkpoint-001"
# ).result()
```

### Response
#### Success Response (200)
- **RestClient** (object) - A `RestClient` instance configured for REST API operations.

#### Response Example
```json
{
  "message": "REST client created successfully.",
  "api_base_url": "https://api.thinkingmachines.ai/v1"
}
```
```

--------------------------------

### Tinker API: Forward-Backward Pass with Built-in Loss Functions (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms

Demonstrates how to use Tinker's `forward_backward_async` method with built-in loss functions like 'importance_sampling' (REINFORCE) and 'ppo' for training. It shows the structure of the `Datum` object, including `model_input` and `loss_fn_inputs`, and how to extract results from `ForwardBackwardOutput`.

```python
import tinker
import torch
from tinker import TensorData

# Assume input_tokens, target_tokens, sampling_logprobs, and advantages are defined

# Create training data with required inputs
datum = tinker.Datum(
    model_input=input_tokens,
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
        "logprobs": TensorData.from_torch(torch.tensor(sampling_logprobs)),  # Reference logprobs
        "advantages": TensorData.from_torch(torch.tensor(advantages)),
    }
)

# Option 1: Use importance sampling REINFORCE
fwd_bwd_result = await training_client.forward_backward_async(
    [datum], loss_fn="importance_sampling"
)

# Option 2: Use PPO with clipping
fwd_bwd_result = await training_client.forward_backward_async(
    [datum], loss_fn="ppo"
)

# Accessing results (example)
# loss_fn_outputs = fwd_bwd_result.loss_fn_outputs
```

--------------------------------

### List User Checkpoints (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Lists all checkpoints for the current user across all their training runs, sorted by time (newest first). Supports pagination using limit and offset parameters for efficient retrieval of large numbers of checkpoints. Returns a Future containing a CheckpointsListResponse with checkpoints and cursor information.

```python
def list_user_checkpoints(
        limit: int = 100,
        offset: int = 0) -> ConcurrentFuture[types.CheckpointsListResponse]

List all checkpoints for the current user across all their training runs.
This method retrieves checkpoints from all training runs owned by the authenticated user, sorted by time (newest first). It supports pagination for efficiently handling large numbers of checkpoints.
Args:
  * `limit`: Maximum number of checkpoints to return (default 100)
  * `offset`: Offset for pagination (default 0)


Returns:
  * A `Future` containing the `CheckpointsListResponse` with checkpoints and cursor info


Example:
```
future = rest_client.list_user_checkpoints(limit=50)
response = future.result()
print(f"Found {len(response.checkpoints)} checkpoints")
print(f"Total: {response.cursor.total_count if response.cursor else 'Unknown'}")
for checkpoint in response.checkpoints:
    print(f"  {checkpoint.training_run_id}/{checkpoint.checkpoint_id}")
# Get next page if there are more checkpoints
if response.cursor and response.cursor.offset + response.cursor.limit < response.cursor.total_count:
    next_page = rest_client.list_user_checkpoints(limit=50, offset=50)
```
```

--------------------------------

### Perform Training Updates with Forward-Backward and Optim Step

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This code snippet performs multiple training updates on a batch of data. It uses `training_client.forward_backward` to compute gradients and `training_client.optim_step` to apply weight updates using the Adam optimizer. Futures are used for asynchronous execution to improve performance.

```python
import numpy as np

for _ in range(6):
    fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
    optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

    # Wait for the results
    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    # fwdbwd_result contains the logprobs of all the tokens we put in. Now we can compute the weighted
    # average log loss per token.
    logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
    print(f"Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}")
```

--------------------------------

### TinkerTokenCompleter and TinkerMessageCompleter

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Concrete implementations of TokenCompleter and MessageCompleter provided by the Tinker Cookbook, which are wrappers around a `tinker.SamplingClient`.

```APIDOC
## Tinker Completer Implementations

### Description
The Tinker Cookbook provides `TinkerTokenCompleter` and `TinkerMessageCompleter`, which are concrete implementations of the `TokenCompleter` and `MessageCompleter` interfaces, respectively. Both are wrappers around a `tinker.SamplingClient`.

`TinkerTokenCompleter` operates directly on tokens, making it suitable for RL training.

`TinkerMessageCompleter` requires a renderer to be instantiated and operates on messages, useful for sampling in Judge models or multi-agent environments.

### Method
`__init__` (for instantiation)
`__call__` (for generation, as per parent interfaces)

### Endpoint
N/A (Class Definitions)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **TinkerTokenCompleter**: Requires `tinker.SamplingClient` instance.
- **TinkerMessageCompleter**: Requires `tinker.SamplingClient` instance and a `renderer`.

### Request Example
```python
from tinker import SamplingClient
from tinker_cookbook.completers import TinkerTokenCompleter, TinkerMessageCompleter
# Assume 'client' is an initialized tinker.SamplingClient
# Assume 'renderer' is an initialized renderer object

# For TokenCompleter
token_completer = TinkerTokenCompleter(client=client)

# For MessageCompleter
message_completer = TinkerMessageCompleter(client=client, renderer=renderer)
```

### Response
#### Success Response (200)
Depends on the underlying `__call__` method of the parent interfaces (`TokenCompleter` or `MessageCompleter`).

#### Response Example
See examples for `TokenCompleter` and `MessageCompleter` interfaces.
```

--------------------------------

### Build Generation Prompt with Renderer - Python

Source: https://tinker-docs.thinkingmachines.ai/rendering

Converts a conversation history into a prompt suitable for model inference. It utilizes the `Renderer` class and `tokenizer_utils` to process messages and generate a `ModelInput` object. This is crucial for reinforcement learning and deployment scenarios.

```python
from tinker_cookbook import renderers, tokenizer_utils
tokenizer = tokenizer_utils.get_tokenizer('Qwen/Qwen3-30B-A3B')
renderer = renderers.get_renderer('qwen3', tokenizer)

messages = [
    {'role': 'system', 'content': 'Answer concisely; at most one sentence per response'},
    {'role': 'user', 'content': 'What is the longest-lived rodent species?'},
    {'role': 'assistant', 'content': 'The naked mole rat, which can live over 30 years.'},
    {'role': 'user', 'content': 'How do they live so long?'},
    {'role': 'assistant', 'content': 'They evolved multiple protective mechanisms including special hyaluronic acid that prevents cancer, extremely stable proteins, and efficient DNA repair systems that work together to prevent aging.'}
]

prompt = renderer.build_generation_prompt(messages[:-1])
print(prompt)
print('-'*10)
print(tokenizer.decode(prompt.to_ints()))
```

--------------------------------

### Flexible Loss Functions

Source: https://tinker-docs.thinkingmachines.ai/llms

Provides access to more flexible, albeit potentially slower, methods for computing custom loss functions.

```APIDOC
## Flexible loss functions: `forward_backward_custom`

For use cases outside of the above, we've provided the more flexible (but slower) methods `forward_backward_custom` and `forward_backward_custom_async` to compute a more general class of loss functions.
```

--------------------------------

### Compute Prompt Logprobs

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This endpoint allows computing the log probabilities of a given sequence using the prefill step. It returns the log probabilities for each token in the prompt.

```APIDOC
## POST /compute/prompt/logprobs

### Description
Computes the log probabilities for a given sequence, utilizing the prefill step. The output includes '_prompt logprobs_'.

### Method
POST

### Endpoint
/compute/prompt/logprobs

### Parameters
#### Request Body
- **prompt** (types.ModelInput) - Required - The input prompt for which to compute logprobs.
- **num_samples** (int) - Required - The number of samples to generate.
- **sampling_params** (tinker.SamplingParams) - Required - Sampling parameters, including `max_tokens` which must be at least 1 for the prefill step.
- **include_prompt_logprobs** (bool) - Required - Whether to include prompt log probabilities in the response.

### Request Example
```json
{
  "prompt": {"ints": [1, 2, 3]},
  "num_samples": 1,
  "sampling_params": {"max_tokens": 1},
  "include_prompt_logprobs": true
}
```

### Response
#### Success Response (200)
- **prompt_logprobs** (list) - A list where the first element is `None` (for the first token) and subsequent elements are the logprobs of each token in the prompt.

#### Response Example
```json
{
  "prompt_logprobs": [null, -9.54505, -1.64629, -8.81116, -3.50217, -8.25927]
}
```
```

--------------------------------

### Asynchronous Sampler Retrieval (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Offers an asynchronous way to fetch sampler details using a sampler ID. This async method corresponds to `get_sampler` and returns a `GetSamplerResponse`.

```python
async def get_sampler_async(sampler_id: str) -> types.GetSamplerResponse

Async version of get_sampler.
```

--------------------------------

### Sample and Parse Model Response (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This snippet shows how to sample a response from a model and parse it into a message format using Tinker. It involves creating a sampling client, defining sampling parameters including stop sequences obtained from the renderer, sampling the prompt, and then using `parse_response` to convert the sampled tokens back into a structured message. It requires the `tinker` library and `tinker.types`.

```python
import tinker
from tinker.types import SamplingParams
service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model='Qwen/Qwen3-30B-A3B')
stop_sequences = renderer.get_stop_sequences()
print(f"Stop sequences: {stop_sequences}")
sampling_params = SamplingParams(max_tokens=100, temperature=0.5, stop=stop_sequences)
output = sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1).result()
print(f"Sampled tokens: {output.sequences[0].tokens}")
sampled_message, parse_success = renderer.parse_response(output.sequences[0].tokens)
print(f"Sampled message: {sampled_message}")
print(f"Parse success: {parse_success}")
```

--------------------------------

### Compute Forward Pass with Training Data

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Illustrates how to perform a forward pass on training data using a specified loss function and optional configuration. This method computes the loss without calculating gradients. It returns an APIFuture containing the outputs and loss.

```python
data = [types.Datum(
    model_input=types.ModelInput.from_ints(tokenizer.encode("Hello")),
    loss_fn_inputs={"target_tokens": types.ModelInput.from_ints(tokenizer.encode("world"))}
)]
future = training_client.forward(data, "cross_entropy")
result = await future
print(f"Loss: {result.loss}")
```

--------------------------------

### Non-Pipelined Training Loop

Source: https://tinker-docs.thinkingmachines.ai/clock_cycles.py

Implements a non-pipelined training loop where each batch is submitted, waited for, and then the next is submitted. This can lead to missed clock cycles due to gaps between batch completions and submissions. Requires 'time', 'tinker', and 'numpy'.

```python
import time
import tinker
from tinker import types
import numpy as np

def main_non_pipelined():
    """
    Non-pipelined training loop: Submit a batch, wait for it to complete, then submit the next.

    This approach may waste clock cycles because there's a gap between when one batch finishes
    and when the next batch is submitted.
    """
    print("\n" + "=" * 70)
    print("RUNNING NON-PIPELINED TRAINING")
    print("=" * 70)
    print(
        "In this version, we wait for each batch to complete before submitting the next."
    )
    print("This can lead to missing clock cycles during the gap between batches.\n")

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model="meta-llama/Llama-3.2-1B"
    )

    num_steps = 5
    last_completion_time = None
    last_clock_cycle = None
    first_clock_cycle = None
    start_time = time.time()

    for step in range(num_steps):
        step_start = time.time()
        print(f"\n[Step {step}]")

        # Submit forward-backward pass
        print(f"[Step {step}] Submitting forward_backward...")
        fwdbwd_future = training_client.forward_backward(
            [create_dummy_datum()],
            loss_fn="cross_entropy",
        )

        # Submit optimizer step
        print(f"[Step {step}] Submitting optim_step...")
        optim_future = training_client.optim_step(
            adam_params=types.AdamParams(learning_rate=1e-4)
        )

        # Wait for results
        print(f"[Step {step}] Waiting for forward_backward to complete...")
        fwdbwd_result = fwdbwd_future.result()

        print(f"[Step {step}] Waiting for optim_step to complete...")
        optim_result = optim_future.result()

        # Extract metrics (clock cycle is in fwdbwd_result.metrics)
        fwdbwd_metrics = fwdbwd_result.metrics
        current_clock_cycle = fwdbwd_metrics["clock_cycle:unique"]
        loss = fwdbwd_metrics["loss:sum"]

        if first_clock_cycle is None:
            first_clock_cycle = current_clock_cycle

        completion_time = time.time()
        step_duration = completion_time - step_start

        print(f"[Step {step}] ✓ Step completed in {step_duration:.2f}s")
        print(f"[Step {step}]   Loss: {loss:.4f}")
        print(f"[Step {step}]   Clock cycle: {current_clock_cycle}")

        # Calculate and display clock cycles elapsed and time since last completion
        if last_clock_cycle is not None:
            cycles_elapsed = current_clock_cycle - last_clock_cycle
            time_since_last = completion_time - last_completion_time
            print(
                f"[Step {step}]   Clock cycles elapsed since last step: {cycles_elapsed}"
            )
            print(f"[Step {step}]   Time since last completion: {time_since_last:.2f}s")
            if cycles_elapsed > 1:
                print(
                    f"[Step {step}]   ⚠️  MISSED {cycles_elapsed - 1} CLOCK CYCLE(S) (expected - non-pipelined wastes cycles)"
                )

        last_completion_time = completion_time
        last_clock_cycle = current_clock_cycle

    total_time = time.time() - start_time
    total_clock_cycles = current_clock_cycle - first_clock_cycle

    print("\n" + "=" * 70)
    print(f"Non-pipelined training completed: {num_steps} steps")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total clock cycles used: {total_clock_cycles}")
    print("=" * 70 + "\n")

    return {"total_time": total_time, "total_clock_cycles": total_clock_cycles}
```

--------------------------------

### Python BaseModel for Server Capabilities

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the structure for GetServerCapabilitiesResponse, which includes a list of supported models. It leverages Pydantic's BaseModel for data validation and serialization.

```python
class GetServerCapabilitiesResponse(BaseModel):
    supported_models: List[SupportedModel]
```

--------------------------------

### Pipelined Training Loop

Source: https://tinker-docs.thinkingmachines.ai/clock_cycles.py

Implements a pipelined training loop where the next batch is submitted before the current one finishes. This maximizes clock cycle utilization by ensuring a continuous queue of requests. Requires 'asyncio', 'tinker', and 'numpy'.

```python
import asyncio
import tinker
from tinker import types
import numpy as np

async def main_pipelined():
    """
    Pipelined training loop: Submit the next batch before waiting for the current batch.

    This approach maximizes clock cycle utilization by ensuring there's always a request
    queued when a new cycle starts.
    """
    print("\n" + "=" * 70)
    print("RUNNING PIPELINED TRAINING")
    print("=" * 70)
    print(
        "In this version, we submit the next batch before waiting for the current one."
    )
    print("This ensures we don't miss any clock cycles.\n")

    service_client = tinker.ServiceClient()
```

--------------------------------

### Load Model Weights

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Loads model weights from a specified checkpoint path. This method only loads weights, not the optimizer state. An asynchronous version is also provided. Use `load_state_with_optimizer` to restore optimizer state.

```python
def load_state(path: str) -> APIFuture[types.LoadWeightsResponse]:
    """
    Load model weights from a saved checkpoint.
    This loads only the model weights, not optimizer state (e.g., Adam momentum). To also restore optimizer state, use load_state_with_optimizer.
    Args:
      * `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")


    Returns:
      * `APIFuture` containing the load response
    """
    pass

# Example:
# Load checkpoint to continue training (weights only, optimizer resets)
# load_future = training_client.load_state("tinker://run-id/weights/checkpoint-001")
# await load_future
# # Continue training from loaded state
```

```python
async def load_state_async(path: str) -> APIFuture[types.LoadWeightsResponse]:
    """
    Async version of load_state.
    """
    pass
```

--------------------------------

### Create RL Environment Groups - Python

Source: https://tinker-docs.thinkingmachines.ai/rl/rl-envs

Defines a builder for creating groups of RL environments. This is essential for multi-agent training or scenarios requiring multiple samples. The `make_envs` method should return a sequence of `Env` instances.

```python
class EnvGroupBuilder:
    """
    Builds a group of environments.
    """

    async def make_envs(self) -> Sequence[Env]:
        raise NotImplementedError
```

--------------------------------

### Asynchronous Checkpoint Listing (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Offers an asynchronous way to list checkpoints for the current user. It accepts `limit` and `offset` parameters for pagination and returns a `CheckpointsListResponse` asynchronously.

```python
async def list_user_checkpoints_async(limit: int = 100,
                                      offset: int = 0
                                      ) -> types.CheckpointsListResponse

Async version of list_user_checkpoints.
```

--------------------------------

### Compute Forward/Backward with Custom Loss Function

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Demonstrates the use of `forward_backward_custom` to compute forward and backward passes with a user-defined custom loss function. This allows for flexible loss computations beyond standard options. The custom function receives log probabilities and returns loss and metrics.

```python
def custom_loss(data, logprobs_list):
    # Custom loss computation
    loss = torch.mean(torch.stack([torch.mean(lp) for lp in logprobs_list]))
    metrics = {"custom_metric": loss.item()}
    return loss, metrics

future = training_client.forward_backward_custom(data, custom_loss)
result = future.result()
print(f"Custom loss: {result.loss}")
print(f"Metrics: {result.metrics}")
```

--------------------------------

### Cursor Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Pagination cursor information.

```APIDOC
## Cursor Objects
```
class Cursor(BaseModel)
```

#### `offset`
The offset used for pagination
#### `limit`
The maximum number of items requested
#### `total_count`
The total number of items available
```

--------------------------------

### List Sessions with Pagination (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Lists available sessions with support for pagination. Allows specifying the maximum number of sessions to return (limit) and an offset for retrieving subsequent pages. Returns a Future containing a ListSessionsResponse with a list of session IDs.

```python
def list_sessions(
        limit: int = 20,
        offset: int = 0) -> ConcurrentFuture[types.ListSessionsResponse]

List sessions with pagination support.
Args:
  * `limit`: Maximum number of sessions to return (default 20)
  * `offset`: Offset for pagination (default 0)


Returns:
  * A `Future` containing the `ListSessionsResponse` with list of session IDs


Example:
```
future = rest_client.list_sessions(limit=50)
response = future.result()
print(f"Found {len(response.sessions)} sessions")
# Get next page
next_page = rest_client.list_sessions(limit=50, offset=50)
```
```

--------------------------------

### Publish Checkpoint from Tinker Path Async (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

An asynchronous version of `publish_checkpoint_from_tinker_path`. This function enables non-blocking publishing of checkpoints via their Tinker paths, returning `None` upon success. It's suitable for asynchronous environments where waiting for publication is not desired. Dependencies: None explicitly shown, relies on underlying async capabilities.

```python
async def publish_checkpoint_from_tinker_path_async(tinker_path: str) -> None:
    # ... implementation details ...
    pass
```

--------------------------------

### SampleRequest Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the parameters for requesting samples from a model.

```APIDOC
## SampleRequest Objects
```
class SampleRequest(StrictBase)
```

#### `num_samples`
Number of samples to generate
#### `base_model`
Optional base model name to sample from. Is inferred from model_path, if provided. If sampling against a base model, this is required.
#### `model_path`
Optional tinker:// path to your model weights or LoRA weights. If not provided, samples against the base model.
#### `sampling_session_id`
Optional sampling session ID to use instead of model_path/base_model. If provided along with seq_id, the model configuration will be loaded from the sampling session. This is useful for multi-turn conversations.
#### `seq_id`
Sequence ID within the sampling session. Required when sampling_session_id is provided. Used to generate deterministic request IDs for the sampling request.
#### `prompt_logprobs`
If set to `true`, computes and returns logprobs on the prompt tokens. Defaults to false.
#### `topk_prompt_logprobs`
If set to a positive integer, returns the top-k logprobs for each prompt token.
```

--------------------------------

### Use CustomEvaluator with Toy Dataset and Grader in Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This Python code demonstrates how to instantiate and use the 'CustomEvaluator'. It defines a simple QA dataset and a grading function. It then creates an instance of 'CustomEvaluator', sets up a 'ServiceClient' and 'SamplingClient', and runs an asynchronous evaluation, printing the resulting metrics. It requires the 'asyncio' library for running the main asynchronous function.

```python
import asyncio

QA_DATASET = [
    {"input": "What is the capital of France?", "output": "Paris"},
    {"input": "What is the capital of Germany?", "output": "Berlin"},
    {"input": "What is the capital of Italy?", "output": "Rome"},
]

def grader_fn(response: str, target: str) -> bool:
    return target.lower() in response.lower()

evaluator = CustomEvaluator(
    dataset=QA_DATASET,
    grader_fn=grader_fn,
    renderer_name="llama3",
    model_name="meta-llama/Llama-3.1-8B-Instruct",

)

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model="meta-llama/Llama-3.1-8B-Instruct")

async def main():
    result = await evaluator(sampling_client)
    print(result)

asyncio.run(main())

```

--------------------------------

### Create Multimodal Message with Image and Text (Python)

Source: https://tinker-docs.thinkingmachines.ai/rendering

This code illustrates how to construct a multimodal message for vision-language models. It includes both an image part, specified by a URL, and a text part, allowing the model to process both visual and textual information within a single message.

```Python
from tinker_cookbook.renderers import Message, TextPart, ImagePart

# Text-only message (standard)
text_message = Message(role='user', content='What is this?')

# Multimodal message with image
multimodal_message = Message(
    role='user',
    content=[
        ImagePart(type='image', image='https://thinkingmachines.ai/blog/on-policy-distillation/images/chess.png'),
        TextPart(type='text', text='What is in this image?'),
    ]
)
```

--------------------------------

### Compute Top-K Prompt Logprobs

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This endpoint computes the top-k log probabilities for each token in a given sequence. This is useful for understanding alternative token predictions.

```APIDOC
## POST /compute/topk/prompt/logprobs

### Description
Computes the top-k log probabilities for each token in a sequence. This provides insight into what the model "would have said" after each prefix.

### Method
POST

### Endpoint
/compute/topk/prompt/logprobs

### Parameters
#### Request Body
- **prompt** (types.ModelInput) - Required - The input prompt for which to compute top-k logprobs.
- **num_samples** (int) - Required - The number of samples to generate.
- **sampling_params** (tinker.SamplingParams) - Required - Sampling parameters, including `max_tokens` which must be at least 1.
- **include_prompt_logprobs** (bool) - Required - Whether to include prompt log probabilities.
- **topk_prompt_logprobs** (int) - Required - The number of top-k log probabilities to compute for each token.

### Request Example
```json
{
  "prompt": {"ints": [1, 2, 3]},
  "num_samples": 1,
  "sampling_params": {"max_tokens": 1},
  "include_prompt_logprobs": true,
  "topk_prompt_logprobs": 5
}
```

### Response
#### Success Response (200)
- **topk_prompt_logprobs** (list) - For each position, a list of `(token_id, logprob)` pairs for the top-k most likely tokens at that position.

#### Response Example
```json
{
  "topk_prompt_logprobs": [
    null,
    [[14924, -1.17005], [755, -2.23255], [2, -2.73255], [791, -3.67005], [16309, -4.29505]],
    [[25, -1.64629], [3137, -2.39629], [11630, -2.89629], [21460, -3.83379], [14881, -4.02129]]
  ]
}
```
```

--------------------------------

### Create SamplingClient from Model Path (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Creates a SamplingClient instance from specified model weights stored at a Tinker path. It accepts an optional retry configuration for handling failed requests. This client is used for performing inference.

```python
def create_sampling_client(
        model_path: str,
        retry_config: RetryConfig | None = None) -> SamplingClient:
    """
    Create a SamplingClient from saved weights.
    Args:
      * `model_path`: Tinker path to saved weights
      * `retry_config`: Optional configuration for retrying failed requests
    Returns:
      * `SamplingClient` configured with the specified weights
    """
    pass

sampling_client = training_client.create_sampling_client(
    "tinker://run-id/weights/checkpoint-001"
)
# Use sampling_client for inference
```

--------------------------------

### PPO Clipping Loss for Reinforcement Learning (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms

Implements the Proximal Policy Optimization (PPO) clipping objective, which limits policy updates to prevent large changes. It computes a probability ratio and applies clipping to balance clipped and unclipped objectives, using target log probabilities, sampling log probabilities, and advantages. The implementation is token-wise.

```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
```

--------------------------------

### Tinker Core Functionality

Source: https://tinker-docs.thinkingmachines.ai/llms

This section details the core functions provided by the Tinker API for managing the training process, including forward/backward passes, optimization steps, and model sampling.

```APIDOC
## Tinker API Core Functions

### Description
The Tinker API provides several key functions to facilitate LLM fine-tuning. These functions abstract away the complexities of distributed training, allowing users to focus on their models and data.

### Functions

- `forward_backward`: Feed data and a loss function to compute and accumulate gradients.
- `optim_step`: Update model weights using the accumulated gradients.
- `sample`: Generate outputs from the trained model.
- Other functions for saving and loading model and optimizer states.
```

--------------------------------

### GetInfoResponse Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Response containing information about a training client's model.

```APIDOC
## GetInfoResponse Objects
```
class GetInfoResponse(BaseModel)
```

#### `type`
Response type identifier.
#### `model_data`
Detailed metadata about the model.
#### `model_id`
Unique identifier for the model.
#### `is_lora`
Whether this is a LoRA fine-tuned model.
#### `lora_rank`
The rank of the LoRA adaptation, if applicable.
#### `model_name`
The name of the model.
```

--------------------------------

### Use CustomEvaluator with Toy Dataset and Async Execution

Source: https://tinker-docs.thinkingmachines.ai/evals

This Python code demonstrates how to instantiate and use the `CustomEvaluator`. It defines a simple QA dataset and a `grader_fn` to check if the target output is present in the model's response. It then creates an evaluator instance, sets up a `ServiceClient` and `SamplingClient`, and runs the evaluation within an async `main` function.

```python
import asyncio

QA_DATASET = [
    {"input": "What is the capital of France?", "output": "Paris"},
    {"input": "What is the capital of Germany?", "output": "Berlin"},
    {"input": "What is the capital of Italy?", "output": "Rome"},
]

def grader_fn(response: str, target: str) -> bool:
    return target.lower() in response.lower()


evaluator = CustomEvaluator(
    dataset=QA_DATASET,
    grader_fn=grader_fn,
    renderer_name="llama3",
    model_name="meta-llama/Llama-3.1-8B-Instruct",

)

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model="meta-llama/Llama-3.1-8B-Instruct")

async def main():
    result = await evaluator(sampling_client)
    print(result)

asyncio.run(main())

```

--------------------------------

### Publish Model Checkpoint using Tinker CLI

Source: https://tinker-docs.thinkingmachines.ai/publish-weights

Publishes a previously saved model checkpoint to the Tinker community. The `$TINKER_CHECKPOINT_PATH` variable must point to a valid checkpoint, typically in the format 'tinker://<run_id>/weights/<checkpoint_id>'. This allows others to load and use your model.

```bash
tinker checkpoint publish $TINKER_CHECKPOINT_PATH
```

--------------------------------

### TrainingRun Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Represents a single training run, including model details and status.

```APIDOC
## TrainingRun Objects
```
class TrainingRun(BaseModel)
```

#### `training_run_id`
The unique identifier for the training run
#### `base_model`
The base model name this model is derived from
#### `model_owner`
The owner/creator of this model
#### `is_lora`
Whether this model uses LoRA (Low-Rank Adaptation)
#### `corrupted`
Whether the model is in a corrupted state
#### `lora_rank`
The LoRA rank if this is a LoRA model, null otherwise
#### `last_request_time`
The timestamp of the last request made to this model
#### `last_checkpoint`
The most recent training checkpoint, if available
#### `last_sampler_checkpoint`
The most recent sampler checkpoint, if available
#### `user_metadata`
Optional metadata about this training run, set by the end-user
```

--------------------------------

### Compute Top-K Logprobs for Distillation

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Computes the top-k log probabilities for each token in a sequence. This is useful for distillation tasks, providing insight into alternative token predictions after each prefix. The output is a list of (token_id, logprob) pairs for each position.

```python
sample_response = sampling_client.sample(
    prompt=prompt,
    num_samples=1,
    sampling_params=tinker.SamplingParams(max_tokens=1),
    include_prompt_logprobs=True,
    topk_prompt_logprobs=5,
).result()

# example:
# [
#  None,
#  [(14924, -1.17005), (755, -2.23255), (2, -2.73255), (791, -3.67005), (16309, -4.29505)],
#  [(25, -1.64629), (3137, -2.39629), (11630, -2.89629), (21460, -3.83379), (14881, -4.02129)],
#  [(41, -3.49866), (42, -3.49866), (49, -4.24866), (38, -4.37366), (54, -4.49866)],
#  [(311, -1.00217), (656, -2.25217), (2057, -2.75217), (649, -3.25217), (10470, -3.37717)],
#  ...]
sample_response.topk_prompt_logprobs
```

--------------------------------

### Authentication Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

HTTP 401: Indicates missing or invalid authentication credentials.

```APIDOC
## AuthenticationError Objects

### Description
HTTP 401: Authentication credentials are missing or invalid.

### Class Definition
```python
class AuthenticationError(APIStatusError)
```
```

--------------------------------

### Proximal Policy Optimization (PPO) Loss Implementation (Python)

Source: https://tinker-docs.thinkingmachines.ai/losses

Implements the PPO clipped objective. It calculates the probability ratio, applies clipping to this ratio, and then computes both the unclipped and clipped objectives. The minimum of these two objectives is taken to form the PPO objective, and the loss is the negative sum of this objective. This method helps prevent large policy updates.

```python
# Compute probability ratio
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
# Apply clipping
clipped_ratio = torch.clamp(prob_ratio, clip_low_threshold, clip_high_threshold)
# Compute both objectives
unclipped_objective = prob_ratio * advantages
clipped_objective = clipped_ratio * advantages
# Take minimum (most conservative)
ppo_objective = torch.min(unclipped_objective, clipped_objective)
# PPO loss is negative of objective
loss = -ppo_objective.sum()
```

--------------------------------

### optim_step_async

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Asynchronously updates model parameters using the Adam optimizer.

```APIDOC
## optim_step_async

### Description
Asynchronous version of `optim_step`. Updates model parameters using Adam optimizer.

### Method
POST

### Endpoint
/optim_step_async

### Parameters
#### Request Body
- **adam_params** (types.AdamParams) - Required - Adam optimizer parameters including learning_rate, betas, eps, weight_decay.

### Request Example
```json
{
  "adam_params": {
    "learning_rate": 0.0001,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01
  }
}
```

### Response
#### Success Response (200)
- **APIFuture** (APIFuture[types.OptimStepResponse]) - Contains the optimizer step response.

#### Response Example
```json
{
  "future_id": "optim_step_async_67890"
}
```
```

--------------------------------

### CreateSamplingSessionResponse Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Response containing the ID for a newly created sampling session.

```APIDOC
## CreateSamplingSessionResponse Objects
```
class CreateSamplingSessionResponse(BaseModel)
```

#### `sampling_session_id`
The generated sampling session ID
```

--------------------------------

### Set OpenAI Client Base URL with Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This Python code demonstrates how to configure an OpenAI client to use a Tinker-provided endpoint for OpenAI-compatible inference. It requires setting the base URL to the specified Tinker endpoint and using a Tinker sampler weight path as the model name.

```python
client = OpenAI(
    base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
    api_key="YOUR_TINKER_API_KEY"
)

response = client.chat.completions.create(
    model="tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080",
    messages=[
        {"role": "user", "content": "Say this is a test!"}
    ]
)
```

--------------------------------

### Create REST Client for API Operations (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient

Instantiates a RestClient for interacting with Tinker AI's REST API. This client is used for operations such as listing checkpoints, retrieving training run information, and managing checkpoint visibility.

```python
def create_rest_client() -> RestClient:
    # Implementation details here
```

```python
rest_client = service_client.create_rest_client()

# List checkpoints for a training run
checkpoints = rest_client.list_checkpoints("run-id").result()

# Get training run info
training_run = rest_client.get_training_run("run-id").result()

# Publish a checkpoint
rest_client.publish_checkpoint_from_tinker_path(
    "tinker://run-id/weights/checkpoint-001"
).result()
```

--------------------------------

### Save and Load Training State using Python

Source: https://tinker-docs.thinkingmachines.ai/save-load

Saves both the model weights and the optimizer state, allowing for a full resumption of training. This method is useful for multi-step training, hyperparameter adjustments, or recovery from interruptions. It requires a `name` for saving and a `path` for loading.

```python
# Save a checkpoint that you can resume from
resume_path = training_client.save_state(name="0010").result().path
 
# Load that checkpoint
training_client.load_state(resume_path)
```

--------------------------------

### List Sessions API

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Provides a paginated list of available sessions.

```APIDOC
## GET /list_sessions

### Description
List sessions with pagination support.

### Method
GET

### Endpoint
/list_sessions

### Parameters
#### Path Parameters
None

#### Query Parameters
- **limit** (int) - Optional - Maximum number of sessions to return (default 20)
- **offset** (int) - Optional - Offset for pagination (default 0)

### Request Example
(No request body for GET requests)

### Response
#### Success Response (200)
- **sessions** (array) - A list of session objects.

#### Response Example
```json
{
  "sessions": [
    {
      "session_id": "session-id-1",
      "created_at": "2023-12-18T09:00:00Z"
    },
    {
      "session_id": "session-id-2",
      "created_at": "2023-12-17T14:00:00Z"
    }
  ]
}
```
```

--------------------------------

### Compute Top-K Logprobs for a Sequence

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

Computes the top-k most likely log-probabilities for each token in a given prompt sequence. This is particularly useful for distillation tasks, providing insight into alternative predictions the model might have made. The output is a list of lists, where each inner list contains (token_id, logprob) pairs for the top-k tokens.

```python
sample_response = sampling_client.sample(
    prompt=prompt,
    num_samples=1,
    sampling_params=tinker.SamplingParams(max_tokens=1),
    include_prompt_logprobs=True,
    topk_prompt_logprobs=5,
).result()
 
# example:
# [
#  None,
#  [(14924, -1.17005), (755, -2.23255), (2, -2.73255), (791, -3.67005), (16309, -4.29505)],
#  [(25, -1.64629), (3137, -2.39629), (11630, -2.89629), (21460, -3.83379), (14881, -4.02129)],
#  [(41, -3.49866), (42, -3.49866), (49, -4.24866), (38, -4.37366), (54, -4.49866)],
#  [(311, -1.00217), (656, -2.25217), (2057, -2.75217), (649, -3.25217), (10470, -3.37717)],
#  ...]
sample_response.topk_prompt_logprobs
```

--------------------------------

### Configure CISPO Loss with Custom Clipping Thresholds

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Shows how to specify custom clipping thresholds for the CISPO loss function using the `forward_backward_async` method. This allows for the adjustment of the clipping range for the importance ratio, affecting the policy update. The configuration is passed via the `loss_fn_config` parameter.

```python
fwd_bwd_result = await training_client.forward_backward_async(
    data=data,
    loss_fn="cispo",
    loss_fn_config={\"clip_low_threshold\": 0.8, \"clip_high_threshold\": 1.2}
)
```

--------------------------------

### Determine Optimal Learning Rate from Sweep Data (Python)

Source: https://tinker-docs.thinkingmachines.ai/supervised-learning/sweep-case-study

Calculates and prints the optimal learning rate that corresponds to the minimum final loss from the collected sweep data. This Python snippet identifies the index of the minimum loss and retrieves the associated learning rate, formatting it to two decimal places in scientific notation.

```python
optimal_lr = df["learning_rate"][df["final_loss"].idxmin()]
print(f"The optimal LR is {optimal_lr:.2e}")
```

--------------------------------

### SampleResponse Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the structure of the response when requesting samples.

```APIDOC
## SampleResponse Objects
```
class SampleResponse(BaseModel)
```

#### `prompt_logprobs`
If prompt_logprobs was set to true in the request, logprobs are computed for every token in the prompt. The `prompt_logprobs` response contains a float32 value for every token in the prompt.
#### `topk_prompt_logprobs`
If topk_prompt_logprobs was set to a positive integer k in the request, the top-k logprobs are computed for every token in the prompt. The `topk_prompt_logprobs` response contains, for every token in the prompt, a list of up to k (token_id, logprob) tuples.
```

--------------------------------

### Load Public Weights using Tinker Python Client

Source: https://tinker-docs.thinkingmachines.ai/publish-weights

Loads public model weights programmatically using the Tinker Python client. This process is identical to loading non-public weights. It involves creating a training client instance initialized with the checkpoint path, enabling further training or sampling.

```python
ckpt_path = ...
training_client = service_client.create_training_client_from_state(ckpt_path)
```

--------------------------------

### get_tokenizer

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Retrieves the tokenizer compatible with the current model.

```APIDOC
## get_tokenizer

### Description
Get the tokenizer for the current model.

### Method
GET

### Endpoint
/tokenizer

### Response
#### Success Response (200)
- **PreTrainedTokenizer** (PreTrainedTokenizer) - A tokenizer object compatible with the model.

#### Response Example
(Returns a tokenizer object, not a JSON response. Example usage):
```python
tokenizer = training_client.get_tokenizer()
encoded = tokenizer.encode("Hello world")
decoded = tokenizer.decode(encoded)
print(decoded) # Output: Hello world
```
```

--------------------------------

### Define GetInfoResponse for Client Model Information

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the GetInfoResponse class, containing information about a training client's model. It includes the response type, model data, model ID, LoRA status, LoRA rank, and model name.

```python
class GetInfoResponse(BaseModel)

Response containing information about a training client's model.
#### `type`
Response type identifier.
#### `model_data`
Detailed metadata about the model.
#### `model_id`
Unique identifier for the model.
#### `is_lora`
Whether this is a LoRA fine-tuned model.
#### `lora_rank`
The rank of the LoRA adaptation, if applicable.
#### `model_name`
The name of the model.
```

--------------------------------

### save_state_async

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Asynchronously saves the current model weights to persistent storage.

```APIDOC
## save_state_async

### Description
Asynchronous version of `save_state`. Saves model weights to persistent storage.

### Method
POST

### Endpoint
/save_state_async

### Parameters
#### Request Body
- **name** (string) - Required - A name for the saved checkpoint.

### Request Example
```json
{
  "name": "checkpoint-001"
}
```

### Response
#### Success Response (200)
- **APIFuture** (APIFuture[types.SaveWeightsResponse]) - Contains the save response, including the checkpoint path.

#### Response Example
```json
{
  "future_id": "save_state_async_44556",
  "path": "tinker://runs/run-id/weights/checkpoint-001"
}
```
```

--------------------------------

### Optimize Model Parameters with Adam

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Updates model parameters using the Adam optimizer. It requires Adam optimizer parameters such as learning rate, betas, epsilon, and weight decay. Returns an APIFuture containing the optimizer step response.

```python
def optim_step(
        adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]:
    """
    Update model parameters using Adam optimizer.
    Args:
      * `adam_params`: Adam optimizer parameters (learning_rate, betas, eps, weight_decay)


    Returns:
      * `APIFuture` containing optimizer step response
    """
    pass

# Example:
# First compute gradients
# fwdbwd_future = training_client.forward_backward(data, "cross_entropy")
# 
# # Then update parameters
# optim_future = training_client.optim_step(
#     types.AdamParams(
#         learning_rate=1e-4,
#         weight_decay=0.01
#     )
# )
# 
# # Wait for both to complete
# fwdbwd_result = await fwdbwd_future
# optim_result = await optim_future
```

```python
async def optim_step_async(
        adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]:
    """
    Async version of optim_step.
    """
    pass
```

--------------------------------

### Saving and Loading Model State

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Methods for saving and loading model weights and optimizer states for sampling and resuming training.

```APIDOC
## Saving and Loading Model State

### Description
Provides methods to save and load model weights and optimizer states. This is crucial for creating checkpoints for model sampling and for resuming training sessions.

### Methods

- `save_weights_for_sampler(name: str)`: Saves a copy of the model weights suitable for sampling.
- `save_state(name: str)`: Saves both the model weights and the optimizer state, allowing for a full resumption of training.
- `load_state(path: str)`: Loads the model weights and optimizer state from a specified path, enabling training resumption.

### Parameters

#### Path Parameters
None

#### Query Parameters
None

#### Request Body

- `name` (str) - Required - A string identifier for the checkpoint (e.g., "0000", "step_1000").
- `path` (str) - Required (for `load_state`) - The fully-qualified path to the checkpoint to be loaded.

### Request Example

**Saving for sampling:**
```python
# Setup
import tinker
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(base_model="meta-llama/Llama-3.2-1B", rank=32)

# Save a checkpoint for sampling
sampling_path_info = training_client.save_weights_for_sampler(name="0000").result()
sampling_path = sampling_path_info.path

# Create a sampling client
sampling_client = service_client.create_sampling_client(model_path=sampling_path)
```

**Shortcut for saving and getting sampling client:**
```python
sampling_client = training_client.save_weights_and_get_sampling_client(name="0000")
```

**Saving to resume training:**
```python
# Save a checkpoint to resume from
resume_path_info = training_client.save_state(name="0010").result()
resume_path = resume_path_info.path

# Load the checkpoint to resume training
training_client.load_state(resume_path)
```

### Response

#### Success Response (200)

- `path` (str) - The fully-qualified, persistent path to the saved checkpoint (e.g., "tinker://<model_id>/<name>").

#### Response Example

**For `save_weights_for_sampler` and `save_state`:**
```json
{
  "path": "tinker://<model_id>/0000"
}
```
```

--------------------------------

### Pipelined Async Operations for Efficiency (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This Python code snippet illustrates a pipelined asynchronous approach for maximizing clock cycle efficiency. Both forward-backward and optimization steps are submitted immediately within the same clock cycle N, followed by waiting for their results, thereby utilizing clock cycles more effectively.

```python
# Submit both requests immediately. They'll both be slotted into the same clock cycle N
fwd_bwd_future = await client.forward_backward_async(batch, loss_fn)
optim_future = await client.optim_step_async(adam_params)

# Now wait for results - both operations happen on cycle N
fwd_bwd_result = await fwd_bwd_future
optim_result = await optim_future

# Total: both operations on cycle N
# This takes 1 clock cycle
```

--------------------------------

### Calculate LoRA Parameter Count

Source: https://tinker-docs.thinkingmachines.ai/lora-primer

This utility calculates the number of trainable parameters when using LoRA for a given model and LoRA rank. This is useful for determining if the number of LoRA parameters is sufficient for tasks like supervised fine-tuning on large datasets.

```python
from tinker_cookbook.hyperparam_utils import get_lora_param_count

model_name = "meta-llama/Llama-3.1-8B"
print(get_lora_param_count(model_name, lora_rank=32))
```

--------------------------------

### OpenAI Compatible Inference Endpoints

Source: https://tinker-docs.thinkingmachines.ai/compatible-apis/openai

Tinker provides OpenAI-compatible API endpoints for inference. You can use standard OpenAI SDKs or HTTP clients by overriding the base URL and providing a Tinker sampler weight path as the model name. Authentication is done using your Tinker API key.

```APIDOC
## OpenAI Compatible Inference Endpoints

### Description
This section details how to utilize Tinker's OpenAI-compatible inference, allowing interaction with model checkpoints using familiar OpenAI API structures. It supports both `/completions` and `/chat/completions` endpoints.

### Method
POST

### Endpoint
`/services/tinker-prod/oai/api/v1/completions`
`/services/tinker-prod/oai/api/v1/chat/completions`

### Parameters
#### Query Parameters
None

#### Headers
- **Authorization** (string) - Required - Your Tinker API key, e.g., `Bearer YOUR_TINKER_API_KEY`

#### Request Body
* **model** (string) - Required - A Tinker sampler weight path, e.g., `tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080`
* **prompt** (string) - Required (for `/completions`) - The input prompt for the model.
* **messages** (array) - Required (for `/chat/completions`) - A list of message objects representing the conversation.
* **max_tokens** (integer) - Optional - The maximum number of tokens to generate.
* **temperature** (number) - Optional - Controls randomness; higher values mean more random output.
* **top_p** (number) - Optional - Controls nucleus sampling; values closer to 1.0 mean more focused output.

### Request Example (Completions)
```json
{
  "model": "tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080",
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### Response
#### Success Response (200)
Returns a response object similar to the OpenAI Completions or Chat Completions API, containing generated text or message content.

* **choices** (array) - The generated completions or chat messages.
    * **text** (string) - The generated text for completions.
    * **message** (object) - The generated message for chat completions.
        * **content** (string) - The content of the message.

#### Response Example (Completions)
```json
{
  "id": "cmpl-xxxxxxxxxxxxxxxxx",
  "object": "text_completion",
  "created": 1677652288,
  "model": "tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080",
  "choices": [
    {
      "text": " Paris.",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 2,
    "total_tokens": 8
  }
}
```

### Notes
* The `BASE_URL` for these endpoints is `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`.
* Any valid Tinker sampler checkpoint path can be used as the `model` value.
* For chat requests, if your checkpoint expects a different chat template than the default Hugging Face one, you need to render the prompt yourself and use the `/completions` endpoint.
```

--------------------------------

### Save Weights for Sampler Client

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Saves model weights specifically for use with a SamplingClient. It takes a name for the saved weights and returns an APIFuture with the save response, including the path to the sampler weights. An asynchronous version is provided.

```python
def save_weights_for_sampler(
        name: str) -> APIFuture[types.SaveWeightsForSamplerResponse]:
    """
    Save model weights for use with a SamplingClient.
    Args:
      * `name`: Name for the saved sampler weights


    Returns:
      * `APIFuture` containing the save response with sampler path
    """
    pass

# Example:
# Save weights for inference
# save_future = training_client.save_weights_for_sampler("sampler-001")
# result = await save_future
# print(f"Sampler weights saved to: {result.path}")
# 
# # Use the path to create a sampling client
# sampling_client = service_client.create_sampling_client(
#     model_path=result.path
# )
```

```python
async def save_weights_for_sampler_async(
        name: str) -> APIFuture[types.SaveWeightsForSamplerResponse]:
    """
    Async version of save_weights_for_sampler.
    """
    pass
```

--------------------------------

### Check Checkpoint Publication Status with Tinker CLI

Source: https://tinker-docs.thinkingmachines.ai/publish-weights

Retrieves detailed information about a specific checkpoint, including its publication status. The output table shows various properties like 'Checkpoint ID', 'Type', 'Size', and crucially, the 'Public' status (Yes/No). This is useful for verifying if a checkpoint has been successfully published.

```bash
tinker checkpoint info tinker://14bdf3a1-0b95-55c7-8659-5edb1bc870af/weights/checkpoint_id_to_publish
```

--------------------------------

### Collect and Process Sweep Results (Python)

Source: https://tinker-docs.thinkingmachines.ai/supervised-learning/sweep-case-study

Aggregates metrics from multiple supervised learning experiments, extracts the final loss and learning rate, and stores them in a pandas DataFrame. This script reads `metrics.jsonl` and `config.json` files from specified log paths. It filters out incomplete experiments and returns a list of dictionaries containing experiment metadata and final loss.

```python
from glob import glob
import pandas
import os
import json
 
data = []
for fname in sorted(glob(os.path.expanduser("/tmp/sft-lr-sweep/*/metrics.jsonl"))):
    df = pandas.read_json(fname, lines=True)
    # make sure the experiment is completed
    if len(df) == 0 or df["progress"].iloc[-1] < 0.98:
        continue
    config_fname = fname.replace("metrics.jsonl", "config.json")
    with open(config_fname, "rb") as f:
        metadata = json.load(f)
    data.append({
        "fname": fname,
        "learning_rate": metadata["learning_rate"],
        "final_loss": df["train_mean_nll"].iloc[-1].item()
    })
 
print(f"Read metrics for {len(data)} experiments")
```

--------------------------------

### Create ModelInput from integers

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

This method allows the creation of a ModelInput object from a list of integer tokens. It is a class method, meaning it is called on the class itself rather than an instance. The input is a list of integers representing token IDs.

```python
def from_ints(cls, tokens: List[int]) -> "ModelInput"

```

--------------------------------

### PPO Loss Calculation in PyTorch

Source: https://tinker-docs.thinkingmachines.ai/llms

Calculates the Proximal Policy Optimization (PPO) loss using PyTorch. It involves clipping the probability ratio and taking the minimum between the unclipped and clipped objectives. The final loss is the negative sum of the PPO objective. Dependencies include PyTorch tensors for probability ratios, advantages, and thresholds.

```python
clipped_ratio = torch.clamp(prob_ratio, clip_low_threshold, clip_high_threshold)
unclipped_objective = prob_ratio * advantages
clipped_objective = clipped_ratio * advantages
ppo_objective = torch.min(unclipped_objective, clipped_objective)
loss = -ppo_objective.sum()
```

--------------------------------

### Define TrainingRunsResponse for Training Run Listing

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the TrainingRunsResponse class, used to return a list of training runs. It contains a list of training runs and pagination cursor information for navigating through results.

```python
class TrainingRunsResponse(BaseModel)

#### `training_runs`
List of training runs
#### `cursor`
Pagination cursor information
```

--------------------------------

### Cross-Entropy Loss Calculation (Python)

Source: https://tinker-docs.thinkingmachines.ai/losses

Illustrates the calculation of the cross-entropy loss for supervised learning. It involves applying token-level weights to the negative log-probabilities of the target tokens and then summing the result to obtain the total loss.

```python
# Apply weights and compute elementwise loss
elementwise_loss = -target_logprobs * weights
# Apply sum reduction to get the total loss
loss = elementwise_loss.sum()  # scalar
```

--------------------------------

### Calculate LoRA LR Scaling Factor

Source: https://tinker-docs.thinkingmachines.ai/lora-primer

This utility calculates the factor by which to scale a full fine-tuning learning rate to obtain an equivalent LoRA learning rate. It is crucial for successful LoRA fine-tuning, as LoRA typically requires much higher learning rates than full fine-tuning.

```python
from tinker_cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr

model_name = "meta-llama/Llama-3.1-8B"
print(get_lora_lr_over_full_finetune_lr(model_name))
```

--------------------------------

### save_weights_for_sampler_async

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Asynchronously saves model weights for use with a SamplingClient.

```APIDOC
## save_weights_for_sampler_async

### Description
Asynchronous version of `save_weights_for_sampler`. Saves model weights for use with a SamplingClient.

### Method
POST

### Endpoint
/save_weights_for_sampler_async

### Parameters
#### Request Body
- **name** (string) - Required - A name for the saved sampler weights.

### Request Example
```json
{
  "name": "sampler-001"
}
```

### Response
#### Success Response (200)
- **APIFuture** (APIFuture[types.SaveWeightsForSamplerResponse]) - Contains the save response, including the sampler path.

#### Response Example
```json
{
  "future_id": "save_sampler_async_19202",
  "path": "tinker://runs/run-id/sampler_weights/sampler-001"
}
```
```

--------------------------------

### Define RL Dataset Structure - Python

Source: https://tinker-docs.thinkingmachines.ai/rl/rl-envs

Represents a dataset of `EnvGroupBuilder` instances. The `get_batch` method is used to retrieve a list of `EnvGroupBuilder` objects for training. This modular structure allows for flexible data loading independent of the environment.

```python
class RLDataset:
    """
    Dataset of EnvGroupBuilders.
    """

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        raise NotImplementedError
```

--------------------------------

### The Renderer Class

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Introduces the Renderer class, the primary interface for converting message data types into token representations for training and inference.

```APIDOC
## The Renderer Class

### Description
The `Renderer` class is the central interface for converting list-of-message data types into their token representations, suitable for model training and inference. It is designed for the complete training lifecycle, encompassing supervised learning, reinforcement learning, and deployment.

### Location
The `Renderer` class can be found in `renderers.py` within the Tinker cookbook.

### Usage Example
```python
messages=[
    {'role': 'system', 'content': 'Answer concisely; at most one sentence per response'},
    {'role': 'user', 'content': 'What is the longest-lived rodent species?'},
    {'role': 'assistant', 'content': 'The naked mole rat, which can live over 30 years.'},
    {'role': 'user', 'content': 'How do they live so long?'},
    {'role': 'assistant', 'content': 'They evolved multiple protective mechanisms including special hyaluronic acid that prevents cancer, extremely stable proteins, and efficient DNA repair systems that work together to prevent aging.'}
]

# Assume 'renderer' is an instance of the Renderer class
# tokenized_output = renderer.render(messages)
```

### Related
For a practical implementation, see the [CookbookLink](https://github.com/thinkingmachines/tinker/blob/main/tinker_cookbook/renderers.py) for `renderers.py`.
```

--------------------------------

### SaveWeightsResponse Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Response containing the URI for saved model weights.

```APIDOC
## SaveWeightsResponse Objects
```
class SaveWeightsResponse(BaseModel)
```

#### `path`
A tinker URI for model weights at a specific step
```

--------------------------------

### TokenCompleter Interface

Source: https://tinker-docs.thinkingmachines.ai/llms-full

The TokenCompleter is the foundational interface used by RL algorithms because they work directly with tokens. It defines how to generate token sequences with optional log probabilities.

```APIDOC
## TokenCompleter Interface

### Description
The `TokenCompleter` is the foundational interface used by RL algorithms because they work directly with tokens. It defines the method for generating token sequences with optional log probabilities.

### Method
`__call__`

### Endpoint
N/A (Interface Definition)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **model_input** (`types.ModelInput`) - Required - The input to the model.
- **stop** (`StopCondition`) - Required - Stop conditions, either a list of strings or token IDs.

### Request Example
```python
# This is a conceptual example as TokenCompleter is an interface.
# Actual implementation would involve instantiating a concrete class.
async def generate_tokens(completer: TokenCompleter, model_input: types.ModelInput, stop_condition: StopCondition):
    return await completer(model_input, stop_condition)
```

### Response
#### Success Response (200)
- **tokens** (list[str]) - The generated token sequence.
- **maybe_logprobs** (list[float] | None) - Optional log probabilities for each token.

#### Response Example
```json
{
  "tokens": ["token1", "token2", "token3"],
  "maybe_logprobs": [-0.1, -0.2, -0.3]
}
```
```

--------------------------------

### Visualize Hyperparameter Sweep Results (Python)

Source: https://tinker-docs.thinkingmachines.ai/supervised-learning/sweep-case-study

Generates a plot showing the relationship between learning rate and final loss from collected sweep results. This function uses matplotlib to create a scatter plot with a logarithmic x-axis and highlights the minimum loss. It helps in visually identifying the optimal learning rate.

```python
import matplotlib.pyplot as plt
df = pandas.DataFrame(data)
plt.plot(df["learning_rate"], df["final_loss"], marker='o')
plt.axhline(y=df["final_loss"].min(), color="green", linestyle="--")
plt.ylim(1.65, 1.8)
plt.xscale("log")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Final Loss")
plt.title("Final Loss vs Learning Rate")
plt.show()
```

--------------------------------

### List User Checkpoints API

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Retrieves a paginated list of all checkpoints for the authenticated user across all their training runs, sorted by time (newest first).

```APIDOC
## GET /list_user_checkpoints

### Description
List all checkpoints for the current user across all their training runs. This method retrieves checkpoints from all training runs owned by the authenticated user, sorted by time (newest first). It supports pagination for efficiently handling large numbers of checkpoints.

### Method
GET

### Endpoint
/list_user_checkpoints

### Parameters
#### Path Parameters
None

#### Query Parameters
- **limit** (int) - Optional - Maximum number of checkpoints to return (default 100)
- **offset** (int) - Optional - Offset for pagination (default 0)

### Request Example
(No request body for GET requests)

### Response
#### Success Response (200)
- **checkpoints** (array) - A list of checkpoint objects.
- **cursor** (object) - Pagination information.
  - **total_count** (int) - The total number of checkpoints available.
  - **offset** (int) - The current offset.
  - **limit** (int) - The current limit.

#### Response Example
```json
{
  "checkpoints": [
    {
      "training_run_id": "run-id-1",
      "checkpoint_id": "0001",
      "timestamp": "2023-12-18T10:00:00Z"
    },
    {
      "training_run_id": "run-id-2",
      "checkpoint_id": "0005",
      "timestamp": "2023-12-17T15:30:00Z"
    }
  ],
  "cursor": {
    "total_count": 150,
    "offset": 0,
    "limit": 50
  }
}
```
```

--------------------------------

### List Checkpoints API

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Lists available checkpoints for a given training run ID. Supports both synchronous and asynchronous operations.

```APIDOC
## LIST CHECKPOINTS

### Description
Lists available checkpoints (both training and sampler) for a specific training run.

### Method
GET

### Endpoint
`/training_runs/{training_run_id}/checkpoints`

### Parameters
#### Path Parameters
- **training_run_id** (string) - Required - The ID of the training run to list checkpoints for.

### Request Example
```python
# Synchronous
future = rest_client.list_checkpoints("run-id")
response = future.result()

# Asynchronous
response = await rest_client.list_checkpoints_async("run-id")

for checkpoint in response.checkpoints:
    if checkpoint.checkpoint_type == "training":
        print(f"Training checkpoint: {checkpoint.checkpoint_id}")
    elif checkpoint.checkpoint_type == "sampler":
        print(f"Sampler checkpoint: {checkpoint.checkpoint_id}")
```

### Response
#### Success Response (200)
- **checkpoints** (array) - A list of checkpoint objects.
  - **checkpoint_id** (string) - The unique identifier for the checkpoint.
  - **checkpoint_type** (string) - The type of the checkpoint (e.g., 'training', 'sampler').

#### Response Example
```json
{
  "checkpoints": [
    {
      "checkpoint_id": "checkpoint-123",
      "checkpoint_type": "training"
    },
    {
      "checkpoint_id": "sampler-456",
      "checkpoint_type": "sampler"
    }
  ]
}
```
```

--------------------------------

### Tinker Library Sync vs. Async API Calls

Source: https://tinker-docs.thinkingmachines.ai/async

Demonstrates the naming convention for synchronous and asynchronous methods in the Tinker Python library. Async variants typically append '_async' to the synchronous method name.

```python
from tinker import ServiceClient, TrainingClient, SamplingClient, RestClient

# Synchronous client creation
service_client = ServiceClient()
training_client = TrainingClient()
sampling_client = SamplingClient()
rest_client = RestClient()

# Asynchronous client creation (convention)
# async def create_async_clients():
#     service_client_async = await ServiceClient.create_lora_training_client_async()
#     training_client_async = TrainingClient()
#     sampling_client_async = SamplingClient()
#     rest_client_async = RestClient()

# Example of calling sync and async methods
# sync_result = training_client.forward()
# async_result = await training_client.forward_async()
# sync_list = rest_client.list_training_run_ids()
# async_list = await rest_client.list_training_run_ids_async()
```

--------------------------------

### Asynchronous Session Listing (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Enables asynchronous listing of sessions with pagination support. Similar to `list_sessions`, this method returns a `ListSessionsResponse` asynchronously.

```python
async def list_sessions_async(limit: int = 20,
                              offset: int = 0) -> types.ListSessionsResponse

Async version of list_sessions.
```

--------------------------------

### Define TrainingRun for Training Run Details

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the TrainingRun class, providing detailed information about a specific training run. It includes the training run ID, base model, model owner, LoRA status, corruption status, LoRA rank, and timestamps.

```python
class TrainingRun(BaseModel)

#### `training_run_id`
The unique identifier for the training run
#### `base_model`
The base model name this model is derived from
#### `model_owner`
The owner/creator of this model
#### `is_lora`
Whether this model uses LoRA (Low-Rank Adaptation)
#### `corrupted`
Whether the model is in a corrupted state
#### `lora_rank`
The LoRA rank if this is a LoRA model, null otherwise
#### `last_request_time`
The timestamp of the last request made to this model
#### `last_checkpoint`
The most recent training checkpoint, if available
#### `last_sampler_checkpoint`
The most recent sampler checkpoint, if available
#### `user_metadata`
Optional metadata about this training run, set by the end-user
```

--------------------------------

### Bad Request Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

HTTP 400: Indicates the request was invalid or malformed.

```APIDOC
## BadRequestError Objects

### Description
HTTP 400: The request was invalid or malformed.

### Class Definition
```python
class BadRequestError(APIStatusError)
```
```

--------------------------------

### Define SampleRequest for Generating Samples

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the SampleRequest class, used for requesting samples from a model. It includes parameters for the number of samples, base model, model path, sampling session ID, sequence ID, and log probability settings.

```python
class SampleRequest(StrictBase)

#### `num_samples`
Number of samples to generate
#### `base_model`
Optional base model name to sample from. Is inferred from model_path, if provided. If sampling against a base model, this is required.
#### `model_path`
Optional tinker:// path to your model weights or LoRA weights. If not provided, samples against the base model.
#### `sampling_session_id`
Optional sampling session ID to use instead of model_path/base_model. If provided along with seq_id, the model configuration will be loaded from the sampling session. This is useful for multi-turn conversations.
#### `seq_id`
Sequence ID within the sampling session. Required when sampling_session_id is provided. Used to generate deterministic request IDs for the sampling request.
#### `prompt_logprobs`
If set to `true`, computes and returns logprobs on the prompt tokens. Defaults to false.
#### `topk_prompt_logprobs`
If set to a positive integer, returns the top-k logprobs for each prompt token.
```

--------------------------------

### Tinker Renderer Class Usage

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Illustrates the basic usage of the Renderer class in Tinker for converting message datatypes into token representations. This is crucial for model training and inference, supporting various learning paradigms.

```python
messages =[
    {'role': 'system', 'content': 'Answer concisely; at most one sentence per response'},
    {'role': 'user', 'content': 'What is the longest-lived rodent species?'},
    {'role': 'assistant', 'content': 'The naked mole rat, which can live over 30 years.'},
    {'role': 'user', 'content': 'How do they live so long?'},
    {'role': 'assistant', 'content': 'They evolved multiple protective mechanisms including special hyaluronic acid that prevents cancer, extremely stable proteins, and efficient DNA repair systems that work together to prevent aging.'}
]

# Assuming 'renderer' is an instance of the Renderer class
# tokens = renderer.render(messages)
```

--------------------------------

### Save Model Weights

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Saves the current model weights to persistent storage. Requires a name for the checkpoint and returns an APIFuture with the save response, including the checkpoint path. An async version is available.

```python
def save_state(name: str) -> APIFuture[types.SaveWeightsResponse]:
    """
    Save model weights to persistent storage.
    Args:
      * `name`: Name for the saved checkpoint


    Returns:
      * `APIFuture` containing the save response with checkpoint path
    """
    pass

# Example:
# Save after training
# save_future = training_client.save_state("checkpoint-001")
# result = await save_future
# print(f"Saved to: {result.path}")
```

```python
async def save_state_async(name: str) -> APIFuture[types.SaveWeightsResponse]:
    """
    Async version of save_state.
    """
    pass
```

--------------------------------

### load_state

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Loads model weights from a specified checkpoint path. This does not load optimizer state.

```APIDOC
## load_state

### Description
Load model weights from a saved checkpoint. This loads only the model weights, not optimizer state (e.g., Adam momentum).

### Method
POST

### Endpoint
/load_state

### Parameters
#### Request Body
- **path** (string) - Required - The Tinker path to the saved weights (e.g., "tinker://run-id/weights/checkpoint-001").

### Request Example
```json
{
  "path": "tinker://run-id/weights/checkpoint-001"
}
```

### Response
#### Success Response (200)
- **APIFuture** (APIFuture[types.LoadWeightsResponse]) - Contains the load response.

#### Response Example
```json
{
  "future_id": "load_state_77889"
}
```
```

--------------------------------

### optim_step

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Updates model parameters using the Adam optimizer. This is a synchronous operation.

```APIDOC
## optim_step

### Description
Update model parameters using Adam optimizer.

### Method
POST

### Endpoint
/optim_step

### Parameters
#### Request Body
- **adam_params** (types.AdamParams) - Required - Adam optimizer parameters including learning_rate, betas, eps, weight_decay.

### Request Example
```json
{
  "adam_params": {
    "learning_rate": 0.0001,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01
  }
}
```

### Response
#### Success Response (200)
- **APIFuture** (APIFuture[types.OptimStepResponse]) - Contains the optimizer step response.

#### Response Example
```json
{
  "future_id": "optim_step_12345"
}
```
```

--------------------------------

### Handling Future Objects in Async Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Explains the use of Future objects in asynchronous Python with the Tinker library. It requires a double 'await': the first to submit the request and the second to retrieve the final computation result.

```python
future = await client.forward_backward_async(data, loss_fn)
result = await future
```

--------------------------------

### Conceptual Gradient Calculation for Custom Loss in Python

Source: https://tinker-docs.thinkingmachines.ai/losses

This snippet illustrates the mathematical concept behind `forward_backward_custom` by showing how a nonlinear loss function can be approximated by a surrogate loss that is linear in the log-probabilities. This enables Tinker to compute gradients efficiently without pickling the custom function.

```python
loss = compute_loss_from_logprobs(compute_target_logprobs(params))

logprobs = compute_target_logprobs(params)
surrogate_loss = (logprobs * logprob_grads).sum()
# where logprob_grads = dLoss/dLogprobs
```

--------------------------------

### CreateModelRequest Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the parameters required to create a new model training run.

```APIDOC
## CreateModelRequest Objects
```
class CreateModelRequest(StrictBase)
```

#### `base_model`
The name of the base model to fine-tune (e.g., 'Qwen/Qwen3-8B').
#### `user_metadata`
Optional metadata about this model/training run, set by the end-user.
#### `lora_config`
LoRA configuration
```

--------------------------------

### Create empty ModelInput

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

This class method creates an empty ModelInput object. It is called on the class itself and returns a new, empty ModelInput instance.

```python
def empty(cls) -> "ModelInput"

```

--------------------------------

### TokenCompleter Interface for RL

Source: https://tinker-docs.thinkingmachines.ai/completers

The TokenCompleter interface is the base for RL algorithms, operating directly on tokens. It accepts model input and stop conditions, returning generated tokens with optional log probabilities. This is crucial for optimizing token sequences during RL training.

```python
class TokenCompleter:
    async def __call__(
        self, model_input: types.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:

```

--------------------------------

### Define SaveWeightsResponse for Model Weight Paths

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the SaveWeightsResponse class, which provides a tinker URI for model weights at a specific step. This is useful for referencing and retrieving saved model checkpoints.

```python
class SaveWeightsResponse(BaseModel)

#### `path`
A tinker URI for model weights at a specific step
```

--------------------------------

### Cross Entropy Loss for Supervised Learning (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms

Implements the standard cross-entropy loss for supervised learning, optimizing the policy to maximize the log-probability of target tokens. It requires target tokens and weights as input and outputs token-level log probabilities and the total weighted loss. Weights are typically 0 or 1.

```python
elementwise_loss = -target_logprobs * weights
loss = elementwise_loss.sum()
```

--------------------------------

### Non-Pipelined Async Operations (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms-full

This Python code snippet demonstrates a non-pipelined asynchronous approach for handling forward-backward and optimization steps. It submits one operation, waits for its completion, and then submits the next, which can lead to inefficient use of clock cycles.

```python
# Due to communication latency, this happens a little after cycle N+1 started
fwd_bwd_result = await fwd_bwd_future

# Submit optim_step, gets queued for clock cycle N+2
optim_future = await client.optim_step_async(adam_params)

# Wait for it to complete, and for client to receive the result
# This happens a little after cycle N+2 finishes
optim_result = await optim_future

# Total: forward_backward on cycle N, optim_step on cycle N+2
# This takes 3 clock cycles (plus the time we waited before cycle N started)
```

--------------------------------

### Internal Server Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

HTTP 500+: Indicates an error occurred on the server.

```APIDOC
## InternalServerError Objects

### Description
HTTP 500+: An error occurred on the server.

### Class Definition
```python
class InternalServerError(APIStatusError)
```
```

--------------------------------

### Publish Checkpoint from Tinker Path API

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Publishes a checkpoint referenced by its Tinker path, making it publicly accessible. Supports both synchronous and asynchronous operations.

```APIDOC
## PUBLISH CHECKPOINT FROM TINKER PATH

### Description
Publishes a checkpoint referenced by a Tinker path, making it publicly accessible. Only the owner of the training run can publish checkpoints.

### Method
POST

### Endpoint
`/checkpoints/tinker/{tinker_path}/publish`

### Parameters
#### Path Parameters
- **tinker_path** (string) - Required - The Tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001").

### Request Example
```python
# Synchronous
future = rest_client.publish_checkpoint_from_tinker_path("tinker://run-id/weights/0001")
future.result()  # Wait for completion
print("Checkpoint published successfully")

# Asynchronous
await rest_client.publish_checkpoint_from_tinker_path_async("tinker://run-id/weights/0001")
print("Checkpoint published successfully")
```

### Response
#### Success Response (200)
Indicates successful publication. No content is returned.

#### Error Responses
- **HTTPException: 400** - If the checkpoint identifier is invalid.
- **HTTPException: 404** - If the checkpoint is not found or the user does not own the training run.
- **HTTPException: 409** - If the checkpoint is already public.
- **HTTPException: 500** - If there is an error publishing the checkpoint.
```

--------------------------------

### Cross Entropy Loss (Supervised Learning)

Source: https://tinker-docs.thinkingmachines.ai/losses

Details the implementation and input/output requirements for the `cross_entropy` loss function, suitable for supervised learning tasks.

```APIDOC
## Supervised learning: `cross_entropy`

### Description
Implements the standard cross-entropy loss (negative log-likelihood) to optimize the policy `pθ` to maximize the log-probability of the target tokens `xxx`. This is typically used for supervised learning.

### Method
`forward_backward` or `forward_backward_async` with `loss_fn="cross_entropy"`

### Parameters
#### Request Body (subset of `loss_fn_inputs`)
- **target_tokens**: `array[(N,), int]` - Target token IDs. Required.
- **weights**: `array[(N,), float]` - Token-level loss weights, typically generated from `renderer.build_supervised_example()`. Required.

### Request Example
```python
import tinker
import torch
from tinker import TensorData

datum = tinker.Datum(
    model_input=input_tokens,
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(target_token_ids)),
        "weights": TensorData.from_torch(torch.tensor(token_weights)),
    }
)

fwd_bwd_result = await training_client.forward_backward_async(
    [datum], loss_fn="cross_entropy"
)
```

### Response
#### Success Response (200) - `loss_fn_outputs`
- **logprobs**: `array[(N,), float]` - Log probabilities of predicted tokens.
- **loss:sum**: `scalar` - Sum of weighted cross-entropy losses.

#### Response Example
```json
{
  "loss_fn_outputs": {
    "logprobs": [-0.5, -0.2, -0.8],
    "loss:sum": 0.75
  }
}
```
```

--------------------------------

### Policy Gradient: Importance Sampling

Source: https://tinker-docs.thinkingmachines.ai/losses

Explains the importance sampling loss function for reinforcement learning, addressing the bias correction needed when the learner and sampling policies differ.

```APIDOC
## Policy gradient: `importance_sampling`

### Description
Implements a variant of the policy gradient objective using importance sampling to correct for bias when the learner policy `pθ` differs from the sampling policy `q`. This ensures an unbiased estimation of the expected reward.

### Method
`forward_backward` or `forward_backward_async` with `loss_fn="importance_sampling"`

### Parameters
#### Request Body (subset of `loss_fn_inputs`)
- **target_logprobs**: `array[(N,), float]` - Log probabilities from the learner policy (`pθ`). Provided during the forward pass.
- **sampling_logprobs**: `array[(N,), float]` - Log probabilities from the sampling policy (`q`), recorded during sampling.
- **advantages**: `array[(N,), float]` - Advantage values (`A`).

### Request Example
```python
import tinker
import torch
from tinker import TensorData

datum = tinker.Datum(
    model_input=input_tokens,
    loss_fn_inputs={
        "target_logprobs": TensorData.from_torch(torch.tensor(learner_logprobs)),
        "sampling_logprobs": TensorData.from_torch(torch.tensor(sampler_logprobs)),
        "advantages": TensorData.from_torch(torch.tensor(advantages)),
    }
)

fwd_bwd_result = await training_client.forward_backward_async(
    [datum], loss_fn="importance_sampling"
)
```

### Response
#### Success Response (200) - `loss_fn_outputs`
- **logprobs**: `array[(N,), float]` - Log probabilities of predicted tokens from the learner policy (`pθ`).
- **prob_ratio**: `array[(N,), float]` - The ratio `pθ(x) / q(x)` used in the importance sampling calculation.

#### Response Example
```json
{
  "loss_fn_outputs": {
    "logprobs": [-0.5, -0.2, -0.8],
    "prob_ratio": [1.1, 0.9, 1.05]
  }
}
```
```

--------------------------------

### Configure DRO Loss with Custom Beta Parameter

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Illustrates how to set the `beta` hyper-parameter for the DRO loss function when using `forward_backward_async`. The `beta` value influences the quadratic penalty term, controlling the constraint on policy updates. This configuration is provided within the `loss_fn_config` dictionary.

```python
fwd_bwd_result = await training_client.forward_backward_async(
    data=data,
    loss_fn="dro",
    loss_fn_config={\"beta\": 0.05}
)
```

--------------------------------

### Apply CISPO Loss Function in Python

Source: https://tinker-docs.thinkingmachines.ai/losses

This snippet demonstrates how to apply the CISPO loss function using PyTorch. It involves clamping probability ratios and calculating the CISPO objective, which is then used to compute the final loss. The loss is then used with the training client.

```python
clipped_ratio = torch.clamp(prob_ratio, clip_low_threshold, clip_high_threshold)
# Compute CISPO objective (detach the clipped ratio)
cispo_objective = clipped_ratio.detach() * target_logprobs * advantages
# CISPO loss is negative of objective
loss = -cispo_objective.sum()
```

--------------------------------

### Proximal Policy Optimization (PPO)

Source: https://tinker-docs.thinkingmachines.ai/losses

Proximal Policy Optimization (PPO) is a policy gradient method that uses a clipping objective to limit policy updates, preventing excessively large changes in the policy space. This implementation calculates the PPO loss token-wise.

```APIDOC
## POST /compute/importance_weighted_loss

### Description
Computes the importance-weighted policy gradient loss, specifically for Proximal Policy Optimization (PPO).

### Method
POST

### Endpoint
/compute/importance_weighted_loss

### Parameters
#### Query Parameters
- **loss_fn** (string) - Required - The loss function to use, should be 'ppo'.
- **loss_fn_config** (object) - Optional - Configuration for the loss function. Supports `clip_low_threshold` and `clip_high_threshold`.
  - **clip_low_threshold** (float) - Optional - The lower bound for clipping the probability ratio. Defaults to 1 - epsilon.
  - **clip_high_threshold** (float) - Optional - The upper bound for clipping the probability ratio. Defaults to 1 + epsilon.

#### Request Body
- **target_tokens** (array[int]) - Required - Target token IDs.
- **logprobs** (array[float]) - Required - Sampling log probabilities for the tokens.
- **advantages** (array[float]) - Required - Advantage values for Reinforcement Learning.
- **sampling_logprobs** (array[float]) - Required - Log probabilities from the sampling distribution.

### Request Example
```json
{
  "target_tokens": [101, 1996, 2744, 2219, 102],
  "logprobs": [-0.1, -0.5, -0.3, -0.7, -0.2],
  "advantages": [0.5, -0.1, 0.3, 0.2, 0.4],
  "sampling_logprobs": [-0.2, -0.4, -0.2, -0.6, -0.3]
}
```

### Response
#### Success Response (200)
- **loss** (object) - Contains the computed loss.
  - **sum** (float) - The sum of the importance-weighted policy gradient losses.
- **logprobs** (array[float]) - Target log probabilities for the tokens.
- **diagnostics** (object) - Diagnostic information.
  - **loss.sum** (scalar) - Sum of importance-weighted policy gradient losses LIS.

#### Response Example
```json
{
  "loss": {
    "sum": 1.52
  },
  "logprobs": [-0.1, -0.5, -0.3, -0.7, -0.2],
  "diagnostics": {
    "loss.sum": 1.52
  }
}
```
```

--------------------------------

### Define CreateModelRequest for Fine-tuning

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the CreateModelRequest class, used to initiate a model fine-tuning process. It specifies the base model, optional user metadata, and LoRA configuration.

```python
class CreateModelRequest(StrictBase)

#### `base_model`
The name of the base model to fine-tune (e.g., 'Qwen/Qwen3-8B').
#### `user_metadata`
Optional metadata about this model/training run, set by the end-user.
#### `lora_config`
LoRA configuration
```

--------------------------------

### ForwardBackwardOutput Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Output from a forward-backward pass, including loss and metrics.

```APIDOC
## ForwardBackwardOutput Objects
```
class ForwardBackwardOutput(BaseModel)
```

#### `loss_fn_output_type`
The class name of the loss function output records (e.g., 'TorchLossReturn', 'ArrayRecord').
#### `loss_fn_outputs`
Dictionary mapping field names to tensor data
#### `metrics`
Training metrics as key-value pairs
```

--------------------------------

### Convert ModelInput to integers

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

This method converts a ModelInput object into a list of integer tokens. It is an instance method and returns a list of integers. Note that this method may throw an exception if the ModelInput contains non-token chunks.

```python
def to_ints() -> List[int]

```

--------------------------------

### Using Built-in Loss Functions

Source: https://tinker-docs.thinkingmachines.ai/losses

This section details how to utilize the predefined loss functions within the Tinker API by passing a string identifier to the `forward_backward` method. It covers the expected input tensors and the structure of the returned output.

```APIDOC
## Using Built-in Loss Functions

### Description
Utilize Tinker's predefined loss functions by specifying a string identifier to the `forward_backward` method. This method expects specific input tensors and returns a `ForwardBackwardOutput` object.

### Method
`forward_backward` or `forward_backward_async`

### Parameters
#### Request Body
- **datum** (tinker.Datum) - Required - Contains model inputs and loss function specific inputs.
  - **model_input**: (object) - The input tokens for the model.
  - **loss_fn_inputs**: (dict) - A dictionary mapping string identifiers to numpy or torch tensors required by the specified loss function.
    - **target_tokens** (array[(N,), int]) - Required for supervised learning. Target token IDs.
    - **weights** (array[(N,), float]) - Required for supervised learning. Token-level loss weights.
    - **logprobs** (array[(N,), float]) - Required for RL. Log probabilities of predicted tokens from the learner policy.
    - **sampling_logprobs** (array[(N,), float]) - Required for RL. Log probabilities of predicted tokens from the sampling policy.
    - **advantages** (array[(N,), float]) - Required for RL. Advantage values.

- **loss_fn** (string) - Required - The identifier for the loss function to use. Supported values: `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, `dro`.

### Request Example
```python
import tinker
import torch
from tinker import TensorData

# Assuming input_tokens, target_tokens, sampling_logprobs, advantages are defined

datum = tinker.Datum(
    model_input=input_tokens,
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
        "logprobs": TensorData.from_torch(torch.tensor(sampling_logprobs)),  # Reference logprobs
        "advantages": TensorData.from_torch(torch.tensor(advantages)),
    }
)

# Option 1: Use importance sampling REINFORCE
fwd_bwd_result = await training_client.forward_backward_async(
    [datum], loss_fn="importance_sampling"
)

# Option 2: Use PPO with clipping
fwd_bwd_result = await training_client.forward_backward_async(
    [datum], loss_fn="ppo"
)
```

### Response
#### Success Response (200)
- **ForwardBackwardOutput** (object) - Contains the results of the forward-backward pass.
  - **loss_fn_outputs**: (dict) - Output tensors from the loss function.
    - **logprobs** (array[(N,), float]) - Log probabilities of predicted tokens.
    - **loss:sum** (scalar) - The computed total loss (for cross_entropy).
    - Other loss-specific outputs may be present.

#### Response Example
```json
{
  "loss_fn_outputs": {
    "logprobs": [0.1, 0.2, 0.3],
    "loss:sum": 1.5
  }
}
```
```

--------------------------------

### Define SampleResponse Class in Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the SampleResponse class, a Pydantic model for handling sampling results. It includes generated sequences, an optional list of log probabilities for the prompt tokens, and optional top-k prompt log probabilities.

```python
class SampleResponse(BaseModel):
    sequences: Sequence[SampledSequence]

    type: Literal["sample"] = "sample"

    prompt_logprobs: Optional[List[Optional[float]]] = None
    """
    If prompt_logprobs was set to true in the request, logprobs are computed for
    every token in the prompt. The `prompt_logprobs` response contains a float32
    value for every token in the prompt.
    """

    topk_prompt_logprobs: Optional[list[Optional[list[tuple[int, float]]]]] = None
    """
    If topk_prompt_logprobs was set to a positive integer k in the request,
    the top-k logprobs are computed for every token in the prompt. The
    `topk_prompt_logprobs` response contains, for every token in the prompt,
    a list of up to k (token_id, logprob) tuples.
    """
```

--------------------------------

### CISPO Loss Calculation in PyTorch

Source: https://tinker-docs.thinkingmachines.ai/llms

Implements the Clipped Importance Sampling Policy Optimization (CISPO) loss in PyTorch. It calculates the probability ratio, applies clipping to this ratio, and then computes the CISPO objective by multiplying the detached clipped ratio with target log-probabilities and advantages. The final loss is the negative sum of this objective.

```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
clipped_ratio = torch.clamp(prob_ratio, clip_low_threshold, clip_high_threshold)
cispo_objective = clipped_ratio.detach() * target_logprobs * advantages
loss = -cispo_objective.sum()
```

--------------------------------

### Publish Checkpoint from Tinker Path (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Publishes a checkpoint referenced by a Tinker path, making it publicly accessible. Only the owner of the training run can perform this action. The function returns a Future that completes when the checkpoint is successfully published. It can raise HTTP exceptions for various error conditions. Dependencies: `ConcurrentFuture` for asynchronous operations.

```python
def publish_checkpoint_from_tinker_path(
        tinker_path: str) -> ConcurrentFuture[None]:
    # ... implementation details ...
    pass

# Example Usage:
# future = rest_client.publish_checkpoint_from_tinker_path("tinker://run-id/weights/0001")
# future.result()  # Wait for completion
# print("Checkpoint published successfully")
```

--------------------------------

### Rate Limit Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

HTTP 429: Indicates that the rate limit has been exceeded.

```APIDOC
## RateLimitError Objects

### Description
HTTP 429: Too many requests, rate limit exceeded.

### Class Definition
```python
class RateLimitError(APIStatusError)
```
```

--------------------------------

### Add Vision Input to Model - Python

Source: https://tinker-docs.thinkingmachines.ai/training-sampling

Demonstrates how to construct a ModelInput object that includes both encoded text and image data. This is achieved by using `EncodedTextChunk` and `ImageChunk` within the `chunks` list.

```python
image_data = requests.get("https://thinkingmachines.ai/blog/on-policy-distillation/images/chess.png").content
model_input = tinker.ModelInput(chunks=[
  types.EncodedTextChunk(tokens=tokenizer.encode("<|im_start|>user\n<|vision_start|>")),
  types.ImageChunk(data=image_data, format="png"),
  types.EncodedTextChunk(tokens=tokenizer.encode("<|vision_end|>What is this?<|im_end|>\n<|im_start|>assistant\n"))
])
```

--------------------------------

### List Checkpoints Async (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

An asynchronous version of `list_checkpoints`. This function directly returns the `CheckpointsListResponse` without requiring manual handling of `Future` objects, making it suitable for use within `async` functions. Dependencies: `types` for type hinting.

```python
async def list_checkpoints_async(
        training_run_id: types.ModelID) -> types.CheckpointsListResponse:
    # ... implementation details ...
    pass
```

--------------------------------

### Incorporate Vision Inputs into ModelInput

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Demonstrates how to create a ModelInput object that includes both encoded text and image data. This is achieved by using ImageChunk for image data and EncodedTextChunk for text tokens, separated by special vision tokens. Requires the 'requests' library to fetch image data.

```python
import requests
import tinker
from tinker import types

image_data = requests.get("https://thinkingmachines.ai/blog/on-policy-distillation/images/chess.png").content
model_input = tinker.ModelInput(chunks=[
  types.EncodedTextChunk(tokens=tokenizer.encode("<|im_start|>user\n<|vision_start|>")),
  types.ImageChunk(data=image_data, format="png"),
  types.EncodedTextChunk(tokens=tokenizer.encode("<|vision_end|>What is this?<|im_end|>\n<|im_start|>assistant\n"))
])
```

--------------------------------

### Save Weights for Sampling using Python

Source: https://tinker-docs.thinkingmachines.ai/save-load

Saves a copy of model weights suitable for sampling. This method is faster and requires less storage than saving the full training state. It takes a `name` argument to identify the checkpoint and returns a persistent path.

```python
# Setup
import tinker
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.2-1B", rank=32
)
 
# Save a checkpoint that you can use for sampling
sampling_path = training_client.save_weights_for_sampler(name="0000").result().path
 
# Create a sampling client with that checkpoint
sampling_client = service_client.create_sampling_client(model_path=sampling_path) #
```

--------------------------------

### Cross Entropy Loss (Supervised Learning)

Source: https://tinker-docs.thinkingmachines.ai/llms

Implements the standard cross-entropy loss for supervised learning. It optimizes the policy to maximize the log-probability of target tokens, optionally weighted.

```APIDOC
## Cross Entropy Loss (Supervised Learning)

### Description
Implements the standard cross-entropy loss (i.e., negative log-likelihood) for supervised learning. This loss function optimizes the policy $p_\theta$ to maximize the log-probability of the target tokens $x$.

### Method
Not applicable (function within a library)

### Endpoint
Not applicable (function within a library)

### Parameters
#### Input Tensors
- **target_tokens** (array[(N,), int]) - Target token IDs.
- **weights** (array[(N,), float]) - Token-level loss weights, typically generated from `renderer.build_supervised_example()`.

### Request Example
```python
# This is a conceptual example, not a direct API call
# Assuming target_logprobs and weights are available
loss = -target_logprobs * weights
loss = loss.sum() # scalar
```

### Response
#### Output Tensors
- **logprobs** (array[(N,), float]) - Log probabilities of predicted tokens.

#### Output Diagnostics
- **loss:sum** (scalar) - Sum of weighted cross-entropy losses.

#### Response Example
```json
{
  "logprobs": [0.1, 0.5, 0.3],
  "loss:sum": 1.234
}
```
```

--------------------------------

### Define Cursor for Pagination Information

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the Cursor class, used for pagination purposes. It includes the offset, limit, and total count of items available.

```python
class Cursor(BaseModel)

#### `offset`
The offset used for pagination
#### `limit`
The maximum number of items requested
#### `total_count`
The total number of items available
```

--------------------------------

### Overlapping Requests for Performance in Tinker Async

Source: https://tinker-docs.thinkingmachines.ai/async

Shows a pattern for maximizing performance in Tinker's asynchronous operations by overlapping requests, such as `forward_backward_async` and `optim_step_async`. This ensures operations are queued efficiently to avoid missing clock cycles.

```python
# Submit forward_backward
fwd_bwd_future = await client.forward_backward_async(batch, loss_fn)
 
# Submit optim_step immediately (don't wait for forward_backward to finish)
optim_future = await client.optim_step_async(adam_params)
 
# Now retrieve results
fwd_bwd_result = await fwd_bwd_future
optim_result = await optim_future
```

--------------------------------

### Compute Direct Reward Optimization (DRO) Loss in Python

Source: https://tinker-docs.thinkingmachines.ai/losses

This snippet illustrates the calculation of the Direct Reward Optimization (DRO) loss using PyTorch. It involves computing a quadratic penalty term and then combining it with advantages and target log-probabilities to form the DRO objective, ultimately leading to the loss calculation.

```python
# Compute quadratic penalty term
quadratic_term = (target_logprobs - sampling_logprobs) ** 2
# Compute DRO objective
dro_objective = target_logprobs * advantages - 0.5 * beta * quadratic_term
# DRO loss is negative of objective
loss = -dro_objective.sum()
```

--------------------------------

### ConvertTensors Method

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Utility method to convert tensors for loss function inputs.

```APIDOC
#### `convert_tensors`
```
def convert_tensors(cls, data: Any) -> Any
```
Convert torch.Tensor and numpy arrays to TensorData in loss_fn_inputs during construction.
```

--------------------------------

### Python BaseModel for Load Weights Response

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the structure for LoadWeightsResponse, indicating the path to model weights and an optional type. This Pydantic BaseModel is used to return information about loaded model weights.

```python
class LoadWeightsResponse(BaseModel):
    path: Optional[str] = None
    """A tinker URI for model weights at a specific step"""

    type: Optional[Literal["load_weights"]] = None
```

--------------------------------

### Define SampleResponse for Sampling Results

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the SampleResponse class, containing the results of a sampling request. It includes computed log probabilities for prompt tokens and top-k log probabilities if requested.

```python
class SampleResponse(BaseModel)

#### `prompt_logprobs`
If prompt_logprobs was set to true in the request, logprobs are computed for every token in the prompt. The `prompt_logprobs` response contains a float32 value for every token in the prompt.
#### `topk_prompt_logprobs`
If topk_prompt_logprobs was set to a positive integer k in the request, the top-k logprobs are computed for every token in the prompt. The `topk_prompt_logprobs` response contains, for every token in the prompt, a list of up to k (token_id, logprob) tuples.
```

--------------------------------

### Python StrictBase for Image Asset Pointer Chunk

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the ImageAssetPointerChunk type using StrictBase. This class represents a pointer to an image asset, specifying its format, location, and optionally, the expected number of tokens. It includes a property to retrieve the length based on expected tokens.

```python
class ImageAssetPointerChunk(StrictBase):
    format: Literal["png", "jpeg"]
    """Image format"""

    location: str
    """Path or URL to the image asset"""

    expected_tokens: int | None = None
    """Expected number of tokens this image represents.
    This is only advisory: the tinker backend will compute the number of tokens
    from the image, and we can fail requests quickly if the tokens does not
    match expected_tokens."""

    type: Literal["image_asset_pointer"] = "image_asset_pointer"

    @property
    def length(self) -> int:
        if self.expected_tokens is None:
            raise ValueError("ImageAssetPointerChunk expected_tokens needs to be set in order to compute the length")
        return self.expected_tokens
```

--------------------------------

### Define ForwardBackwardOutput for Training Outputs

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the ForwardBackwardOutput class, representing the output of a forward and backward pass during training. It includes the loss function output type, loss function outputs, and training metrics.

```python
class ForwardBackwardOutput(BaseModel)

#### `loss_fn_output_type`
The class name of the loss function output records (e.g., 'TorchLossReturn', 'ArrayRecord').
#### `loss_fn_outputs`
Dictionary mapping field names to tensor data
#### `metrics`
Training metrics as key-value pairs
```

--------------------------------

### Save and Load Training State - Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Saves the complete training state, including model weights and optimizer state, allowing for full resumption of training. This is crucial for multi-step pipelines, hyperparameter adjustments, or recovery from interruptions. The `load_state` function restores the training from a saved checkpoint.

```python
# Save a checkpoint that you can resume from
resume_path = training_client.save_state(name="0010").result().path

# Load that checkpoint
training_client.load_state(resume_path)
```

--------------------------------

### List Checkpoints (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Lists available checkpoints for a given training run ID. This function returns a Future object that resolves to a response containing a list of checkpoints. It can retrieve both training and sampler checkpoints. Dependencies: `types` for type hinting, `ConcurrentFuture` for asynchronous operations.

```python
def list_checkpoints(
    training_run_id: types.ModelID
) -> ConcurrentFuture[types.CheckpointsListResponse]:
    # ... implementation details ...
    pass

# Example Usage:
# future = rest_client.list_checkpoints("run-id")
# response = future.result()
# for checkpoint in response.checkpoints:
#     if checkpoint.checkpoint_type == "training":
#         print(f"Training checkpoint: {checkpoint.checkpoint_id}")
#     elif checkpoint.checkpoint_type == "sampler":
#         print(f"Sampler checkpoint: {checkpoint.checkpoint_id}")
```

--------------------------------

### Asynchronous Checkpoint Unpublishing (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Provides an asynchronous interface for unpublishing a checkpoint from a Tinker path. This method is the asynchronous counterpart to `unpublish_checkpoint_from_tinker_path` and returns `None` upon successful completion.

```python
async def unpublish_checkpoint_from_tinker_path_async(
        tinker_path: str) -> None

Async version of unpublish_checkpoint_from_tinker_path.
```

--------------------------------

### Asynchronous Session Retrieval (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Provides an asynchronous method to retrieve session information using a session ID. This is the async equivalent of `get_session` and returns a `GetSessionResponse` object.

```python
async def get_session_async(session_id: str) -> types.GetSessionResponse

Async version of get_session.
```

--------------------------------

### Define Datum for Loss Function Inputs and Conversion

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the Datum class, which holds inputs for the loss function and includes a method for converting tensors. The `convert_tensors` method handles the conversion of torch.Tensor and numpy arrays to TensorData.

```python
class Datum(StrictBase)

#### `loss_fn_inputs`
Dictionary mapping field names to tensor data
#### `convert_tensors`
```
def convert_tensors(cls, data: Any) -> Any
```

Convert torch.Tensor and numpy arrays to TensorData in loss_fn_inputs during construction.
```

--------------------------------

### Cross Entropy Loss for Supervised Learning (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Implements the standard cross-entropy loss for supervised learning, optimizing the policy to maximize the log-probability of target tokens. It uses token-level loss weights and supports numpy.ndarray or torch.Tensor inputs, returning the same type.

```Python
elementwise_loss = -target_logprobs * weights
loss = elementwise_loss.sum()  # scalar
```

--------------------------------

### Importance Sampling Objective Calculation (Python)

Source: https://tinker-docs.thinkingmachines.ai/losses

Shows the implementation for calculating the importance sampling objective used in policy gradient methods for reinforcement learning. It computes the probability ratio between the learner and sampler policies.

```python
# Compute probability ratio
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
```

--------------------------------

### Proximal Policy Optimization (PPO) Loss

Source: https://tinker-docs.thinkingmachines.ai/llms

The PPO loss function, which uses a clipping objective to limit policy updates and prevent large changes in policy space, ensuring more stable training.

```APIDOC
## Proximal Policy Optimization (PPO) Loss

### Description
Implements the Proximal Policy Optimization (PPO) loss objective. PPO uses a clipping mechanism on the importance ratio to constrain policy updates, preventing excessively large changes and promoting training stability. The final PPO loss is the minimum of the clipped and unclipped objectives.

### Method
Not applicable (function within a library)

### Endpoint
Not applicable (function within a library)

### Parameters
#### Input Tensors
- **target_tokens** (array[(N,), int]) - Target token IDs (from the learner $p_\theta$).
- **logprobs** (array[(N,), float]) - `sampling_logprobs` for the tokens (from the sampler $q$).
- **advantages** (array[(N,), float]) - Advantage values for RL (positive to reinforce, negative to discourage).

### Request Example
```python
# This is a conceptual example, not a direct API call
# Assuming target_logprobs, sampling_logprobs, and advantages are available
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
# Clipping logic would follow here, using epsilon_low and epsilon_high (fixed at 0.2 in Tinker)
# clipped_objective = torch.clamp(prob_ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages
# loss = -torch.min(prob_ratio * advantages, clipped_objective).sum() # scalar
```

### Response
#### Output Tensors
- **logprobs** (array[(N,), float]) - `target_logprobs` for the tokens (from the learner $p_\theta$).

#### Output Diagnostics
- **loss:sum** (scalar) - Sum of PPO policy gradient losses $\mathcal L_{\text{PPO}}$.

#### Response Example
```json
{
  "logprobs": [0.15, 0.45, 0.35],
  "loss:sum": -0.850
}
```
```

--------------------------------

### Define SessionStartEvent for Telemetry

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the SessionStartEvent class, used for telemetry. It includes properties for the event type and log severity level. This is typically used to mark the beginning of a session for tracking purposes.

```python
class SessionStartEvent(BaseModel)

#### `event`
Telemetry event type
#### `severity`
Log severity level
```

--------------------------------

### NotFoundError - HTTP 404 Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Represents an HTTP 404 Not Found error. This exception is raised when the requested resource on the API could not be located.

```python
class NotFoundError(APIStatusError)

```

--------------------------------

### Overlapping Async Requests for Performance

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Shows an effective pattern for improving performance in Tinker by overlapping asynchronous requests. This involves submitting subsequent requests (e.g., forward_backward_async and optim_step_async) without waiting for the previous one to complete, maximizing utilization of clock cycles.

```python
# Submit forward_backward
fwd_bwd_future = await client.forward_backward_async(batch, loss_fn)

# Submit optim_step immediately (don't wait for forward_backward to finish)
optim_future = await client.optim_step_async(adam_params)

# Now retrieve results
fwd_bwd_result = await fwd_bwd_future
optim_result = await optim_future
```

--------------------------------

### Compute Importance-Weighted Loss (Python)

Source: https://tinker-docs.thinkingmachines.ai/losses

Calculates the importance-weighted loss for policy gradient methods. It takes target token IDs, sampling log probabilities, and advantage values as input. The output is the sum of the importance-weighted policy gradient losses.

```python
loss = -(prob_ratio * advantages).sum()
```

--------------------------------

### Unprocessable Entity Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

HTTP 422: Indicates the request was well-formed but contains semantic errors.

```APIDOC
## UnprocessableEntityError Objects

### Description
HTTP 422: The request was well-formed but contains semantic errors.

### Class Definition
```python
class UnprocessableEntityError(APIStatusError)
```
```

--------------------------------

### save_weights_for_sampler

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Saves model weights specifically for use with a SamplingClient. This is a synchronous operation.

```APIDOC
## save_weights_for_sampler

### Description
Save model weights for use with a SamplingClient.

### Method
POST

### Endpoint
/save_weights_for_sampler

### Parameters
#### Request Body
- **name** (string) - Required - A name for the saved sampler weights.

### Request Example
```json
{
  "name": "sampler-001"
}
```

### Response
#### Success Response (200)
- **APIFuture** (APIFuture[types.SaveWeightsForSamplerResponse]) - Contains the save response, including the sampler path.

#### Response Example
```json
{
  "future_id": "save_sampler_17181",
  "path": "tinker://runs/run-id/sampler_weights/sampler-001"
}
```
```

--------------------------------

### TinkerError Objects

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

The base exception for all Tinker-related errors.

```APIDOC
## TinkerError Objects

### Description
Base exception for all Tinker-related errors.

### Class Definition
```python
class TinkerError(Exception)
```
```

--------------------------------

### Direct Reward Optimization (DRO)

Source: https://tinker-docs.thinkingmachines.ai/llms

DRO is an off-policy reinforcement learning method that uses a quadratic penalty term to constrain policy updates. It requires a specific soft formulation of advantage estimation.

```APIDOC
## Direct Reward Optimization (DRO)

DRO is a general off-policy (and even offline) reinforcement learning method that uses a quadratic penalty term to constrain the policy update. Notice that this loss uses a dfferent (soft) formulation of the advantage estimation, which needs to be implemented on the client side.

**Mathematical Objective:**
$$ \mathcal{L}_{\text{DRO}}(\theta) = \mathbb{E}_{x \sim q}\left[\log p_\theta(x) \cdot A(x) - \frac{1}{2}\beta \left(\log \frac{p_\theta(x)}{q(x)}\right)^2\right] $$

**Implementation Details:**
```python
# Compute quadratic penalty term
quadratic_term = (target_logprobs - sampling_logprobs) ** 2
# Compute DRO objective
dro_objective = target_logprobs * advantages - 0.5 * beta * quadratic_term
# DRO loss is negative of objective
loss = -dro_objective.sum()
```

**Example with custom hyper-parameter:**
```python
fwd_bwd_result = await training_client.forward_backward_async(
    data=data,
    loss_fn="dro",
    loss_fn_config={"beta": 0.05}
)
```
```

--------------------------------

### Define OptimStepResponse Class in Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the OptimStepResponse class, a Pydantic model used to return metrics from an optimization step. It optionally contains a dictionary of metric names to their float values.

```python
class OptimStepResponse(BaseModel):
    metrics: Optional[Dict[str, float]] = None
    """Optimization step metrics as key-value pairs"""
```

--------------------------------

### Data Structures Reference

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

This section details the various data structures used throughout the Tinker API, including request parameters, response formats, and internal object representations.

```APIDOC
## Data Structures Reference

This section details the various data structures used throughout the Tinker API, including request parameters, response formats, and internal object representations.

### `AdamParams` Objects

Represents parameters for the Adam optimizer.

- **`learning_rate`** (float) - Learning rate for the optimizer.
- **`beta1`** (float) - Coefficient used for computing running averages of gradient.
- **`beta2`** (float) - Coefficient used for computing running averages of gradient square.
- **`eps`** (float) - Term added to the denominator to improve numerical stability.
- **`weight_decay`** (float) - Weight decay for the optimizer. Uses decoupled weight decay.
- **`grad_clip_norm`** (float) - Gradient clip norm for the optimizer. 0.0 means no clipping.

### `SupportedModel` Objects

Information about a model supported by the server.

- **`model_name`** (string) - The name of the supported model.

### `GetServerCapabilitiesResponse` Objects

Response containing the server's supported models and capabilities.

- **`supported_models`** (list[SupportedModel]) - List of models available on the server.

### `OptimStepResponse` Objects

Response from an optimization step.

- **`metrics`** (dict) - Optimization step metrics as key-value pairs.

### `ModelInput` Objects

Represents input data for a model, composed of chunks.

- **`chunks`** (list[ModelInputChunk]) - Sequence of input chunks.

#### Methods:
- **`from_ints(cls, tokens: List[int]) -> "ModelInput"`**: Create a `ModelInput` from a list of ints (tokens).
- **`to_ints() -> List[int]`**: Convert the `ModelInput` to a list of ints (tokens). Throws exception if there are any non-token chunks.
- **`length() -> int`**: Return the total context length used by this `ModelInput`.
- **`empty(cls) -> "ModelInput"`**: Create an empty `ModelInput`.
- **`append(chunk: ModelInputChunk) -> "ModelInput"`**: Add a new chunk, return a new `ModelInput`.
- **`append_int(token: int) -> "ModelInput"`**: Add a new token, return a new `ModelInput`.

### `WeightsInfoResponse` Objects

Minimal information for loading public checkpoints.

### `Checkpoint` Objects

Information about a model checkpoint.

- **`checkpoint_id`** (string) - The checkpoint ID.
- **`checkpoint_type`** (string) - The type of checkpoint (training or sampler).
- **`time`** (datetime) - The time when the checkpoint was created.
- **`tinker_path`** (string) - The tinker path to the checkpoint.
- **`size_bytes`** (integer) - The size of the checkpoint in bytes.
- **`public`** (boolean) - Whether the checkpoint is publicly accessible.

### `ParsedCheckpointTinkerPath` Objects

Represents a parsed Tinker path for a checkpoint.

- **`tinker_path`** (string) - The tinker path to the checkpoint.
- **`training_run_id`** (string) - The training run ID.
- **`checkpoint_type`** (string) - The type of checkpoint (training or sampler).
- **`checkpoint_id`** (string) - The checkpoint ID.

#### Methods:
- **`from_tinker_path(cls, tinker_path: str) -> "ParsedCheckpointTinkerPath"`**: Parse a tinker path to an instance of `ParsedCheckpointTinkerPath`.

### `CheckpointArchiveUrlResponse` Objects

Response containing a signed URL to download a checkpoint archive.

- **`url`** (string) - Signed URL to download the checkpoint archive.
- **`expires`** (integer) - Unix timestamp when the signed URL expires, if available.

### `SampledSequence` Objects

Represents a sequence of sampled tokens.

- **`stop_reason`** (string) - Reason why sampling stopped.
- **`tokens`** (list[int]) - List of generated token IDs.
- **`logprobs`** (list[float], optional) - Log probabilities for each token.

### `TryAgainResponse` Objects

Indicates that a request needs to be retried.

- **`request_id`** (string) - Request ID that is still pending.

### `LoadWeightsRequest` Objects

Request to load model weights.

- **`path`** (string) - A tinker URI for model weights at a specific step.
- **`optimizer`** (boolean) - Whether to load optimizer state along with model weights.

### `TelemetrySendRequest` Objects

Request to send telemetry data.

- **`platform`** (string) - Host platform name.
- **`sdk_version`** (string) - SDK version string.

### `ImageAssetPointerChunk` Objects

Represents a pointer to an image asset within input data.

- **`format`** (string) - Image format.
- **`location`** (string) - Path or URL to the image asset.
- **`expected_tokens`** (integer) - Expected number of tokens this image represents. This is only advisory: the tinker backend will compute the number of tokens from the image, and we can fail requests quickly if the tokens do not match `expected_tokens`.

### `CheckpointsListResponse` Objects

Response containing a list of available model checkpoints.

- **`checkpoints`** (list[Checkpoint]) - List of available model checkpoints for the model.
- **`cursor`** (string, optional) - Pagination cursor information (None for unpaginated responses).

### `GenericEvent` Objects

Represents a generic telemetry event.

- **`event`** (string) - Telemetry event type.
- **`event_name`** (string) - Low-cardinality event name.
- **`severity`** (string) - Log severity level.
- **`event_data`** (dict) - Arbitrary structured JSON payload.

### `EncodedTextChunk` Objects

Represents a chunk of text encoded as token IDs.

- **`tokens`** (list[int]) - Array of token IDs.

### `ForwardBackwardInput` Objects

Input data for a forward/backward pass.

- **`data`** (list) - Array of input data for the forward/backward pass.
- **`loss_fn`** (string) - Fully qualified function path for the loss function.

```

--------------------------------

### Python StrictBase for LoRA Configuration

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the LoraConfig type using StrictBase. This class specifies parameters for Low-Rank Adaptation (LoRA), including rank, an optional seed for reproducibility, and boolean flags to control training of unembedding, MLP, and attention layers.

```python
class LoraConfig(StrictBase):
    rank: int
    """LoRA rank (dimension of low-rank matrices)"""

    seed: Optional[int] = None
    """Seed used for initialization of LoRA weights.

    Useful if you need deterministic or reproducible initialization of weights.
    """

    train_unembed: bool = True
    """Whether to add lora to the unembedding layer"""

    train_mlp: bool = True
    """Whether to add loras to the MLP layers (including MoE layers)"""

    train_attn: bool = True
    """Whether to add loras to the attention layers"""
```

--------------------------------

### Append integer token to ModelInput

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

This method appends a single integer token to a ModelInput object. It is an instance method and returns a new ModelInput object with the token appended, leaving the original object unchanged.

```python
def append_int(token: int) -> "ModelInput"

```

--------------------------------

### Datum Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Represents a single data point, potentially with associated loss function inputs.

```APIDOC
## Datum Objects
```
class Datum(StrictBase)
```

#### `loss_fn_inputs`
Dictionary mapping field names to tensor data
```

--------------------------------

### Importance Sampling Loss for Reinforcement Learning (Python)

Source: https://tinker-docs.thinkingmachines.ai/llms

Calculates the importance sampling objective for reinforcement learning, correcting bias when the learner policy differs from the sampling policy. It uses target log probabilities, sampling log probabilities, and advantages to compute the weighted loss. The output includes target log probabilities and the sum of importance-weighted losses.

```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
loss = -(prob_ratio * advantages).sum()
```

--------------------------------

### Direct Reward Optimization (DRO)

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Details the Direct Reward Optimization (DRO) loss function, a general off-policy RL method that uses a quadratic penalty to constrain policy updates. It highlights the need for client-side advantage estimation and provides the mathematical formulation and code implementation.

```APIDOC
## Direct Reward Optimization: `dro`

### Description
DRO is a reinforcement learning method that employs a quadratic penalty to constrain policy updates. It's designed to be general and can be used off-policy or even offline. Note that this loss requires a specific (soft) formulation of advantage estimation implemented on the client side.

### Method
This is a conceptual description of the DRO loss calculation, not a direct API endpoint.

### Endpoint
N/A

### Parameters
N/A

### Request Example
```python
fwd_bwd_result = await training_client.forward_backward_async(
    data=data,
    loss_fn="dro",
    loss_fn_config={"beta": 0.05}
)
```

### Response
N/A
```

--------------------------------

### Conflict Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

HTTP 409: Indicates the request conflicts with the current state of the resource.

```APIDOC
## ConflictError Objects

### Description
HTTP 409: The request conflicts with the current state of the resource.

### Class Definition
```python
class ConflictError(APIStatusError)
```
```

--------------------------------

### save_state

Source: https://tinker-docs.thinkingmachines.ai/api-reference/trainingclient

Saves the current model weights to persistent storage. This is a synchronous operation.

```APIDOC
## save_state

### Description
Save model weights to persistent storage.

### Method
POST

### Endpoint
/save_state

### Parameters
#### Request Body
- **name** (string) - Required - A name for the saved checkpoint.

### Request Example
```json
{
  "name": "checkpoint-001"
}
```

### Response
#### Success Response (200)
- **APIFuture** (APIFuture[types.SaveWeightsResponse]) - Contains the save response, including the checkpoint path.

#### Response Example
```json
{
  "future_id": "save_state_11223",
  "path": "tinker://runs/run-id/weights/checkpoint-001"
}
```
```

--------------------------------

### Collect and Structure Sweep Experiment Results

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Reads metrics.jsonl files from multiple sweep experiments, extracts relevant metadata (learning rate, final loss), and structures them into a pandas DataFrame for analysis. It filters out incomplete experiments and ensures data integrity.

```python
from glob import glob
import pandas
import os
import json

data = []
for fname in sorted(glob(os.path.expanduser("/tmp/sft-lr-sweep/*/metrics.jsonl"))):
    df = pandas.read_json(fname, lines=True)
    # make sure the experiment is completed
    if len(df) == 0 or df["progress"].iloc[-1] < 0.98:
        continue
    config_fname = fname.replace("metrics.jsonl", "config.json")
    with open(config_fname, "rb") as f:
        metadata = json.load(f)
    data.append({
        "fname": fname,
        "learning_rate": metadata["learning_rate"],
        "final_loss": df["train_mean_nll"].iloc[-1].item()
    })

print(f"Read metrics for {len(data)} experiments")
```

--------------------------------

### Unpublish Checkpoint API

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Allows users to unpublish a checkpoint, making it private again. This operation can only be performed by the exact owner of the training run.

```APIDOC
## POST /unpublish_checkpoint_from_tinker_path

### Description
Unpublish a checkpoint referenced by a tinker path to make it private again. Only the exact owner of the training run can unpublish checkpoints. This reverses the effect of publishing a checkpoint.

### Method
POST

### Endpoint
/unpublish_checkpoint_from_tinker_path

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **tinker_path** (str) - Required - The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")

### Request Example
```json
{
  "tinker_path": "tinker://run-id/weights/0001"
}
```

### Response
#### Success Response (200)
- **None** - Indicates the checkpoint was successfully unpublished.

#### Response Example
(No specific response body for success, typically an empty success status or 204 No Content)

### Errors
- **HTTPException: 400** - If checkpoint identifier is invalid.
- **HTTPException: 404** - If checkpoint not found or user doesn't own the training run.
- **HTTPException: 409** - If checkpoint is already private.
- **HTTPException: 500** - If there's an error unpublishing the checkpoint.
```

--------------------------------

### ModelData Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Metadata about a model's architecture and configuration.

```APIDOC
## ModelData Objects
```
class ModelData(BaseModel)
```

Metadata about a model's architecture and configuration.
#### `arch`
The model architecture identifier.
#### `model_name`
The human-readable model name.
#### `tokenizer_id`
The identifier of the tokenizer used by this model.
```

--------------------------------

### Parse Tinker path to Checkpoint info

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

This class method parses a Tinker path string into a ParsedCheckpointTinkerPath object. It takes a string representing the Tinker path and returns an instance containing details about the checkpoint.

```python
def from_tinker_path(cls, tinker_path: str) -> "ParsedCheckpointTinkerPath"

```

--------------------------------

### Handling Future Objects in Sync Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Demonstrates how to work with Future objects returned by synchronous Tinker API calls. The '.result()' method is called on the Future object to block execution until the operation completes and return the result.

```python
future = client.forward_backward(data, loss_fn)
result = future.result() # Blocks until complete
```

--------------------------------

### APIError Objects

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Base class for all API-related errors, providing access to the response body.

```APIDOC
## APIError Objects

### Description
Base class for all API-related errors.

### Class Definition
```python
class APIError(TinkerError)
```

### Properties
* **body** (dict | str | None) - The API response body. This will be the decoded JSON if valid, the raw response if not valid JSON, or None if there was no response.
```

--------------------------------

### Clipped Importance Sampling Policy Optimization (CISPO)

Source: https://tinker-docs.thinkingmachines.ai/losses

Clipped Importance Sampling Policy Optimization (CISPO) is a policy gradient method that uses a clipped importance ratio to weight the policy gradient. It differs from PPO by clipping the ratio directly before applying it to the log probability.

```APIDOC
## POST /compute/importance_weighted_loss

### Description
Computes the importance-weighted policy gradient loss, specifically for Clipped Importance Sampling Policy Optimization (CISPO).

### Method
POST

### Endpoint
/compute/importance_weighted_loss

### Parameters
#### Query Parameters
- **loss_fn** (string) - Required - The loss function to use, should be 'cispo'.
- **loss_fn_config** (object) - Optional - Configuration for the loss function. Supports `clip_low_threshold` and `clip_high_threshold`.
  - **clip_low_threshold** (float) - Optional - The lower bound for clipping the probability ratio. Defaults to 1 - epsilon.
  - **clip_high_threshold** (float) - Optional - The upper bound for clipping the probability ratio. Defaults to 1 + epsilon.

#### Request Body
- **target_tokens** (array[int]) - Required - Target token IDs.
- **logprobs** (array[float]) - Required - Sampling log probabilities for the tokens.
- **advantages** (array[float]) - Required - Advantage values for Reinforcement Learning.
- **sampling_logprobs** (array[float]) - Required - Log probabilities from the sampling distribution.

### Request Example
```json
{
  "target_tokens": [101, 1996, 2744, 2219, 102],
  "logprobs": [-0.1, -0.5, -0.3, -0.7, -0.2],
  "advantages": [0.5, -0.1, 0.3, 0.2, 0.4],
  "sampling_logprobs": [-0.2, -0.4, -0.2, -0.6, -0.3]
}
```

### Response
#### Success Response (200)
- **loss** (object) - Contains the computed loss.
  - **sum** (float) - The sum of the clipped importance-weighted policy gradient losses.
- **logprobs** (array[float]) - Target log probabilities for the tokens.
- **diagnostics** (object) - Diagnostic information.
  - **loss.sum** (scalar) - Sum of CISPO losses.

#### Response Example
```json
{
  "loss": {
    "sum": 1.45
  },
  "logprobs": [-0.1, -0.5, -0.3, -0.7, -0.2],
  "diagnostics": {
    "loss.sum": 1.45
  }
}
```
```

--------------------------------

### Detecting Extension Property for RL Datum Creation - Python

Source: https://tinker-docs.thinkingmachines.ai/rl/sequence-extension

This Python function determines if a reinforcement learning trajectory can be converted into a single Datum or multiple Datums based on the 'extension property'. If successive observations extend the previous ones, a single Datum is returned; otherwise, multiple Datums are created, impacting compute efficiency.

```python
def trajectory_to_data(traj: Trajectory, traj_advantage: float) -> list[tinker.Datum]:
    """
    Return one or more Datum objects corresponding to the trajectory.
    If the sequence grows by appending, i.e., each successive observation contains
    the previous observation+action as a prefix, then we can return a single Datum.
    However, if we get a sequence that's not an extension of the previous sequence,
    then that results in a new Datum.
    """
    pass
```

--------------------------------

### Define CreateSamplingSessionResponse for Session ID

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the CreateSamplingSessionResponse class, which returns the generated sampling session ID upon successful creation of a sampling session.

```python
class CreateSamplingSessionResponse(BaseModel)

#### `sampling_session_id`
The generated sampling session ID
```

--------------------------------

### Convert TensorData to PyTorch Tensor

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

The `to_torch` method of the `TensorData` object facilitates the conversion of the tensor data into a PyTorch tensor. This is crucial for deep learning tasks that utilize the PyTorch framework.

```python
def to_torch() -> "torch.Tensor":
    """Convert TensorData to torch tensor."""
    # Implementation details would go here to convert flattened data and shape into a torch tensor.
```

--------------------------------

### AuthenticationError - HTTP 401 Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Represents an HTTP 401 Unauthorized error. This exception is raised when the API indicates that the authentication credentials provided were missing or invalid.

```python
class AuthenticationError(APIStatusError)

```

--------------------------------

### MessageCompleter Interface for Chat-like Interactions

Source: https://tinker-docs.thinkingmachines.ai/completers

The MessageCompleter interface handles structured messages, similar to chat APIs. It takes a list of messages and returns a single assistant message. This is useful for semantic evaluations and multi-agent environments where message-level understanding is required.

```python
class MessageCompleter:
    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:

```

--------------------------------

### Implement Custom SamplingClientEvaluator in Python

Source: https://tinker-docs.thinkingmachines.ai/evals

This snippet defines a custom `CustomEvaluator` class inheriting from `SamplingClientEvaluator`. It allows for custom evaluation logic by taking a dataset, a grading function, and model/renderer names during initialization. The `__call__` method iterates through the dataset, generates responses using a `SamplingClient`, grades them using the provided `grader_fn`, and calculates accuracy.

```python
from typing import Any, Callable

import tinker
from tinker import types

from tinker_cookbook import renderers
from tinker_cookbook.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer

class CustomEvaluator(SamplingClientEvaluator):
    """
    A toy SamplingClientEvaluator that runs a custom evaluation and returns its metrics.
    """

    def __init__(
        self,
        dataset: Any,
        grader_fn: Callable[[str, str], bool],
        model_name: str,
        renderer_name: str,
    ):
        """
        Initialize the CustomEvaluator.
        Args:
            config: Configuration object containing all evaluation parameters
        """
        self.dataset = dataset
        self.grader_fn = grader_fn

        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run custom evaluation on the given sampling client and return metrics.
        Args:
            sampling_client: The sampling client to evaluate
        Returns:
            Dictionary of metrics from inspect evaluation
        """

        metrics = {}

        num_examples = len(self.dataset)
        num_correct = 0

        sampling_params = types.SamplingParams(
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )

        for datum in self.dataset:
            model_input: types.ModelInput = self.renderer.build_generation_prompt(
                [renderers.Message(role="user", content=datum["input"])]
            )
            # Generate response
            r: types.SampleResponse = await sampling_client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=sampling_params
            )
            tokens: list[int] = r.sequences[0].tokens
            response: renderers.Message = self.renderer.parse_response(tokens)[0]
            if self.grader_fn(response["content"], datum["output"]):
                num_correct += 1

        metrics["accuracy"] = num_correct / num_examples
        return metrics

```

--------------------------------

### Serialize Image Bytes to Base64 String

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

The `serialize_data` method within the `ImageChunk` object handles the serialization of image data. It takes image data as bytes and converts it into a base64 encoded string, which is suitable for transmission or storage in JSON formats.

```python
def serialize_data(value: bytes) -> str:
    """Serialize bytes to base64 string for JSON."""
    # Implementation details would go here to handle bytes to base64 string conversion.
```

--------------------------------

### Python: CreateModelResponse Type Definition

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the `CreateModelResponse` class, inheriting from `BaseModel`. It contains a `model_id` and a `type` field, which is a literal string 'create_model'. This structure is likely used for responses when a model is created.

```python
class CreateModelResponse(BaseModel):
    model_id: ModelID

    type: Literal["create_model"] = "create_model"
```

--------------------------------

### Append chunk to ModelInput

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

This method appends a new ModelInputChunk to an existing ModelInput object. It is an instance method and returns a new ModelInput object with the chunk appended; it does not modify the original object.

```python
def append(chunk: ModelInputChunk) -> "ModelInput"

```

--------------------------------

### Python BaseModel for Model Data

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the ModelData type using Pydantic's BaseModel. It includes optional fields for architecture, model name, and tokenizer ID, used for identifying and describing models.

```python
class ModelData(BaseModel):
    arch: Optional[str] = None

    model_name: Optional[str] = None

    tokenizer_id: Optional[str] = None
```

--------------------------------

### Define ModelInput Class in Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the ModelInput class for handling sequences of input chunks, including methods for conversion to and from integer tokens, calculating length, creating empty instances, and appending chunks or tokens. This class is essential for preparing model inputs.

```python
class ModelInput(StrictBase):
    chunks: List[ModelInputChunk]
    """Sequence of input chunks (formerly TokenSequence)"""


    @classmethod
    def from_ints(cls, tokens: List[int]) -> "ModelInput":
        """
        Create a ModelInput from a list of ints (tokens).
        """
        return cls(chunks=[EncodedTextChunk(tokens=tokens)])

    def to_ints(self) -> List[int]:
        """
        Convert the ModelInput to a list of ints (tokens)
        Throws exception if there are any non-token chunks
        """
        if not all(isinstance(chunk, EncodedTextChunk) for chunk in self.chunks):
            raise ValueError(f"to_ints only supported for ModelInput with EncodedTextChunks, got {[type(chunk) for chunk in self.chunks]}")
        return [token for chunk in self.chunks for token in chunk.tokens]

    @property
    def length(self) -> int:
        """
        Return the total context length used by this ModelInput.
        """
        return sum(chunk.length for chunk in self.chunks)

    @classmethod
    def empty(cls) -> "ModelInput":
        """
        Create an empty ModelInput.
        """
        return cls(chunks=[])

    def append(self, chunk: ModelInputChunk) -> "ModelInput":
        """
        Add a new chunk, return a new ModelInput.
        """
        return ModelInput(chunks=self.chunks + [chunk])

    def append_int(self, token: int) -> "ModelInput":
        """
        Add a new token, return a new ModelInput.
        """
        return self.append(EncodedTextChunk(tokens=[token]))
```

--------------------------------

### Define ModelData for Model Metadata

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the ModelData class, which holds metadata about a model's architecture and configuration. It includes the architecture identifier, model name, and tokenizer ID.

```python
class ModelData(BaseModel)

Metadata about a model's architecture and configuration.
#### `arch`
The model architecture identifier.
#### `model_name`
The human-readable model name.
#### `tokenizer_id`
The identifier of the tokenizer used by this model.
```

--------------------------------

### DRO Loss Calculation in PyTorch

Source: https://tinker-docs.thinkingmachines.ai/llms

Calculates the Direct Reward Optimization (DRO) loss in PyTorch. This involves computing a quadratic penalty term based on the difference between target and sampling log-probabilities. The DRO objective is then formed by combining the target log-probabilities multiplied by advantages and the quadratic penalty term scaled by beta. The final loss is the negative sum of the DRO objective.

```python
quadratic_term = (target_logprobs - sampling_logprobs) ** 2
dro_objective = target_logprobs * advantages - 0.5 * beta * quadratic_term
loss = -dro_objective.sum()
```

--------------------------------

### Importance Sampling Loss (Reinforcement Learning)

Source: https://tinker-docs.thinkingmachines.ai/llms

A variant of the policy gradient objective that corrects for bias when the learner policy differs from the sampling policy using an importance sampling ratio.

```APIDOC
## Importance Sampling Loss (Reinforcement Learning)

### Description
Implements a variant of the policy gradient objective using importance sampling to correct for bias when the learner policy $p_\theta$ differs from the sampling policy $q$. The objective is $\mathcal L_{\text{IS}}(\theta) = \mathbb E_{x\sim q}[\frac{p_\theta(x)}{q(x)}A(x)]$.

### Method
Not applicable (function within a library)

### Endpoint
Not applicable (function within a library)

### Parameters
#### Input Tensors
- **target_tokens** (array[(N,), int]) - Target token IDs (from the sampler $q$).
- **logprobs** (array[(N,), float]) - `sampling_logprobs` for the tokens (from the sampler $q$).
- **advantages** (array[(N,), float]) - Advantage values for RL (positive to reinforce, negative to discourage).

### Request Example
```python
# This is a conceptual example, not a direct API call
# Assuming target_logprobs, sampling_logprobs, and advantages are available
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
loss = -(prob_ratio * advantages).sum() # scalar
```

### Response
#### Output Tensors
- **logprobs** (array[(N,), float]) - `target_logprobs` for the tokens (from the learner $p_\theta$).

#### Output Diagnostics
- **loss:sum** (scalar) - Sum of importance-weighted policy gradient losses $\mathcal L_{\text{IS}}$.

#### Response Example
```json
{
  "logprobs": [0.2, 0.4, 0.3],
  "loss:sum": -0.987
}
```
```

--------------------------------

### MessageCompleter Interface

Source: https://tinker-docs.thinkingmachines.ai/llms-full

The MessageCompleter operates at a higher level with structured messages, similarly to standard chat APIs. It takes a list of messages and returns a single assistant message response.

```APIDOC
## MessageCompleter Interface

### Description
The `MessageCompleter` operates at a higher level with structured messages, similar to standard chat APIs. It takes a list of messages and returns a single assistant message response.

### Method
`__call__`

### Endpoint
N/A (Interface Definition)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **messages** (list[renderers.Message]) - Required - A list of messages representing the conversation history.

### Request Example
```python
# This is a conceptual example as MessageCompleter is an interface.
# Actual implementation would involve instantiating a concrete class with a renderer.
async def generate_message(completer: MessageCompleter, conversation: list[renderers.Message]):
    return await completer(conversation)
```

### Response
#### Success Response (200)
- (renderers.Message) - A single assistant message response.

#### Response Example
```json
{
  "role": "assistant",
  "content": "This is the assistant's response."
}
```
```

--------------------------------

### Unpublish Checkpoint from Tinker Path (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Unpublishes a checkpoint referenced by a Tinker path, making it private again. Only the owner of the training run can unpublish. This operation reverses the effect of publishing a checkpoint. It raises HTTPExceptions for various error conditions such as invalid identifiers, not found, or ownership issues.

```python
def unpublish_checkpoint_from_tinker_path(
        tinker_path: str) -> ConcurrentFuture[None]

Unpublish a checkpoint referenced by a tinker path to make it private again.
Only the exact owner of the training run can unpublish checkpoints. This reverses the effect of publishing a checkpoint.
Args:
  * `tinker_path`: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")


Returns:
  * A `Future` that completes when the checkpoint is unpublished


Raises: HTTPException: 400 if checkpoint identifier is invalid HTTPException: 404 if checkpoint not found or user doesn't own the training run HTTPException: 409 if checkpoint is already private HTTPException: 500 if there's an error unpublishing the checkpoint
Example:
```
future = rest_client.unpublish_checkpoint_from_tinker_path("tinker://run-id/weights/0001")
future.result()  # Wait for completion
print("Checkpoint unpublished successfully")
```
```

--------------------------------

### Define SamplingParams Class in Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the SamplingParams class, a Pydantic model for configuring text generation parameters. It includes settings for maximum tokens, random seed, stop sequences, temperature, top-k, and top-p sampling.

```python
class SamplingParams(BaseModel):
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate"""

    seed: Optional[int] = None
    """Random seed for reproducible generation"""

    stop: Union[str, Sequence[str], Sequence[int], None] = None
    """Stop sequences for generation"""

    temperature: float = 1
    """Sampling temperature"""

    top_k: int = -1
    """Top-k sampling parameter (-1 for no limit)"""

    top_p: float = 1
    """Nucleus sampling probability"""
```

--------------------------------

### APIFuture Abstract Base Class

Source: https://tinker-docs.thinkingmachines.ai/api-reference/apifuture

APIFuture provides a unified interface for handling asynchronous operations that can be accessed both synchronously (via result()) and asynchronously (via await or result_async()).

```APIDOC
## APIFuture Abstract Base Class

### Description
Abstract base class for futures that can be awaited or accessed synchronously. Provides a unified interface for handling async operations in both sync and async contexts.

### Usage
Can be awaited directly in async contexts (`result = await api_future`) or accessed synchronously (`result = api_future.result()`).

### Methods

#### `result_async`
```async
async def result_async(timeout: float | None = None) -> T
```

Get the result asynchronously with an optional timeout.

**Parameters**
- `timeout` (float | None): Maximum time to wait in seconds. `None` means wait indefinitely.

**Returns**
- `T`: The result value.

**Raises**
- `TimeoutError`: If the timeout is exceeded.

#### `result`
```
def result(timeout: float | None = None) -> T
```

Get the result synchronously with an optional timeout.

**Parameters**
- `timeout` (float | None): Maximum time to wait in seconds. `None` means wait indefinitely.

**Returns**
- `T`: The result value.

**Raises**
- `TimeoutError`: If the timeout is exceeded.
- `Exception`: Any exception raised by the underlying operation.
```

--------------------------------

### Request Failed Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Raised when an asynchronous request completes in a failed state.

```APIDOC
## RequestFailedError Objects

### Description
Raised when an asynchronous request completes in a failed state.

### Class Definition
```python
class RequestFailedError(TinkerError)
```
```

--------------------------------

### Define SaveWeightsForSamplerResponse Class in Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the SaveWeightsForSamplerResponse class, a Pydantic model indicating the path to saved model weights specifically for sampling. It includes a 'path' attribute which is a tinker URI.

```python
class SaveWeightsForSamplerResponse(BaseModel):
    path: str
    """A tinker URI for model weights for sampling at a specific step"""

    type: Optional[Literal["save_weights_for_sampler"]] = None
```

--------------------------------

### AwaitableConcurrentFuture Implementation

Source: https://tinker-docs.thinkingmachines.ai/api-reference/apifuture

AwaitableConcurrentFuture is an implementation of APIFuture that wraps a concurrent.futures.Future, bridging Python's concurrent.futures with asyncio.

```APIDOC
## AwaitableConcurrentFuture Implementation

### Description
Implementation of APIFuture that wraps a `concurrent.futures.Future`. This class allows a standard Future to be used in async contexts and is commonly returned by Tinker API methods.

### Usage
Can be used synchronously via `result()` or asynchronously via `await` or `result_async()`.

### Methods

#### `result`
```
def result(timeout: float | None = None) -> T
```

Get the result synchronously with an optional timeout.

**Parameters**
- `timeout` (float | None): Maximum time to wait in seconds. `None` means wait indefinitely.

**Returns**
- `T`: The result value.

**Raises**
- `TimeoutError`: If timeout is exceeded.
- `Exception`: Any exception raised by the underlying operation.

**Example**
```python
future = rest_client.get_training_run("run-id")
result = future.result(timeout=30)  # Wait up to 30 seconds
```

#### `result_async`
```async
async def result_async(timeout: float | None = None) -> T
```

Async version of `result` with an optional timeout.

**Parameters**
- `timeout` (float | None): Maximum time to wait in seconds. `None` means wait indefinitely.

**Returns**
- `T`: The result value.

**Raises**
- `TimeoutError`: If the timeout is exceeded.

#### `future`
```
def future() -> ConcurrentFuture[T]
```

Get the underlying `concurrent.futures.Future` object.

**Returns**
- `ConcurrentFuture[T]`: The wrapped `ConcurrentFuture` object.

**Example**
```python
api_future = rest_client.get_training_run("run-id")
concurrent_future = api_future.future()
# Can now use standard concurrent.futures methods
if concurrent_future.done():
    result = concurrent_future.result()
```
```

--------------------------------

### API Connection Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Raised when a connection error occurs during an API request.

```APIDOC
## APIConnectionError Objects

### Description
Raised when a connection error occurs while making an API request.

### Class Definition
```python
class APIConnectionError(APIError)
```
```

--------------------------------

### AwaitableConcurrentFuture: Wrapping concurrent.futures.Future

Source: https://tinker-docs.thinkingmachines.ai/api-reference/apifuture

AwaitableConcurrentFuture implements the APIFuture interface by wrapping a standard concurrent.futures.Future. This allows traditional futures to be used seamlessly within asyncio applications, providing both synchronous result retrieval and asynchronous awaitability.

```python
from concurrent.futures import Future as ConcurrentFuture
from typing import TypeVar

# Assuming APIFuture is defined as above
T = TypeVar('T')

class AwaitableConcurrentFuture(APIFuture[T]):
    def __init__(self, future: ConcurrentFuture[T]):
        self._future = future

    async def result_async(self, timeout: float | None = None) -> T:
        # Implementation details to handle async waiting on concurrent.futures.Future
        ...

    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout=timeout)

    def future(self) -> ConcurrentFuture[T]:
        return self._future

# Example usage:
# concurrent_future = some_operation()
# api_future = AwaitableConcurrentFuture(concurrent_future)
# result = api_future.result()
```

--------------------------------

### API Status Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Raised for API responses with a 4xx or 5xx status code.

```APIDOC
## APIStatusError Objects

### Description
Raised when an API response has a status code of 4xx or 5xx.

### Class Definition
```python
class APIStatusError(APIError)
```
```

--------------------------------

### API Timeout Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Raised when an API request times out.

```APIDOC
## APITimeoutError Objects

### Description
Raised when an API request times out.

### Class Definition
```python
class APITimeoutError(APIConnectionError)
```
```

--------------------------------

### TensorData Class: Convert between TensorData, NumPy, and PyTorch

Source: https://tinker-docs.thinkingmachines.ai/llms-full

The TensorData class facilitates the representation and conversion of tensor data. It supports creation from NumPy arrays and PyTorch tensors, and conversion back to these formats. Dependencies include NumPy and optionally PyTorch.

```python
class TensorData(StrictBase):
    data: Union[List[int], List[float]]
    """Flattened tensor data as array of numbers."""

    dtype: TensorDtype

    shape: Optional[List[int]] = None
    """Optional.

    The shape of the tensor (see PyTorch tensor.shape). The shape of a
    one-dimensional list of length N is `(N,)`. Can usually be inferred if not
    provided, and is generally inferred as a 1D tensor.
    """

    @classmethod
    def from_numpy(cls, array: npt.NDArray[Any]) -> "TensorData":
        return cls(
            data=array.flatten().tolist(),
            dtype=_convert_numpy_dtype_to_tensor(array.dtype),
            shape=list(array.shape),
        )

    @classmethod
    def from_torch(cls, tensor: "torch.Tensor") -> "TensorData":
        return cls(
            data=tensor.flatten().tolist(),
            dtype=_convert_torch_dtype_to_tensor(tensor.dtype),
            shape=list(tensor.shape),
        )

    def to_numpy(self) -> npt.NDArray[Any]:
        """Convert TensorData to numpy array."""
        numpy_dtype = _convert_tensor_dtype_to_numpy(self.dtype)
        arr = np.array(self.data, dtype=numpy_dtype)
        if self.shape is not None:
            arr = arr.reshape(self.shape)
        return arr

    def to_torch(self) -> "torch.Tensor":
        """Convert TensorData to torch tensor."""
        if not _HAVE_TORCH:
            raise ImportError("PyTorch is not installed. Cannot convert to torch tensor.")

        torch_dtype = _convert_tensor_dtype_to_torch(self.dtype)
        tensor = torch.tensor(self.data, dtype=torch_dtype)
        if self.shape is not None:
            tensor = tensor.reshape(self.shape)
        return tensor

    def tolist(self) -> List[Any]:
        return self.to_numpy().tolist()
```

--------------------------------

### Unpublish Model Checkpoint using Tinker CLI

Source: https://tinker-docs.thinkingmachines.ai/publish-weights

Removes a previously published model checkpoint from the Tinker platform. Use this command with the checkpoint's Tinker path to make it private again. This is useful for managing shared resources or retracting checkpoints.

```bash
tinker checkpoint unpublish $TINKER_CHECKPOINT_PATH
```

--------------------------------

### Python: AdamParams Type Definition

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the `AdamParams` class, inheriting from `StrictBase`. This class represents parameters for the Adam optimizer, including learning rate, beta1, beta2, and epsilon. Default values are provided for each parameter.

```python
class AdamParams(StrictBase):
    learning_rate: float = 0.0001
    """Learning rate for the optimizer"""

    beta1: float = 0.9
    """Coefficient used for computing running averages of gradient"""

    beta2: float = 0.95
    """Coefficient used for computing running averages of gradient square"""

    eps: float = 1e-12
    """Term added to the denominator to improve numerical stability"""
```

--------------------------------

### Define SampledSequence Class in Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the SampledSequence class, a Pydantic model representing a single generated sequence. It includes the stop reason, the list of generated token IDs, and optional log probabilities for each token.

```python
class SampledSequence(BaseModel):
    stop_reason: StopReason
    """Reason why sampling stopped"""

    tokens: List[int]
    """List of generated token IDs"""

    logprobs: Optional[List[float]] = None
    """Log probabilities for each token (optional)"""
```

--------------------------------

### Python: ForwardBackwardInput Type Definition

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the `ForwardBackwardInput` class, inheriting from `StrictBase`. This class bundles data for a forward/backward pass, including a list of `Datum` objects, the `loss_fn` path, and optional `loss_fn_config` parameters.

```python
class ForwardBackwardInput(StrictBase):
    data: List[Datum]
    """Array of input data for the forward/backward pass"""

    loss_fn: LossFnType
    """Fully qualified function path for the loss function"""

    loss_fn_config: Optional[Dict[str, float]] = None
    """Optional configuration parameters for the loss function (e.g., PPO clip thresholds, DPO beta)"""
```

--------------------------------

### Delete Checkpoint Async (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

An asynchronous version of `delete_checkpoint`. This function allows for non-blocking deletion of checkpoints, returning `None` upon successful completion. It's designed for use in asynchronous contexts. Dependencies: `types` for type hinting.

```python
async def delete_checkpoint_async(training_run_id: types.ModelID,
                                  checkpoint_id: str) -> None:
    # ... implementation details ...
    pass
```

--------------------------------

### Define Logprob Squared Loss Function in Python

Source: https://tinker-docs.thinkingmachines.ai/llms

Defines a custom loss function that calculates the sum of squared log probabilities. This function takes a list of data and a list of log probabilities as input and returns the computed loss and associated metrics. It is designed to be used with Tinker's `forward_backward_custom` method.

```python
def logprob_squared_loss(data: list[Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    loss = (logprobs ** 2).sum()
    return loss, {"logprob_squared_loss": loss.item()}
```

--------------------------------

### Delete Checkpoint from Tinker Path Async (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

An asynchronous version of `delete_checkpoint_from_tinker_path`. This function enables non-blocking deletion of checkpoints using their Tinker paths, returning `None` upon success. Ideal for integration into asynchronous workflows. Dependencies: None explicitly shown, relies on underlying async capabilities.

```python
async def delete_checkpoint_from_tinker_path_async(tinker_path: str) -> None:
    # ... implementation details ...
    pass
```

--------------------------------

### SessionEndEvent Object

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Represents an event signaling the end of a session.

```APIDOC
## SessionEndEvent Objects
```
class SessionEndEvent(BaseModel)
```
```

--------------------------------

### Deserialize Base64 Image Data to Bytes

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

The `validate_data` method within the `ImageChunk` object is responsible for deserializing image data. It accepts either bytes or a base64 encoded string and returns the image data as bytes. This is useful for ensuring consistent data types before processing.

```python
def validate_data(cls, value: Union[bytes, str]) -> bytes:
    """Deserialize base64 string to bytes if needed."""
    # Implementation details would go here to handle bytes or base64 string conversion.
```

--------------------------------

### Python Type Alias for Loss Function Output

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines LossFnOutput as a type alias for a dictionary mapping strings to TensorData. This represents the expected output format from loss functions in the Tinker AI system.

```python
LossFnOutput: TypeAlias = Dict[str, TensorData]
```

--------------------------------

### PermissionDeniedError - HTTP 403 Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Represents an HTTP 403 Forbidden error. This exception is raised when the API denies access to a resource because the authenticated user lacks the necessary permissions.

```python
class PermissionDeniedError(APIStatusError)

```

--------------------------------

### API Response Validation Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Raised when the API response does not match the expected schema.

```APIDOC
## APIResponseValidationError Objects

### Description
Raised when API response doesn't match expected schema.

### Class Definition
```python
class APIResponseValidationError(APIError)
```
```

--------------------------------

### APIFuture Class: Asynchronous and Synchronous Result Access

Source: https://tinker-docs.thinkingmachines.ai/api-reference/apifuture

The APIFuture abstract base class allows operations to be awaited directly in async contexts or accessed synchronously via the result() method. It provides a unified interface for handling futures that can be used in both sync and async code.

```python
from abc import ABC
from typing import TypeVar, Generic

T = TypeVar('T')

class APIFuture(ABC, Generic[T]):
    async def result_async(self, timeout: float | None = None) -> T:
        """Get the result asynchronously with optional timeout."""
        ...

    def result(self, timeout: float | None = None) -> T:
        """Get the result synchronously with optional timeout."""
        ...

# Example usage in async context
# result = await api_future

# Example usage in sync context
# result = api_future.result()
```

--------------------------------

### ConflictError - HTTP 409 Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Represents an HTTP 409 Conflict error. This exception is raised when the request conflicts with the current state of the target resource on the API.

```python
class ConflictError(APIStatusError)

```

--------------------------------

### Delete Checkpoint API

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Deletes a specified checkpoint for a training run or via a Tinker path. Supports both synchronous and asynchronous operations.

```APIDOC
## DELETE CHECKPOINT

### Description
Deletes a checkpoint associated with a training run or identified by a Tinker path.

### Method
DELETE

### Endpoints
1. `/training_runs/{training_run_id}/checkpoints/{checkpoint_id}` (by training run and checkpoint ID)
2. `/checkpoints/tinker/{tinker_path}` (by Tinker path)

### Parameters
#### Path Parameters (Endpoint 1)
- **training_run_id** (string) - Required - The ID of the training run.
- **checkpoint_id** (string) - Required - The ID of the checkpoint to delete.

#### Path Parameters (Endpoint 2)
- **tinker_path** (string) - Required - The Tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001").

### Request Example
```python
# By training run and checkpoint ID (Synchronous)
future = rest_client.delete_checkpoint("run-id", "checkpoint-123")
future.result()

# By training run and checkpoint ID (Asynchronous)
await rest_client.delete_checkpoint_async("run-id", "checkpoint-123")

# By Tinker path (Synchronous)
future = rest_client.delete_checkpoint_from_tinker_path("tinker://run-id/weights/0001")
future.result()

# By Tinker path (Asynchronous)
await rest_client.delete_checkpoint_from_tinker_path_async("tinker://run-id/weights/0001")
```

### Response
#### Success Response (204)
No content is returned upon successful deletion.
```

--------------------------------

### RateLimitError - HTTP 429 Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Represents an HTTP 429 Too Many Requests error. This exception is raised when the client has exceeded the API's rate limits.

```python
class RateLimitError(APIStatusError)

```

--------------------------------

### Python: Datum Type Definition

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the `Datum` class, inheriting from `StrictBase`. This class represents a single data point for a forward/backward pass, containing `loss_fn_inputs` and `model_input`. It includes a model validator to convert tensors and arrays to `TensorData`.

```python
class Datum(StrictBase):
    loss_fn_inputs: LossFnInputs
    """Dictionary mapping field names to tensor data"""

    model_input: ModelInput

    @model_validator(mode="before")
    @classmethod
    def convert_tensors(cls, data: Any) -> Any:
        """Convert torch.Tensor and numpy arrays to TensorData in loss_fn_inputs during construction."""
        if isinstance(data, dict) and "loss_fn_inputs" in data:
            loss_fn_inputs = data["loss_fn_inputs"]
            if isinstance(loss_fn_inputs, dict):
                converted_inputs = {}
                for key, value in loss_fn_inputs.items():
                    converted_inputs[key] = cls._maybe_convert_array(key, value)
                data = dict(data)  # Make a copy
                data["loss_fn_inputs"] = converted_inputs
        return data

    @classmethod
    def _maybe_convert_array(cls, key: str, value: Any) -> Any:
        """Convert torch.Tensor, numpy array, or 1-D list to TensorData if needed."""
        if _HAVE_TORCH and isinstance(value, torch.Tensor):
            return TensorData.from_torch(value)
        elif isinstance(value, np.ndarray):
            return TensorData.from_numpy(value)
        elif isinstance(value, list):
            # assume it's 1d and infer the dtype from the key
            return TensorData(data=value, dtype=_key_to_type[key], shape=[len(value)])
        else:
            return value


_key_to_type = {
    "target_tokens": "int64",
    "weights": "float32",
    "advantages": "float32",
    "logprobs": "float32",
    "clip_low_threshold": "float32",
    "clip_high_threshold": "float32",
}
```

--------------------------------

### BadRequestError - HTTP 400 Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Represents an HTTP 400 Bad Request error. This exception is raised when the API determines that the request made by the client was invalid or malformed.

```python
class BadRequestError(APIStatusError)

```

--------------------------------

### InternalServerError - HTTP 500+ Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Represents an HTTP 5xx Internal Server Error. This exception is raised when an error occurs on the API server side, preventing the request from being fulfilled.

```python
class InternalServerError(APIStatusError)

```

--------------------------------

### Python Type Alias for Loss Function Inputs

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines LossFnInputs as a type alias for a dictionary mapping strings to TensorData. This represents the expected input format for loss functions within the Tinker AI system.

```python
LossFnInputs: TypeAlias = Dict[str, TensorData]
```

--------------------------------

### APIConnectionError Hierarchy - API Connection Issues

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Raised when a connection-related error occurs during an API request. This includes network issues, dropped connections, or problems establishing communication with the API endpoint.

```python
class APIConnectionError(APIError)

```

--------------------------------

### UnprocessableEntityError - HTTP 422 Error

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Represents an HTTP 422 Unprocessable Entity error. This exception is raised when the API request was well-formed but contained semantic errors that prevented its processing.

```python
class UnprocessableEntityError(APIStatusError)

```

--------------------------------

### Delete Checkpoint (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Deletes a specified checkpoint for a given training run ID. This operation is synchronous and returns a Future that completes when the deletion is done. It does not return any specific data upon successful completion. Dependencies: `types` for type hinting, `ConcurrentFuture` for asynchronous operations.

```python
def delete_checkpoint(training_run_id: types.ModelID,
                      checkpoint_id: str) -> ConcurrentFuture[None]:
    # ... implementation details ...
    pass
```

--------------------------------

### Define SessionEndEvent for Session Termination

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

Defines the SessionEndEvent class, used to signify the end of a session. This event is typically logged for tracking and resource management purposes.

```python
class SessionEndEvent(BaseModel)

```

--------------------------------

### Define SupportedModel Class in Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the SupportedModel class, a Pydantic model representing a model that is supported by the system. It contains an optional 'model_name' field.

```python
class SupportedModel(BaseModel):
    model_name: Optional[str] = None
```

--------------------------------

### APITimeoutError - API Request Timeout

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Raised when an API request exceeds the configured time limit and times out. This indicates that the API did not respond within the expected timeframe, potentially due to network latency or server load.

```python
class APITimeoutError(APIConnectionError)

```

--------------------------------

### APIError Hierarchy - Base API Exception

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Base class for all API-related errors within the Tinker library. This exception is the parent for errors directly stemming from API interactions, including issues with responses and connections.

```python
class APIError(TinkerError)

```

--------------------------------

### APIStatusError Hierarchy - HTTP Status Code Errors

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Raised when an API response returns an HTTP status code indicating a client-side (4xx) or server-side (5xx) error. This is a general exception for HTTP errors, with more specific exceptions inheriting from it.

```python
class APIStatusError(APIError)

```

--------------------------------

### UnloadModelResponse Pydantic Model

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the Pydantic model for the UnloadModelResponse. It includes a model_id and an optional type field, which defaults to 'unload_model'. This model is used for response structures related to unloading models.

```python
class UnloadModelResponse(BaseModel):
    model_id: ModelID

    type: Optional[Literal["unload_model"]] = None
```

--------------------------------

### TinkerError Base Exception

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

The base exception class for all errors originating from the Tinker library. It serves as a common ancestor for all other Tinker-specific exceptions, facilitating centralized error handling.

```python
class TinkerError(Exception)

```

--------------------------------

### Convert TensorData to NumPy Array

Source: https://tinker-docs.thinkingmachines.ai/api-reference/types

The `to_numpy` method of the `TensorData` object allows for the conversion of flattened tensor data into a NumPy array. This is essential for performing numerical computations and analysis using the NumPy library.

```python
def to_numpy() -> npt.NDArray[Any]:
    """Convert TensorData to numpy array."""
    # Implementation details would go here to convert flattened data and shape into a numpy array.
```

--------------------------------

### RequestFailedError - Asynchronous Request Failure

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Raised when an asynchronous request initiated by Tinker completes in a failed state. This indicates that the background operation did not succeed.

```python
class RequestFailedError(TinkerError)

```

--------------------------------

### Python Type Alias for Loss Function Type

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines LossFnType as a type alias using Literal for a string that can be one of several specified loss function names. This enforces valid loss function types.

```python
LossFnType: TypeAlias = Literal["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"]
```

--------------------------------

### Define ModelInputChunk Type Alias in Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines a type alias for ModelInputChunk, which can be one of several types including EncodedTextChunk, ImageAssetPointerChunk, or ImageChunk. This provides flexibility in representing different types of input chunks for models.

```python
ModelInputChunk: TypeAlias = Annotated[
    Union[EncodedTextChunk, ImageAssetPointerChunk, ImageChunk], PropertyInfo(discriminator="type")
]
```

--------------------------------

### Delete Checkpoint from Tinker Path (Python)

Source: https://tinker-docs.thinkingmachines.ai/api-reference/restclient

Deletes a checkpoint identified by its Tinker path. This function provides an alternative way to delete checkpoints without needing the training run ID and checkpoint ID separately. It returns a Future that completes upon successful deletion. Dependencies: `ConcurrentFuture` for asynchronous operations.

```python
def delete_checkpoint_from_tinker_path(
        tinker_path: str) -> ConcurrentFuture[None]:
    # ... implementation details ...
    pass
```

--------------------------------

### Python Type Alias for Model ID

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines ModelID as a simple type alias for a string. This convention is used throughout the Tinker AI project to represent unique identifiers for models.

```python
ModelID: TypeAlias = str
```

--------------------------------

### Define StopReason Type Alias in Python

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines a type alias for StopReason, which can be either 'length' (indicating generation stopped due to reaching max tokens) or 'stop' (indicating generation stopped due to a stop sequence).

```python
StopReason: TypeAlias = Literal["length", "stop"]
```

--------------------------------

### TensorDtype Type Alias Definition

Source: https://tinker-docs.thinkingmachines.ai/llms-full

Defines the TensorDtype type alias, specifying the allowed string literals for tensor data types as 'int64' and 'float32'. This ensures type safety for tensor data types.

```python
TensorDtype: TypeAlias = Literal["int64", "float32"]
```

--------------------------------

### APIResponseValidationError - API Exception

Source: https://tinker-docs.thinkingmachines.ai/api-reference/exceptions

Raised when an API response does not conform to the expected data schema. This indicates a mismatch between the structure of the data received from the API and the structure anticipated by the client.

```python
class APIResponseValidationError(APIError)

```

=== COMPLETE CONTENT === This response contains all available snippets from this library. No additional content exists. Do not make further requests.