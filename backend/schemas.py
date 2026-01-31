from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class MessageResponse(BaseModel):
    message: str


class ChatRequest(BaseModel):
    prompt: str
    model_id: Optional[int] = None
    run_id: Optional[int] = None
    base_model: Optional[str] = None  # Allow specifying base model directly
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    response: str
    model: str
    usage: Usage


class UserRead(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    api_key: Optional[str] = None

    class Config:
        from_attributes = True


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectCreate(ProjectBase):
    pass


class ProjectRead(ProjectBase):
    id: int
    owner_id: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class RunConfig(BaseModel):
    base_model: Optional[str] = Field(
        None, description="Base model identifier, e.g. meta-llama/Llama-3.1-8B-Instruct"
    )
    renderer_name: Optional[str] = Field(
        None, description="Renderer/chat template to use"
    )
    dataset: Optional[Union[dict[str, Any], str]] = Field(
        default=None,
        description="Dataset specification (type, name, parameters) or dataset name string",
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Training hyperparameters (learning_rate, lora_rank, epochs, etc.)",
    )
    notes: Optional[str] = None
    # RL-specific configuration
    reward_model_path: Optional[str] = Field(
        None, description="Path to reward model for RL training"
    )
    reference_model_path: Optional[str] = Field(
        None, description="Path to reference model for RL training"
    )
    # Multi-agent configuration
    num_agents: Optional[int] = Field(
        1, description="Number of agents for multi-agent training"
    )
    # Evaluation configuration
    eval_datasets: Optional[list[str]] = Field(
        default_factory=list, description="Datasets to evaluate on"
    )

    @field_validator("dataset", mode="before")
    @classmethod
    def convert_dataset_string_to_dict(cls, v):
        """Convert legacy string dataset names to dict format for backward compatibility."""
        if isinstance(v, str):
            return {"name": v, "type": "builtin"}
        return v


class RunBase(BaseModel):
    project_id: int
    dataset_id: Optional[int] = None
    recipe_type: Literal[
        "SFT",  # Supervised Fine-Tuning
        "DPO",  # Direct Preference Optimization
        "DISTILLATION",  # Knowledge Distillation
        "RL",  # Reinforcement Learning
        "PPO",  # Proximal Policy Optimization
        "GRPO",  # Group Relative Policy Optimization
        "PROMPT_DISTILLATION",  # Prompt Distillation
        "MATH_RL",  # Math Reasoning RL
        "TOOL_USE",  # Tool Use Training
        "PREFERENCE",  # Preference Learning (RLHF)
        "MULTIPLAYER_RL",  # Multi-Agent RL
        "CHAT_SL",  # Chat Supervised Learning
        "EVAL",  # Evaluation
        "SAMPLE",  # Sampling
    ]
    config_json: RunConfig


class RunCreate(RunBase):
    pass


class RunRead(BaseModel):
    id: int
    project_id: int
    dataset_id: Optional[int]
    recipe_type: str
    status: str
    progress: Optional[float] = None
    config_json: RunConfig
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    log_path: Optional[str]

    class Config:
        from_attributes = True


class RunListResponse(BaseModel):
    runs: list[RunRead]


class RunDetailResponse(RunRead):
    logs_tail: Optional[str] = None
    metrics: Optional[list[dict[str, Any]]] = None
    latest_metrics: Optional[dict[str, Any]] = None
    checkpoints: list["CheckpointRead"] = Field(default_factory=list)


class RunCancelResponse(BaseModel):
    run_id: int
    status: str


class CheckpointBase(BaseModel):
    kind: str
    step: int
    meta: Optional[dict[str, Any]] = None


class CheckpointCreate(CheckpointBase):
    run_id: int
    tinker_path: str


class CheckpointRead(CheckpointBase):
    id: int
    run_id: int
    tinker_path: str

    class Config:
        from_attributes = True


class EvaluationBase(BaseModel):
    evaluator_name: str
    metrics: dict[str, Any]


class EvaluationRead(EvaluationBase):
    id: int
    run_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ModelRegistration(BaseModel):
    name: str
    base_model: str
    tinker_path: Optional[str] = None
    description: Optional[str] = None
    meta: Optional[dict[str, Any]] = Field(default_factory=dict)
    project_id: Optional[int] = None
    run_id: Optional[int] = None

    class Config:
        from_attributes = True


class ModelRead(BaseModel):
    id: int
    name: str
    base_model: str
    tinker_path: Optional[str] = None
    description: Optional[str] = None
    meta: Optional[dict[str, Any]] = Field(default_factory=dict)
    project_id: Optional[int] = None
    run_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class RealtimeMetricsResponse(BaseModel):
    run_id: int
    status: str
    progress: float
    current_step: str
    current_loss: str
    current_lr: str
    tokens_processed: str
    timestamp: str


class EvaluationRequest(BaseModel):
    evaluation_type: str = Field(default="pig_latin_translation")


class EvaluationResponse(BaseModel):
    run_id: int
    evaluation_type: str
    timestamp: str
    results: dict[str, Any]


class SampleParams(BaseModel):
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[str] = Field(default_factory=list)


class SampleRequest(BaseModel):
    model_id: Optional[int] = Field(None, description="Registered model identifier")
    model_path: Optional[str] = Field(None, description="Direct tinker:// path")
    prompt: str
    sampling_params: SampleParams = Field(default_factory=SampleParams)

    def resolved_model(self) -> str:
        if self.model_path:
            return self.model_path
        if self.model_id is not None:
            return f"registered:{self.model_id}"
        raise ValueError("model_id or model_path must be provided")


class SampleSequence(BaseModel):
    text: str
    tokens: list[int]
    logprob: Optional[float] = None


class SampleResponse(BaseModel):
    model: str
    prompt: str
    sequences: list[SampleSequence]
    sampling_params: SampleParams


class LogTailResponse(BaseModel):
    run_id: int
    tail: str
    total_bytes: int


class MetricsResponse(BaseModel):
    run_id: int
    metrics: list[dict[str, Any]]


class DatasetRegistration(BaseModel):
    name: str
    kind: Literal["huggingface", "local", "jsonl"]
    spec: dict[str, Any]
    description: Optional[str] = None


class DatasetRead(DatasetRegistration):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


RunDetailResponse.update_forward_refs()


class SupportedModel(BaseModel):
    model_name: str
    description: Optional[str] = None
    parameters: Optional[str] = None
    context_length: Optional[int] = None


class ModelCatalogResponse(BaseModel):
    supported_models: list[SupportedModel]
    registered_models: list[ModelRead]


class HyperparamRequest(BaseModel):
    model_name: str
    recipe_type: str = "sft"
    lora_rank: Optional[int] = None


# HuggingFace Integration Schemas

class HFTokenSaveRequest(BaseModel):
    token: str


class HFTokenStatusResponse(BaseModel):
    connected: bool
    username: Optional[str] = None
    last_verified: Optional[datetime] = None


class DeployToHFRequest(BaseModel):
    repo_name: str
    private: bool = False
    merge_weights: bool = True
    create_inference_endpoint: bool = False


class DeployToHFResponse(BaseModel):
    success: bool
    repo_url: str
    inference_url: Optional[str] = None
    message: str
    deployment_id: Optional[int] = None


class DeploymentRead(BaseModel):
    id: int
    checkpoint_id: int
    user_id: int
    hf_repo_name: str
    hf_repo_url: str
    hf_model_id: str
    is_private: bool
    merged_weights: bool
    status: str
    deployed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True
