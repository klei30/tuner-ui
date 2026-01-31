from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import utility functions
from utils.text_utils import strip_ansi_codes


# Import Tinker for inference
try:
    import tinker
    from tinker import types
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, inspect, text
from sqlalchemy.orm import Session

# Import local modules (with fallback for direct execution)
try:
    from .config import settings, setup_logging
    from .database import Base, SessionLocal, engine, get_db
    from .job_runner import JobRunner
    from .models import (
        Checkpoint,
        Dataset,
        Deployment,
        Evaluation,
        ModelRegistry,
        Project,
        Run,
        User,
    )
    from .schemas import (
        ChatRequest,
        ChatResponse,
        Usage,
        CheckpointRead,
        DatasetRead,
        DatasetRegistration,
        DeploymentRead,
        DeployToHFRequest,
        DeployToHFResponse,
        EvaluationRead,
        EvaluationRequest,
        EvaluationResponse,
        HFTokenSaveRequest,
        HFTokenStatusResponse,
        HyperparamRequest,
        LogTailResponse,
        MessageResponse,
        ModelCatalogResponse,
        ModelRead,
        ModelRegistration,
        ProjectCreate,
        ProjectRead,
        RealtimeMetricsResponse,
        RunCancelResponse,
        RunConfig,
        RunCreate,
        RunDetailResponse,
        RunListResponse,
        RunRead,
        SampleRequest,
        SampleResponse,
        SampleSequence,
        SupportedModel,
        MetricsResponse,
        UserRead,
    )
    from .utils.file_utils import (
        ArtifactPathResolver,
        read_file_tail,
        ensure_directory_exists,
    )
    from .utils.json_utils import read_jsonl_file, parse_json_with_nan
except ImportError:
    from config import settings, setup_logging
    from database import Base, SessionLocal, engine, get_db
    from job_runner import JobRunner
    from models import (
        Checkpoint,
        Dataset,
        Deployment,
        Evaluation,
        ModelRegistry,
        Project,
        Run,
        User,
    )
    from schemas import (
        ChatRequest,
        ChatResponse,
        Usage,
        CheckpointRead,
        DatasetRead,
        DatasetRegistration,
        DeploymentRead,
        DeployToHFRequest,
        DeployToHFResponse,
        EvaluationRead,
        EvaluationRequest,
        EvaluationResponse,
        HFTokenSaveRequest,
        HFTokenStatusResponse,
        HyperparamRequest,
        LogTailResponse,
        MessageResponse,
        ModelCatalogResponse,
        ModelRead,
        ModelRegistration,
        ProjectCreate,
        ProjectRead,
        RealtimeMetricsResponse,
        RunCancelResponse,
        RunConfig,
        RunCreate,
        RunDetailResponse,
        RunListResponse,
        RunRead,
        SampleRequest,
        SampleResponse,
        SampleSequence,
        SupportedModel,
        MetricsResponse,
        UserRead,
    )
    from utils.file_utils import (
        ArtifactPathResolver,
        read_file_tail,
        ensure_directory_exists,
    )
    from utils.json_utils import read_jsonl_file, parse_json_with_nan

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Check Tinker API availability
TINKER_AVAILABLE = importlib.util.find_spec("tinker") is not None
if TINKER_AVAILABLE:
    print("Tinker API available - using real training")
    try:
        from tinker_cookbook.completers import TinkerMessageCompleter
        from tinker_cookbook.renderers import get_renderer
    except ImportError:
        TINKER_AVAILABLE = False
        print("Tinker cookbook not available")
else:
    print("Warning: Tinker API not available. Using simulation mode.")

# Import chat inference helper
try:
    from chat_inference import resolve_model_path, ChatInferenceClient

    CHAT_INFERENCE_AVAILABLE = True
    print("chat_inference module loaded successfully")
except ImportError as e:
    CHAT_INFERENCE_AVAILABLE = False
    print(f"chat_inference module not available: {e}")

# SECURITY: No default API key - must be set via environment variable
API_KEY = os.getenv("TINKER_API_KEY")
if not API_KEY and not os.getenv("ALLOW_ANON", "").lower() in {"1", "true", "yes"}:
    logger.error("CRITICAL: TINKER_API_KEY environment variable is not set!")
    logger.error(
        "Please set TINKER_API_KEY in your .env file or set ALLOW_ANON=true for testing"
    )
    # Don't exit - let the app start but auth will fail properly
ALLOW_ANONYMOUS = os.getenv("ALLOW_ANON", "true").lower() in {"1", "true", "yes"}

SUPPORTED_MODELS: list[SupportedModel] = [
    SupportedModel(
        model_name="deepseek-ai/DeepSeek-V3.1",
        description="DeepSeek V3.1 - Large language model",
        parameters="Unknown",
        context_length=32768,
    ),
    SupportedModel(
        model_name="deepseek-ai/DeepSeek-V3.1-Base",
        description="DeepSeek V3.1 Base - Foundation model",
        parameters="Unknown",
        context_length=32768,
    ),
    SupportedModel(
        model_name="meta-llama/Llama-3.1-70B",
        description="Llama 3.1 70B - Large model for complex tasks",
        parameters="70B",
        context_length=131072,
    ),
    SupportedModel(
        model_name="meta-llama/Llama-3.1-8B",
        description="Llama 3.1 8B - Versatile base model",
        parameters="8B",
        context_length=131072,
    ),
    SupportedModel(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        description="Llama 3.1 8B Instruct - Optimized for chat",
        parameters="8B",
        context_length=131072,
    ),
    SupportedModel(
        model_name="meta-llama/Llama-3.2-1B",
        description="Llama 3.2 1B - Compact efficient model",
        parameters="1B",
        context_length=8192,
    ),
    SupportedModel(
        model_name="meta-llama/Llama-3.2-3B",
        description="Llama 3.2 3B - Balanced performance model",
        parameters="3B",
        context_length=8192,
    ),
    SupportedModel(
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        description="Llama 3.3 70B Instruct - Latest large chat model",
        parameters="70B",
        context_length=131072,
    ),
    SupportedModel(
        model_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
        description="Qwen3 235B - Massive instruction-tuned model",
        parameters="235B",
        context_length=32768,
    ),
    SupportedModel(
        model_name="Qwen/Qwen3-30B-A3B",
        description="Qwen3 30B-A3B - High performance model",
        parameters="30B",
        context_length=32768,
    ),
    SupportedModel(
        model_name="Qwen/Qwen3-30B-A3B-Base",
        description="Qwen3 30B-A3B Base - Foundation model",
        parameters="30B",
        context_length=32768,
    ),
    SupportedModel(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        description="Qwen3 30B-A3B Instruct - Chat optimized",
        parameters="30B",
        context_length=32768,
    ),
    SupportedModel(
        model_name="Qwen/Qwen3-32B",
        description="Qwen3 32B - Large versatile model",
        parameters="32B",
        context_length=32768,
    ),
    SupportedModel(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        description="Qwen3 4B Instruct - Efficient chat model",
        parameters="4B",
        context_length=32768,
    ),
    SupportedModel(
        model_name="Qwen/Qwen3-8B",
        description="Qwen3 8B - Standard performance model",
        parameters="8B",
        context_length=32768,
    ),
    SupportedModel(
        model_name="Qwen/Qwen3-8B-Base",
        description="Qwen3 8B Base - Foundation model",
        parameters="8B",
        context_length=32768,
    ),
    SupportedModel(
        model_name="openai/gpt-oss-120b",
        description="GPT OSS 120B - Open source GPT",
        parameters="120B",
        context_length=8192,
    ),
    SupportedModel(
        model_name="openai/gpt-oss-20b",
        description="GPT OSS 20B - Smaller open source GPT",
        parameters="20B",
        context_length=8192,
    ),
]

# Supported recipe types
SUPPORTED_RECIPES = [
    "SFT",  # Supervised Fine-Tuning
    "DPO",  # Direct Preference Optimization
    "RL",  # Reinforcement Learning
    "PPO",  # Proximal Policy Optimization
    "GRPO",  # Group Relative Policy Optimization
    "DISTILLATION",  # Model Distillation
    "CHAT_SL",  # Chat Supervised Learning
    "PREFERENCE",  # Preference Learning
    "TOOL_USE",  # Tool Use RL
    "MULTIPLAYER_RL",  # Multi-Agent RL
    "PROMPT_DISTILLATION",  # Prompt Distillation
    "MATH_RL",  # Math Reasoning RL
    "EVAL",  # Evaluation
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the application"""
    # Create database tables
    Base.metadata.create_all(bind=engine)

    import json

    with open("models.json", "r") as f:
        models_data = json.load(f)
    session = SessionLocal()
    try:
        for model in models_data:
            existing = (
                session.query(ModelRegistry).filter_by(name=model["name"]).first()
            )
            if not existing:
                entry = ModelRegistry(**model)
                session.add(entry)
        session.commit()
        print("Loaded models from models.json")
    finally:
        session.close()

    session = SessionLocal()
    try:
        if not session.query(Dataset).first():
            project = session.query(Project).first()
            if not project:
                project = Project(
                    name="Sample Project",
                    description="A sample project for testing",
                )
                session.add(project)
                session.commit()

            # Cookbook dataset: Pig Latin examples
            examples = [
                {"input": "banana split", "output": "anana-bay plit-say"},
                {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
                {"input": "donut shop", "output": "onut-day op-shay"},
                {"input": "pickle jar", "output": "ickle-pay ar-jay"},
                {"input": "space exploration", "output": "ace-spay exploration-way"},
                {"input": "rubber duck", "output": "ubber-ray uck-day"},
                {"input": "coding wizard", "output": "oding-cay izard-way"},
            ]
            # Create a jsonl file
            import json

            with open("pig_latin.jsonl", "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            dataset = Dataset(
                name="pig_latin",
                kind="jsonl",
                spec={"path": "pig_latin.jsonl"},
                description="Pig Latin dataset from Tinker Cookbook",
            )
            session.add(dataset)
            session.commit()

            run = Run(
                project_id=project.id,
                dataset_id=dataset.id,
                recipe_type="SFT",
                status="pending",
                config_json={
                    "base_model": "meta-llama/Llama-3.2-1B",
                    "max_steps": 10,
                    "dataset": "pig_latin",
                    "learning_rate": 1e-4,
                },
            )
            session.add(run)
            session.commit()

            if not session.query(ModelRegistry).filter_by(name="sample-model").first():
                model = ModelRegistry(
                    name="sample-model",
                    base_model="Qwen/Qwen3-8B-Base",
                    tinker_path=None,  # Fixed: base models should have null tinker_path
                    description="Sample base model (Qwen 3 8B)",
                )
                session.add(model)
                session.commit()

            print("Sample data added")
    finally:
        session.close()
    _ensure_schema()
    _ensure_default_user()

    # CRITICAL FIX: Re-enable orphaned run recovery to handle crashes
    # This prevents runs from staying stuck in "running" status forever
    await _rehydrate_pending_runs()
    logger.info("Startup: Orphaned runs recovered successfully")

    yield
    await job_runner.cleanup()


job_runner = JobRunner()
app = FastAPI(title="Tinker Platform Backend", version="0.1.0", lifespan=lifespan)

# SECURITY: Configure CORS with explicit origins from environment
# In production, set CORS_ORIGINS="https://yourdomain.com,https://api.yourdomain.com"
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://localhost:3002,http://127.0.0.1:3002",
).split(",")
logger.info(f"CORS configured for origins: {CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Explicit whitelist only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicit methods
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],  # Explicit headers
)

# Mount static files for dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")


def _ensure_schema() -> None:
    try:
        with engine.connect() as conn:
            inspector = inspect(conn)
            run_columns = {col["name"] for col in inspector.get_columns("runs")}
    except Exception:
        run_columns = set()
    if "progress" not in run_columns:
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE runs ADD COLUMN progress FLOAT DEFAULT 0")
                )
        except Exception:
            pass


def _ensure_default_user() -> None:
    session: Session = SessionLocal()
    try:
        if not API_KEY:
            return
        user = (
            session.execute(select(User).where(User.api_key == API_KEY))
            .scalars()
            .first()
        )
        if not user:
            try:
                session.add(
                    User(
                        username="default",
                        email="demo@tinker.local",
                        api_key=API_KEY,
                    )
                )
                session.commit()
            except Exception as e:
                print(f"Error creating default user: {e}")
                session.rollback()
    finally:
        session.close()


def get_current_user(
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> User:
    if API_KEY and not ALLOW_ANONYMOUS:
        if x_api_key != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
            )
        key = API_KEY
    else:
        key = x_api_key or API_KEY or "anonymous"

    user = db.execute(select(User).where(User.api_key == key)).scalars().first()
    if not user:
        user = User(username=f"user-{key[:6]}", api_key=key)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


@app.get("/health", response_model=MessageResponse)
def health() -> MessageResponse:
    return MessageResponse(message="ok")


@app.get("/me", response_model=UserRead)
def get_profile(current_user: User = Depends(get_current_user)) -> UserRead:
    return UserRead.model_validate(current_user)


@app.post("/projects", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ProjectRead:
    project = Project(
        name=payload.name, description=payload.description, owner_id=current_user.id
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return ProjectRead.model_validate(project)


@app.get("/projects", response_model=list[ProjectRead])
def list_projects(
    db: Session = Depends(get_db),
) -> list[ProjectRead]:
    projects = (
        db.execute(select(Project).order_by(Project.created_at.desc())).scalars().all()
    )
    return [ProjectRead.model_validate(project) for project in projects]


@app.post("/datasets", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
def register_dataset(
    payload: DatasetRegistration,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DatasetRead:
    dataset = Dataset(
        name=payload.name,
        kind=payload.kind,
        spec=payload.spec,
        description=payload.description,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return DatasetRead.model_validate(dataset)


@app.get("/datasets", response_model=list[DatasetRead])
def list_datasets(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[DatasetRead]:
    datasets = (
        db.execute(select(Dataset).order_by(Dataset.created_at.desc())).scalars().all()
    )
    return [DatasetRead.model_validate(ds) for ds in datasets]


@app.post("/datasets/validate")
async def validate_dataset(
    dataset_path: str,
    renderer_name: str = "role_colon",
    model_name: str = "meta-llama/Llama-3.1-8B",
    max_examples: int = 10,
) -> dict:
    """Validate dataset format using tinker_cookbook renderers"""
    try:
        from tinker_cookbook.renderers import get_renderer, Message
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        from tinker_cookbook.supervised.data import conversation_to_datum
        import datasets

        errors = []
        warnings = []
        examples_checked = []

        # Load dataset
        try:
            if dataset_path.endswith(".jsonl"):
                dataset = datasets.load_dataset(
                    "json", data_files=dataset_path, split="train"
                )
            elif "/" in dataset_path:  # HuggingFace dataset
                dataset = datasets.load_dataset(dataset_path, split="train")
            else:
                return {
                    "valid": False,
                    "errors": [f"Invalid dataset path: {dataset_path}"],
                }
        except Exception as e:
            return {"valid": False, "errors": [f"Failed to load dataset: {str(e)}"]}

        # Get tokenizer and renderer
        try:
            tokenizer = get_tokenizer(model_name)
            renderer = get_renderer(renderer_name, tokenizer)
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Failed to initialize tokenizer/renderer: {str(e)}"],
            }

        # Validate examples
        total_examples = min(len(dataset), max_examples)

        for i in range(total_examples):
            example = dataset[i]
            example_errors = []

            try:
                # Check if messages field exists
                if "messages" not in example:
                    example_errors.append("Missing 'messages' field")
                    errors.append(f"Example {i}: Missing 'messages' field")
                    continue

                messages = example["messages"]

                # Check if messages is a list
                if not isinstance(messages, list):
                    example_errors.append("'messages' must be a list")
                    errors.append(f"Example {i}: 'messages' must be a list")
                    continue

                # Check if messages is not empty
                if len(messages) == 0:
                    example_errors.append("'messages' list is empty")
                    errors.append(f"Example {i}: 'messages' list is empty")
                    continue

                # Check last message is assistant
                if messages[-1].get("role") != "assistant":
                    warnings.append(
                        f"Example {i}: Last message should be 'assistant' role (got '{messages[-1].get('role')}')"
                    )

                # Validate each message structure
                for j, msg in enumerate(messages):
                    if "role" not in msg:
                        example_errors.append(f"Message {j}: Missing 'role' field")
                    if "content" not in msg:
                        example_errors.append(f"Message {j}: Missing 'content' field")
                    if msg.get("content") is None or msg.get("content") == "":
                        warnings.append(f"Example {i}, Message {j}: Empty content")

                # Try to convert to Datum using tinker_cookbook
                try:
                    datum = conversation_to_datum(
                        conversation=messages, renderer=renderer, max_length=None
                    )
                    token_count = len(datum.tokens)

                    examples_checked.append(
                        {
                            "index": i,
                            "valid": len(example_errors) == 0,
                            "num_messages": len(messages),
                            "num_tokens": token_count,
                            "roles": [msg.get("role") for msg in messages],
                            "errors": example_errors,
                        }
                    )
                except Exception as e:
                    example_errors.append(f"Renderer failed: {str(e)}")
                    errors.append(f"Example {i}: Renderer conversion failed - {str(e)}")
                    examples_checked.append(
                        {"index": i, "valid": False, "errors": example_errors}
                    )

            except Exception as e:
                errors.append(f"Example {i}: Validation error - {str(e)}")
                examples_checked.append(
                    {"index": i, "valid": False, "errors": [str(e)]}
                )

        # Calculate statistics
        valid_count = sum(1 for ex in examples_checked if ex.get("valid", False))
        total_tokens = sum(
            ex.get("num_tokens", 0) for ex in examples_checked if ex.get("valid", False)
        )
        avg_tokens = total_tokens / valid_count if valid_count > 0 else 0

        stats = {
            "total_examples": len(dataset),
            "examples_checked": total_examples,
            "valid_examples": valid_count,
            "invalid_examples": total_examples - valid_count,
            "total_tokens": total_tokens,
            "avg_tokens_per_example": round(avg_tokens, 2),
        }

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
            "examples": examples_checked,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {
            "valid": False,
            "errors": [f"Validation failed: {str(e)}"],
            "warnings": [],
            "stats": {},
            "examples": [],
        }


@app.get("/datasets/preview")
async def preview_dataset(dataset_path: str, limit: int = 5, offset: int = 0) -> dict:
    """Preview first N examples from dataset"""
    try:
        import datasets

        # Load dataset
        if dataset_path.endswith(".jsonl"):
            dataset = datasets.load_dataset(
                "json", data_files=dataset_path, split="train"
            )
        elif "/" in dataset_path:  # HuggingFace dataset
            dataset = datasets.load_dataset(dataset_path, split="train")
        else:
            return {"error": f"Invalid dataset path: {dataset_path}", "examples": []}

        # Get examples
        total_count = len(dataset)
        end_idx = min(offset + limit, total_count)

        examples = []
        for i in range(offset, end_idx):
            examples.append(dataset[i])

        return {
            "total_count": total_count,
            "offset": offset,
            "limit": limit,
            "examples": examples,
        }

    except Exception as e:
        return {"error": f"Failed to preview dataset: {str(e)}", "examples": []}


@app.post("/runs", response_model=RunRead, status_code=status.HTTP_201_CREATED)
async def create_run(
    payload: RunCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RunRead:
    project = db.get(Project, payload.project_id)
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Project not found")

    if payload.dataset_id is not None:
        dataset = db.get(Dataset, payload.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

    run = Run(
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        recipe_type=payload.recipe_type,
        config_json=payload.config_json.dict(exclude_none=True),
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    await job_runner.submit(run.id)
    return _to_run_read(run)


@app.get("/runs", response_model=RunListResponse)
def list_runs(
    project_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
) -> RunListResponse:
    query = select(Run)
    if project_id is not None:
        query = query.where(Run.project_id == project_id)
    runs = db.execute(query.order_by(Run.created_at.desc())).scalars().all()
    return RunListResponse(runs=[_to_run_read(run) for run in runs])


@app.get("/runs/{run_id}", response_model=RunDetailResponse)
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
) -> RunDetailResponse:
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    tail = _read_logs(run.log_path)
    metrics = _read_metrics(run.log_path)
    checkpoints = (
        db.execute(select(Checkpoint).where(Checkpoint.run_id == run.id))
        .scalars()
        .all()
    )

    return RunDetailResponse(
        **_to_run_read(run).model_dump(),
        logs_tail=tail.tail if tail else None,
        metrics=metrics.metrics if metrics else None,
        latest_metrics=metrics.latest if metrics else None,
        checkpoints=[CheckpointRead.model_validate(cp) for cp in checkpoints],
    )


@app.post("/runs/{run_id}/cancel", response_model=RunCancelResponse)
async def cancel_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RunCancelResponse:
    run = db.get(Run, run_id)
    if not run or run.project.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status not in {"pending", "running"}:
        raise HTTPException(status_code=400, detail="Run is not cancellable")

    cancelled = await job_runner.cancel(run_id)
    if not cancelled:
        run.status = "cancelled"
        run.finished_at = datetime.utcnow()
        db.add(run)
        db.commit()

    return RunCancelResponse(run_id=run_id, status="cancelled")


@app.get("/checkpoints/{checkpoint_id}/download")
async def download_checkpoint(
    checkpoint_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get download URL for checkpoint based on Tinker docs"""
    checkpoint = db.get(Checkpoint, checkpoint_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    # Check ownership
    run = db.get(Run, checkpoint.run_id)
    if not run or run.project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    if not TINKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tinker API not available")

    if not checkpoint.tinker_path:
        raise HTTPException(status_code=400, detail="Checkpoint has no tinker path")

    try:
        # Use Tinker API to get signed download URL (from tinker-docs)
        service_client = tinker.ServiceClient()
        rest_client = service_client.create_rest_client()

        # Get signed URL for checkpoint archive
        future = rest_client.get_checkpoint_archive_url_from_tinker_path(
            checkpoint.tinker_path
        )
        checkpoint_archive_url_response = future.result()

        # Return signed URL and expiration
        return {
            "download_url": checkpoint_archive_url_response.url,
            "expires_at": checkpoint_archive_url_response.expires.isoformat()
            if hasattr(checkpoint_archive_url_response.expires, "isoformat")
            else str(checkpoint_archive_url_response.expires),
            "checkpoint_path": checkpoint.tinker_path,
            "filename": f"checkpoint_{checkpoint_id}.tar",
        }
    except Exception as e:
        print(f"Error getting checkpoint download URL: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get download URL: {str(e)}"
        )


@app.post("/runs/{run_id}/resume")
async def resume_training_from_checkpoint(
    run_id: int,
    checkpoint_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Resume training from a checkpoint by creating a new run.

    Based on tinker-cookbook checkpoint_utils.py pattern.
    """
    # Get original run
    original_run = db.get(Run, run_id)
    if not original_run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Check ownership
    if original_run.project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Get checkpoint
    checkpoint = db.get(Checkpoint, checkpoint_id)
    if not checkpoint or checkpoint.run_id != run_id:
        raise HTTPException(status_code=404, detail="Checkpoint not found for this run")

    # Only allow resuming from "state" checkpoints (not "sampler" only)
    if checkpoint.kind != "state" and checkpoint.kind != "both":
        raise HTTPException(
            status_code=400,
            detail="Can only resume from state checkpoints (not sampler-only checkpoints)",
        )

    try:
        # Create new run with checkpoint as starting point
        config = original_run.config_json or {}
        config["load_checkpoint_path"] = (
            checkpoint.tinker_path
        )  # Key field for resuming

        new_run = Run(
            project_id=original_run.project_id,
            dataset_id=original_run.dataset_id,
            recipe_type=original_run.recipe_type,
            status="pending",
            config_json=config,
            logs_path=None,  # Will be created on job start
            created_at=datetime.utcnow(),
        )

        db.add(new_run)
        db.commit()
        db.refresh(new_run)

        logger.info(
            f"Created resume run {new_run.id} from checkpoint {checkpoint_id} of run {run_id}"
        )

        return {
            "success": True,
            "new_run_id": new_run.id,
            "original_run_id": run_id,
            "checkpoint_id": checkpoint_id,
            "message": f"Created new run {new_run.id} that will resume from checkpoint",
        }

    except Exception as e:
        logger.error(f"Failed to create resume run: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to create resume run: {str(e)}"
        )


@app.get("/hyperparameters/auto-lr")
async def get_auto_learning_rate(
    model_name: str = Query(..., description="Model name to calculate optimal LR for"),
    is_lora: bool = Query(True, description="Whether using LoRA fine-tuning"),
):
    """Calculate optimal learning rate for a model using tinker-cookbook hyperparameter utils"""
    try:
        from tinker_cookbook.hyperparam_utils import get_lr

        optimal_lr = get_lr(model_name, is_lora)
        return {
            "model_name": model_name,
            "is_lora": is_lora,
            "optimal_learning_rate": optimal_lr,
            "explanation": f"Calculated using tinker-cookbook hyperparameter utils for {'LoRA' if is_lora else 'full'} fine-tuning",
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to calculate learning rate: {str(e)}"
        )


@app.post("/hyperparameters/calculate")
async def calculate_all_hyperparameters(request: HyperparamRequest):
    """
    Calculate all recommended hyperparameters for a model and recipe.

    Based on Tinker documentation:
    - https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams
    - https://tinker-docs.thinkingmachines.ai/lora-primer
    """
    try:
        import traceback
        from utils.hyperparam_calculator import HyperparamCalculator

        logger.info(
            f"Calculating hyperparameters for model={request.model_name}, recipe={request.recipe_type}, lora_rank={request.lora_rank}"
        )

        recommendations = HyperparamCalculator.get_all_recommendations(
            model_name=request.model_name,
            recipe_type=request.recipe_type,
            lora_rank=request.lora_rank,
        )

        metadata = recommendations.pop("_metadata")

        return {
            "success": True,
            "model_name": request.model_name,
            "recipe_type": request.recipe_type,
            "recommendations": recommendations,
            "explanation": {
                "learning_rate": metadata.get(
                    "lr_formula",
                    f"Calculated LR: {recommendations.get('learning_rate', 'unknown')}",
                ),
                "batch_size": f"Optimized for {request.recipe_type} training. Tinker docs recommend 128 or smaller for SFT.",
                "lora_rank": f"Default is 32 for most use cases. Independent of learning rate.",
                "notes": metadata.get("notes", []),
                "source": metadata.get("source", ""),
            },
        }
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        logger.error(f"Failed to calculate hyperparameters: {str(e)}\n{tb}")
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate hyperparameters: {str(e)}"
        )


@app.get("/models/{model_name:path}/renderers")
async def get_model_renderers(model_name: str):
    """Get recommended renderers for a specific model"""
    try:
        from tinker_cookbook.model_info import get_recommended_renderer_names

        renderers = get_recommended_renderer_names(model_name)
        return {
            "model_name": model_name,
            "recommended_renderers": renderers,
            "default_renderer": renderers[0] if renderers else None,
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to get renderers: {str(e)}"
        )


@app.get("/runs/{run_id}/logs", response_model=LogTailResponse)
def get_run_logs(
    run_id: int,
    tail_lines: int = Query(default=200, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> LogTailResponse:
    run = db.get(Run, run_id)
    if not run or run.project.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Run not found")
    tail = _read_logs(run.log_path, tail_lines=tail_lines)
    if not tail:
        return LogTailResponse(run_id=run_id, tail="", total_bytes=0)
    return LogTailResponse(run_id=run_id, tail=tail.tail, total_bytes=tail.total_bytes)


@app.get("/runs/{run_id}/metrics", response_model=MetricsResponse)
def get_run_metrics(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MetricsResponse:
    run = db.get(Run, run_id)
    if not run or run.project.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Run not found")
    metrics = _read_metrics(run.log_path)
    return MetricsResponse(run_id=run_id, metrics=metrics.metrics if metrics else [])


@app.post("/estimate-cost")
async def estimate_training_cost(
    model_name: str,
    dataset_size: int,
    epochs: int = 3,
    batch_size: int = 1,
) -> dict:
    """
    Estimate training cost based on Tinker rate card
    Note: This is a simplified estimation. Actual costs may vary.
    """
    try:
        # Simplified rate card (USD per million tokens)
        # These are example rates - adjust based on actual Tinker pricing
        model_rates = {
            "meta-llama/Llama-3.1-8B": 0.15,
            "meta-llama/Llama-3.1-8B-Instruct": 0.15,
            "meta-llama/Llama-3.1-70B": 0.75,
            "meta-llama/Llama-3.1-70B-Instruct": 0.75,
            "Qwen/Qwen2.5-7B": 0.15,
            "Qwen/Qwen2.5-7B-Instruct": 0.15,
            "Qwen/Qwen2.5-32B": 0.50,
            "Qwen/Qwen2.5-32B-Instruct": 0.50,
            "Qwen/Qwen2.5-72B": 0.75,
            "Qwen/Qwen2.5-72B-Instruct": 0.75,
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 0.15,
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 0.50,
        }

        # Get rate for model (default to 0.15 if not found)
        rate_per_m_tokens = model_rates.get(model_name, 0.15)

        # Estimate tokens per example (average)
        avg_tokens_per_example = 2048  # Approximate

        # Calculate total tokens
        # tokens = dataset_size * epochs * avg_tokens_per_example
        total_tokens = dataset_size * epochs * avg_tokens_per_example

        # Calculate cost
        cost_usd = (total_tokens / 1_000_000) * rate_per_m_tokens

        # Estimate time (very rough estimate)
        # Assuming ~1000 tokens/second throughput
        estimated_seconds = total_tokens / 1000
        estimated_minutes = estimated_seconds / 60
        estimated_hours = estimated_minutes / 60

        return {
            "estimated_cost_usd": round(cost_usd, 2),
            "estimated_tokens": total_tokens,
            "rate_per_m_tokens": rate_per_m_tokens,
            "dataset_size": dataset_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "estimated_duration_minutes": round(estimated_minutes, 1),
            "note": "This is a rough estimate. Actual costs may vary based on model, sequence length, and system load.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")


@app.get("/models", response_model=ModelCatalogResponse)
def list_models(
    db: Session = Depends(get_db),
) -> ModelCatalogResponse:
    print(f"SUPPORTED_MODELS count: {len(SUPPORTED_MODELS)}")
    registry_entries = (
        db.execute(select(ModelRegistry).order_by(ModelRegistry.created_at.desc()))
        .scalars()
        .all()
    )
    for entry in registry_entries:
        entry.meta = entry.meta or {}
    models = [ModelRead.model_validate(entry) for entry in registry_entries]
    response = ModelCatalogResponse(
        supported_models=SUPPORTED_MODELS, registered_models=models
    )
    return response


@app.post("/models", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
def register_model(
    payload: ModelRegistration,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> ModelRead:
    entry = ModelRegistry(
        name=payload.name,
        base_model=payload.base_model,
        tinker_path=payload.tinker_path,
        description=payload.description,
        meta=payload.meta,
        project_id=payload.project_id,
        run_id=payload.run_id,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return ModelRead.model_validate(entry)


@app.post("/chat", response_model=ChatResponse)
def chat_with_model(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Chat with a model (base model or fine-tuned).

    Supports:
    - request.run_id: Chat with fine-tuned model from a training run
    - request.model_id: Chat with registered model
    - Neither: Chat with default model
    """
    print(f"\n[CHAT] ========== NEW CHAT REQUEST ==========", flush=True)
    print(
        f"[CHAT] Request: run_id={request.run_id}, model_id={request.model_id}",
        flush=True,
    )
    print(f"[CHAT] CHAT_INFERENCE_AVAILABLE={CHAT_INFERENCE_AVAILABLE}", flush=True)
    print(f"[CHAT] TINKER_AVAILABLE={TINKER_AVAILABLE}", flush=True)

    if not CHAT_INFERENCE_AVAILABLE:
        print(
            "[CHAT] chat_inference module not available, using simulation", flush=True
        )
        model_name = "Simulated Model"
        if request.run_id is not None:
            run = db.get(Run, request.run_id)
            if run:
                model_name = (
                    f"Fine-tuned {run.recipe_type} model (run {request.run_id})"
                )
        elif request.model_id is not None:
            entry = db.get(ModelRegistry, request.model_id)
            if entry:
                model_name = entry.name

        response_text = f"[Simulated response from {model_name}] Echo: {request.prompt}"
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(response_text.split())

        return ChatResponse(
            response=response_text,
            model=model_name,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # Use simplified synchronous chat client
    from simple_chat_inference import SimpleChatClient

    # Determine base model with priority order:
    # 1. Explicit base_model in request
    # 2. From run config (if run_id provided)
    # 3. From model registry (if model_id provided)
    # 4. Default
    base_model = request.base_model or "meta-llama/Llama-3.1-8B-Instruct"
    model_name = "Base Model"

    if request.run_id is not None:
        run = db.get(Run, request.run_id)
        if run and run.config_json:
            base_model = run.config_json.get("base_model", base_model)
            model_name = f"Fine-tuned {run.recipe_type} model (run {request.run_id})"
    elif request.model_id is not None:
        entry = db.get(ModelRegistry, request.model_id)
        if entry:
            base_model = entry.base_model
            model_name = entry.name
    elif request.base_model:
        # Direct base model specified
        model_name = request.base_model

    print(f"[CHAT] Using base_model: {base_model}", flush=True)
    print(f"[CHAT] Model name: {model_name}", flush=True)

    try:
        if TINKER_AVAILABLE:
            try:
                print(f"[CHAT] Creating SimpleChatClient...", flush=True)
                client = SimpleChatClient(base_model=base_model)
                client.initialize()

                print(f"[CHAT] Generating response...", flush=True)
                response_text = client.chat(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens or 256,
                    temperature=request.temperature or 0.7,
                )

                print(f"[CHAT] SUCCESS! Real inference completed!", flush=True)
                print(f"[CHAT] Response preview: {response_text[:100]}...", flush=True)

            except Exception as e:
                print(
                    f"[CHAT] Real inference FAILED: {type(e).__name__}: {e}", flush=True
                )
                import traceback

                print(f"[CHAT] Traceback:", flush=True)
                traceback.print_exc()
                print(f"[CHAT] Falling back to simulation", flush=True)
                response_text = (
                    f"[Simulated response from {model_name}] Echo: {request.prompt}"
                )
        else:
            print(f"[CHAT] Tinker not available, using simulation", flush=True)
            response_text = (
                f"[Simulated response from {model_name}] Echo: {request.prompt}"
            )

    except Exception as e:
        print(f"[CHAT] Unexpected error: {type(e).__name__}: {e}", flush=True)
        import traceback

        traceback.print_exc()
        response_text = f"Error generating response: {str(e)}"

    # Calculate token counts (approximate)
    prompt_tokens = len(request.prompt.split())
    completion_tokens = len(response_text.split())
    total_tokens = prompt_tokens + completion_tokens

    print(f"[CHAT] ========== CHAT REQUEST COMPLETE ==========\n", flush=True)

    return ChatResponse(
        response=response_text,
        model=model_name,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


@app.post("/sample", response_model=SampleResponse)
async def sample_model(
    payload: SampleRequest,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> SampleResponse:
    """Sample from a model (supports both model_id and model_path)"""
    model_path: str

    if payload.model_id is not None:
        entry = db.get(ModelRegistry, payload.model_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Model not found")
        model_path = entry.tinker_path or entry.base_model
    elif payload.model_path:
        model_path = payload.model_path
    else:
        raise HTTPException(status_code=400, detail="model_id or model_path required")

    print(
        f"DEBUG: Starting sample_model with TINKER_AVAILABLE={TINKER_AVAILABLE}, model_path={model_path}",
        flush=True,
    )

    # Enhanced simulation that shows which model is selected
    model_name = "Unknown Model"
    if payload.model_id:
        entry = db.get(ModelRegistry, payload.model_id)
        if entry:
            model_name = entry.name
    elif payload.model_path:
        model_name = payload.model_path.split("/")[-1]

    print(f"DEBUG: Sample simulation for model: {model_name}", flush=True)
    base_response = _simulate_model_response(
        payload.prompt, payload.sampling_params.max_tokens - 50
    )
    generated_text = f"[Sample from {model_name}] {base_response}"
    sequence = SampleSequence(
        text=generated_text, tokens=[ord(c) for c in generated_text[:64]]
    )
    return SampleResponse(
        model=model_path,
        prompt=payload.prompt,
        sequences=[sequence],
        sampling_params=payload.sampling_params,
    )


@app.get("/evaluations", response_model=list[EvaluationRead])
def list_evaluations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[EvaluationRead]:
    evaluations = (
        db.execute(
            select(Evaluation)
            .join(Run)
            .join(Project)
            .where(Project.owner_id == current_user.id)
            .order_by(Evaluation.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [EvaluationRead.model_validate(ev) for ev in evaluations]


async def _rehydrate_pending_runs() -> None:
    session: Session = SessionLocal()
    try:
        runs = (
            session.execute(select(Run).where(Run.status.in_(["pending", "running"])))
            .scalars()
            .all()
        )
        for run in runs:
            run.status = "pending"
            run.started_at = None
            run.finished_at = None
            session.add(run)
            session.commit()
            await job_runner.submit(run.id)
    finally:
        session.close()


def _to_run_read(run: Run) -> RunRead:
    config_data = run.config_json or {}
    try:
        if isinstance(config_data, RunConfig):
            config_obj = config_data
        else:
            config_obj = RunConfig.model_validate(config_data)
    except Exception as e:
        print(f"Error validating config for run {run.id}: {e}")
        print(f"Config data: {config_data}")
        # Fallback to a default config
        config_obj = RunConfig()
    return RunRead(
        id=run.id,
        project_id=run.project_id,
        dataset_id=run.dataset_id,
        recipe_type=run.recipe_type,
        status=run.status,
        progress=run.progress,
        config_json=config_obj,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        log_path=run.log_path,
    )


class _LogTail:
    def __init__(self, tail: str, total_bytes: int) -> None:
        self.tail = tail
        self.total_bytes = total_bytes


class _MetricsResult:
    def __init__(self, metrics: list[dict]) -> None:
        self.metrics = metrics
        self.latest = metrics[-1] if metrics else None


def _read_logs(path: Optional[str], tail_lines: int = None) -> Optional[_LogTail]:
    """Read log files for a training run.

    Args:
        path: Path to the main log file
        tail_lines: Number of lines to read from end (None = use config default)

    Returns:
        LogTail object with content and size, or None if path invalid
    """
    if not path:
        logger.warning("No log path provided")
        return None

    file_path = Path(path)
    if not file_path.exists():
        logger.warning(f"Log file does not exist: {file_path}")
        return None

    if tail_lines is None:
        tail_lines = settings.log_tail_lines

    # Read main wrapper logs using utility
    content = read_file_tail(file_path, num_lines=tail_lines)
    if content is None:
        logger.error(f"Failed to read log file: {file_path}")
        return None

    # Strip ANSI codes from content
    content = strip_ansi_codes(content)

    # Also try to read actual training logs from logs/logs.log if exists
    training_logs_path = ArtifactPathResolver.get_training_logs_path(path)
    if training_logs_path:
        training_content = read_file_tail(training_logs_path, num_lines=tail_lines // 2)
        if training_content and training_content.strip():
            training_content = strip_ansi_codes(training_content)
            content += "\n\n=== Training Logs ===\n" + training_content
            logger.debug(f"Appended training logs from {training_logs_path}")

    total_bytes = file_path.stat().st_size
    return _LogTail(tail=content, total_bytes=total_bytes)


def _read_metrics(log_path: Optional[str]) -> Optional[_MetricsResult]:
    """Read metrics from training run's metrics.jsonl file.

    Args:
        log_path: Path to the run's log file

    Returns:
        MetricsResult with parsed metrics, or None if not found
    """
    if not log_path:
        logger.warning("No log path provided for metrics reading")
        return None

    # Use utility to get metrics file path with fallback logic
    metrics_file = ArtifactPathResolver.get_metrics_path(log_path)
    if not metrics_file:
        logger.debug(f"No metrics file found for run at {log_path}")
        return None

    # Use utility to read and parse JSONL file with NaN handling
    try:
        metrics = read_jsonl_file(str(metrics_file), skip_errors=True)
        logger.debug(f"Read {len(metrics)} metric entries from {metrics_file}")
        return _MetricsResult(metrics=metrics)
    except FileNotFoundError:
        logger.warning(f"Metrics file not found: {metrics_file}")
        return None
    except Exception as e:
        logger.error(f"Error reading metrics file {metrics_file}: {e}")
        return None


def _simulate_model_response(prompt: str, max_tokens: int) -> str:
    suffix = " :: generated by simulated sampler"
    base = prompt.strip()
    if len(base) > max_tokens:
        base = base[: max_tokens // 2]
    return f"{base}{suffix}"


@app.get("/runs/{run_id}/realtime-metrics", response_model=RealtimeMetricsResponse)
def get_realtime_metrics(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RealtimeMetricsResponse:
    """Get real-time training metrics for a run"""
    run = db.get(Run, run_id)
    if not run or run.project.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Run not found")

    # Check for real-time metrics file
    metrics_file = (
        Path(run.log_path).parent.parent / "realtime_metrics.json"
        if run.log_path
        else None
    )

    if metrics_file and metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Fallback to basic run info
    return RealtimeMetricsResponse(
        run_id=run_id,
        status=run.status,
        progress=run.progress or 0.0,
        current_step="unknown",
        current_loss="unknown",
        current_lr="unknown",
        tokens_processed="unknown",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/runs/{run_id}/evaluate", response_model=EvaluationResponse)
async def evaluate_run(
    run_id: int,
    evaluation_request: EvaluationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EvaluationResponse:
    """Evaluate a completed training run using trained model with tinker_cookbook"""
    run = db.get(Run, run_id)
    if not run or run.project.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status not in ["completed", "evaluated"]:
        raise HTTPException(
            status_code=400, detail="Run must be completed before evaluation"
        )

    if not TINKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tinker API not available")

    try:
        # Get the trained model path
        trained_model_path = None
        base_model = None

        checkpoints = (
            db.execute(select(Checkpoint).where(Checkpoint.run_id == run_id))
            .scalars()
            .all()
        )

        if checkpoints:
            # Use the latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda cp: cp.step)
            trained_model_path = latest_checkpoint.tinker_path

        # Get base model from run config
        config = run.config_json or {}
        base_model = (
            config.get("base_model") or config.get("model") or config.get("model_name")
        )

        if not trained_model_path:
            # Fallback to run's config if no checkpoint
            trained_model_path = config.get("model_path")

        if not trained_model_path:
            raise HTTPException(
                status_code=404, detail="No trained model found for evaluation"
            )

        if not base_model:
            raise HTTPException(
                status_code=400, detail="Base model not found in run configuration"
            )

        # Create sampling client using tinker_cookbook
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(
            model_path=trained_model_path, base_model=base_model
        )

        # Get renderer name from config
        renderer_name = config.get("renderer_name", "role_colon")

        # Run simple completion test as basic evaluation
        test_prompts = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "user", "content": "Explain what machine learning is."},
        ]

        eval_results = {
            "model_path": trained_model_path,
            "base_model": base_model,
            "renderer": renderer_name,
            "test_completions": [],
        }

        # Test each prompt
        from tinker_cookbook.renderers import get_renderer, Message
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer(base_model)
        renderer = get_renderer(renderer_name, tokenizer)

        for prompt in test_prompts:
            try:
                # Build prompt using renderer
                model_input = renderer.build_generation_prompt([Message(**prompt)])

                # Sample from model with stop sequences from renderer
                future = sampling_client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=100,
                        temperature=0.7,
                        top_p=0.9,
                        stop=renderer.get_stop_sequences(),
                    ),
                )
                result = future.result()

                if result.sequences:
                    response_tokens = result.sequences[0].tokens
                    # Use renderer to parse the response properly
                    parsed_messages = renderer.parse_response(response_tokens)

                    if parsed_messages:
                        response_text = parsed_messages[0]["content"]
                    else:
                        response_text = tokenizer.decode(response_tokens)

                    eval_results["test_completions"].append(
                        {
                            "prompt": prompt["content"],
                            "completion": response_text,
                            "success": True,
                        }
                    )
                else:
                    eval_results["test_completions"].append(
                        {
                            "prompt": prompt["content"],
                            "completion": None,
                            "success": False,
                            "error": "No sequences generated",
                        }
                    )

            except Exception as e:
                eval_results["test_completions"].append(
                    {
                        "prompt": prompt["content"],
                        "completion": None,
                        "success": False,
                        "error": str(e),
                    }
                )

        # Save evaluation results
        eval_results_file = Path(run.log_path).parent.parent / "evaluation_results.json"
        eval_results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_results_file, "w") as f:
            json.dump(eval_results, f, indent=2)

        # Create evaluation record
        evaluation = Evaluation(
            run_id=run_id,
            evaluator_name=evaluation_request.evaluation_type,
            metrics=eval_results,
            created_at=datetime.utcnow(),
        )
        db.add(evaluation)

        # Mark run as evaluated
        if run.status == "completed":
            run.status = "evaluated"
            db.add(run)

        db.commit()

        return EvaluationResponse(
            run_id=run_id,
            evaluation_type=evaluation_request.evaluation_type,
            timestamp=datetime.utcnow().isoformat(),
            results=eval_results,
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/runs/{run_id}/visualization-data", response_model=dict)
def get_visualization_data(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Get visualization data for a training run"""
    run = db.get(Run, run_id)
    if not run or run.project.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Run not found")

    # Get metrics data using utility (handles path resolution and NaN values)
    metrics_data = []
    metrics_file = None

    # Only try to get metrics if log_path exists
    if run.log_path:
        metrics_file = ArtifactPathResolver.get_metrics_path(run.log_path)

    if metrics_file and metrics_file.exists():
        try:
            metrics = read_jsonl_file(str(metrics_file), skip_errors=True)

            for metric in metrics:
                # Support both 'loss' and 'train_mean_nll' field names
                loss_value = metric.get("loss") or metric.get("train_mean_nll", 0)
                metrics_data.append(
                    {
                        "step": metric.get("step", 0),
                        "loss": loss_value,
                        "train_mean_nll": loss_value,  # Include both for compatibility
                        "learning_rate": metric.get("learning_rate", 0),
                        "progress": metric.get("progress", 0),
                        "timestamp": metric.get("timestamp", ""),
                    }
                )

            logger.debug(f"Loaded {len(metrics_data)} metrics for run {run_id}")
        except Exception as e:
            logger.error(f"Error reading metrics for visualization (run {run_id}): {e}")

    # Get evaluation data if available
    eval_data = None
    eval_file = Path(run.log_path).parent.parent / "evaluation_results.json"
    if eval_file.exists():
        try:
            with open(eval_file, "r") as f:
                eval_data = json.load(f)
        except Exception:
            pass

    return {
        "run_id": run_id,
        "status": run.status,
        "metrics": metrics_data,
        "evaluation": eval_data,
        "config": run.config_json,
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================
# HuggingFace Integration Endpoints
# =============================================


@app.post("/settings/huggingface/token", response_model=HFTokenStatusResponse)
async def save_hf_token(
    request: HFTokenSaveRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save and verify HuggingFace token."""
    from utils.encryption import encrypt_token
    from services.huggingface_service import HuggingFaceService

    try:
        # Verify token is valid
        hf_service = HuggingFaceService(token=request.token)
        verification = hf_service.verify_token()

        if not verification["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid HuggingFace token: {verification.get('error', 'Unknown error')}",
            )

        # Encrypt and store token
        encrypted_token = encrypt_token(request.token)
        current_user.hf_token_encrypted = encrypted_token
        current_user.hf_username = verification["username"]
        current_user.hf_token_last_verified = datetime.utcnow()

        db.add(current_user)
        db.commit()
        db.refresh(current_user)

        return HFTokenStatusResponse(
            connected=True,
            username=current_user.hf_username,
            last_verified=current_user.hf_token_last_verified,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save HF token: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save token: {str(e)}")


@app.delete("/settings/huggingface/token", response_model=MessageResponse)
async def remove_hf_token(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Remove stored HuggingFace token."""
    current_user.hf_token_encrypted = None
    current_user.hf_username = None
    current_user.hf_token_last_verified = None

    db.add(current_user)
    db.commit()

    return MessageResponse(message="HuggingFace token removed successfully")


@app.get("/settings/huggingface/status", response_model=HFTokenStatusResponse)
async def get_hf_status(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Check if HuggingFace token is configured."""
    return HFTokenStatusResponse(
        connected=bool(current_user.hf_token_encrypted),
        username=current_user.hf_username,
        last_verified=current_user.hf_token_last_verified,
    )


@app.post(
    "/checkpoints/{checkpoint_id}/deploy/huggingface", response_model=DeployToHFResponse
)
async def deploy_to_huggingface(
    checkpoint_id: int,
    request: DeployToHFRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Deploy checkpoint to HuggingFace Hub."""
    from utils.encryption import decrypt_token
    from services.huggingface_service import HuggingFaceService

    # 1. Verify user has HF token
    if not current_user.hf_token_encrypted:
        raise HTTPException(
            status_code=400,
            detail="HuggingFace token not configured. Please add token in Settings → Integrations.",
        )

    # 2. Get checkpoint
    checkpoint = db.get(Checkpoint, checkpoint_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    # 3. Verify checkpoint belongs to user
    run = db.get(Run, checkpoint.run_id)
    if not run or run.project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # 4. Create deployment record
    deployment = Deployment(
        checkpoint_id=checkpoint_id,
        user_id=current_user.id,
        hf_repo_name=request.repo_name,
        hf_repo_url=f"https://huggingface.co/{request.repo_name}",
        hf_model_id=request.repo_name,
        is_private=1 if request.private else 0,
        merged_weights=1 if request.merge_weights else 0,
        status="pending",
    )
    db.add(deployment)
    db.commit()
    db.refresh(deployment)

    # 5. Create background task for deployment
    background_tasks.add_task(
        deploy_checkpoint_to_hf_background,
        deployment_id=deployment.id,
        checkpoint_id=checkpoint_id,
        repo_name=request.repo_name,
        private=request.private,
        merge_weights=request.merge_weights,
        user_id=current_user.id,
    )

    return DeployToHFResponse(
        success=True,
        repo_url=f"https://huggingface.co/{request.repo_name}",
        message="Deployment started. This may take a few minutes. Check the deployments page for status.",
        deployment_id=deployment.id,
    )


async def deploy_checkpoint_to_hf_background(
    deployment_id: int,
    checkpoint_id: int,
    repo_name: str,
    private: bool,
    merge_weights: bool,
    user_id: int,
):
    """Background task to deploy checkpoint to HuggingFace."""
    from utils.encryption import decrypt_token
    from services.huggingface_service import HuggingFaceService
    import tempfile
    import shutil
    import requests
    import tarfile

    db = SessionLocal()

    try:
        deployment = db.get(Deployment, deployment_id)
        deployment.status = "uploading"
        db.commit()

        # Get user and decrypt token
        user = db.get(User, user_id)
        hf_token = decrypt_token(user.hf_token_encrypted)
        hf_service = HuggingFaceService(token=hf_token)

        # Get checkpoint and run
        checkpoint = db.get(Checkpoint, checkpoint_id)
        run = db.get(Run, checkpoint.run_id)

        # Create temporary directory for checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint"
            checkpoint_path.mkdir()

            # Download checkpoint from Tinker API
            checkpoint_downloaded = False
            if TINKER_AVAILABLE and checkpoint.tinker_path:
                try:
                    logger.info(f"Downloading checkpoint from Tinker: {checkpoint.tinker_path}")

                    # Use Tinker API to get download URL
                    service_client = tinker.ServiceClient()
                    rest_client = service_client.create_rest_client()
                    future = rest_client.get_checkpoint_archive_url_from_tinker_path(
                        checkpoint.tinker_path
                    )
                    checkpoint_archive_url_response = future.result()

                    # Download the checkpoint archive
                    download_url = checkpoint_archive_url_response.url
                    logger.info(f"Downloading from: {download_url}")

                    response = requests.get(download_url, stream=True)
                    response.raise_for_status()

                    # Save as tar file
                    tar_path = Path(temp_dir) / "checkpoint.tar"
                    with open(tar_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    logger.info(f"Downloaded checkpoint archive to {tar_path}")

                    # Extract tar archive
                    with tarfile.open(tar_path, 'r') as tar:
                        tar.extractall(checkpoint_path)

                    logger.info(f"Extracted checkpoint to {checkpoint_path}")
                    checkpoint_downloaded = True

                except Exception as e:
                    logger.error(f"Failed to download checkpoint from Tinker: {e}")
                    logger.info("Will create model card only (no checkpoint files)")
            else:
                logger.warning(f"Tinker API not available or no tinker path: {checkpoint.tinker_path}")
                logger.info("Creating model card only (no checkpoint files)")

            # Create model card
            training_config = run.config_json or {}
            training_config["dataset_name"] = "custom"  # TODO: Get from dataset
            training_config["recipe_type"] = run.recipe_type

            # Create repo
            hf_service.create_model_repo(repo_name=repo_name, private=private)

            # Create model card
            base_model = run.config_json.get("base_model", "unknown")
            hf_service.create_model_card(
                repo_name=repo_name,
                base_model=base_model,
                training_config=training_config,
                metrics=checkpoint.meta,
            )

            # Upload checkpoint files if they were downloaded
            if checkpoint_downloaded:
                logger.info("Uploading checkpoint files to HuggingFace...")
                repo_url = hf_service.upload_checkpoint(
                    checkpoint_path=checkpoint_path,
                    repo_name=repo_name,
                    commit_message=f"Upload checkpoint from Tinker - Run {run.id}, Step {checkpoint.step}"
                )
                logger.info(f"Checkpoint uploaded successfully to {repo_url}")
            else:
                logger.info("Skipping checkpoint upload (files not downloaded)")

        # Update deployment status
        deployment.status = "completed"
        deployment.deployed_at = datetime.utcnow()
        deployment.hf_repo_url = f"https://huggingface.co/{repo_name}"

        # Update checkpoint
        checkpoint.hf_repo_url = f"https://huggingface.co/{repo_name}"
        checkpoint.hf_deployed_at = datetime.utcnow()

        db.commit()

        logger.info(f"Successfully deployed checkpoint {checkpoint_id} to {repo_name}")

    except Exception as e:
        logger.error(f"Failed to deploy checkpoint {checkpoint_id}: {e}")
        import traceback

        traceback.print_exc()

        # Update deployment with error
        deployment = db.get(Deployment, deployment_id)
        deployment.status = "failed"
        deployment.error_message = str(e)
        db.commit()

    finally:
        db.close()


@app.get("/deployments", response_model=list[DeploymentRead])
async def list_deployments(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """List all deployments for current user."""
    try:
        deployments = (
            db.query(Deployment)
            .filter(Deployment.user_id == current_user.id)
            .order_by(Deployment.id.desc())  # Use ID instead of deployed_at to avoid NULL issues
            .all()
        )
        logger.info(f"Found {len(deployments)} deployments for user {current_user.id}")
        return deployments
    except Exception as e:
        logger.error(f"Error fetching deployments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch deployments: {str(e)}")


@app.get("/deployments/{deployment_id}", response_model=DeploymentRead)
async def get_deployment(
    deployment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get deployment details."""
    deployment = db.get(Deployment, deployment_id)

    if not deployment or deployment.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Deployment not found")

    return deployment


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
