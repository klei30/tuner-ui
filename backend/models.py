from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, Text, Float
from sqlalchemy.orm import relationship, synonym
from sqlalchemy.sql import func

try:  # pragma: no cover
    from .database import Base
except ImportError:  # pragma: no cover
    from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=True)
    api_key = Column(String, unique=True, nullable=True, index=True)

    # HuggingFace integration
    hf_token_encrypted = Column(Text, nullable=True)
    hf_username = Column(String, nullable=True)
    hf_token_last_verified = Column(DateTime(timezone=True), nullable=True)

    projects = relationship(
        "Project", back_populates="owner", cascade="all, delete-orphan"
    )
    deployments = relationship(
        "Deployment", back_populates="user", cascade="all, delete-orphan"
    )


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="projects")
    runs = relationship("Run", back_populates="project", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    kind = Column(String)  # huggingface, local, jsonl
    spec = Column(JSON)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __init__(self, **kwargs):
        if "spec" not in kwargs or kwargs["spec"] is None:
            raise ValueError("Dataset requires spec")
        super().__init__(**kwargs)


class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    recipe_type = Column(String)
    config_json = Column(JSON)
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    log_path = Column(String, nullable=True)
    celery_task_id = Column(String, nullable=True, index=True)  # Celery task ID for tracking background jobs

    project = relationship("Project", back_populates="runs")
    dataset = relationship("Dataset")
    checkpoints = relationship(
        "Checkpoint", back_populates="run", cascade="all, delete-orphan"
    )
    evaluations = relationship(
        "Evaluation", back_populates="run", cascade="all, delete-orphan"
    )


class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    tinker_path = Column(String, index=True)
    kind = Column(String)
    step = Column(Integer)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # HuggingFace deployment info
    hf_repo_url = Column(String, nullable=True)
    hf_deployed_at = Column(DateTime(timezone=True), nullable=True)

    run = relationship("Run", back_populates="checkpoints")
    deployments = relationship(
        "Deployment", back_populates="checkpoint", cascade="all, delete-orphan"
    )

    # Backwards-compatible attribute names used by older tests and helpers.
    path = synonym("tinker_path")
    metrics = synonym("meta")

    def __init__(self, **kwargs):
        if "path" in kwargs and "tinker_path" not in kwargs:
            kwargs["tinker_path"] = kwargs.pop("path")
        if "metrics" in kwargs and "meta" not in kwargs:
            kwargs["meta"] = kwargs.pop("metrics")
        if "run_id" not in kwargs or kwargs["run_id"] is None:
            raise ValueError("Checkpoint requires run_id")
        if "step" not in kwargs or kwargs["step"] is None:
            raise ValueError("Checkpoint requires step")
        if not isinstance(kwargs["step"], int):
            raise TypeError("Checkpoint step must be an integer")
        super().__init__(**kwargs)


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    evaluator_name = Column(String)
    metrics = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    run = relationship("Run", back_populates="evaluations")

    def __init__(self, **kwargs):
        if "run_id" not in kwargs or kwargs["run_id"] is None:
            raise ValueError("Evaluation requires run_id")
        super().__init__(**kwargs)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    base_model = Column(String, nullable=False)
    tinker_path = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    meta = Column(JSON, nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project")
    run = relationship("Run")


class Deployment(Base):
    __tablename__ = "deployments"

    id = Column(Integer, primary_key=True, index=True)
    checkpoint_id = Column(Integer, ForeignKey("checkpoints.id"))
    user_id = Column(Integer, ForeignKey("users.id"))

    # HuggingFace details
    hf_repo_name = Column(String)
    hf_repo_url = Column(String)
    hf_model_id = Column(String)
    is_private = Column(Boolean, default=False)
    merged_weights = Column(Boolean, default=True)

    # Status tracking
    status = Column(String, default="pending")  # pending, uploading, completed, failed
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    checkpoint = relationship("Checkpoint", back_populates="deployments")
    user = relationship("User", back_populates="deployments")
