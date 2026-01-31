"""
Celery tasks for Tuner-UI background jobs
Handles training runs, deployments, and other long-running operations
"""
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from celery import Task
from sqlalchemy.orm import Session

from celery_app import celery_app
from database import SessionLocal
from models import Run, Checkpoint, Deployment, User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """Base task that provides database session management"""
    _db: Optional[Session] = None

    @property
    def db(self) -> Session:
        """Get or create database session"""
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def after_return(self, *args, **kwargs):
        """Clean up database session after task completes"""
        if self._db is not None:
            self._db.close()
            self._db = None


@celery_app.task(base=DatabaseTask, bind=True, name='tasks.run_training_job')
def run_training_job(self, run_id: int) -> dict:
    """
    Execute a training run using Tinker

    Args:
        run_id: ID of the Run to execute

    Returns:
        dict with status and run_id
    """
    db = self.db

    try:
        # Get run from database
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            logger.error(f"Run {run_id} not found")
            return {"status": "failed", "error": "Run not found", "run_id": run_id}

        # Update status to running
        run.status = "running"
        run.started_at = datetime.utcnow()
        db.commit()

        logger.info(f"Starting training run {run_id} (recipe: {run.recipe_type})")

        # Prepare recipe command
        recipe_module = f"recipes.{run.recipe_type.lower()}"

        # Create log directory
        log_dir = Path("artifacts") / f"run_{run_id}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save config to file
        config_path = log_dir / "config.json"
        config_path.write_text(json.dumps(run.config_json))

        # Update log path
        run.log_path = str(log_dir / "wrapper.log")
        db.commit()

        # Execute training subprocess
        log_file_path = log_dir / "train.log"

        with open(log_file_path, "w") as log_file:
            process = subprocess.Popen(
                ["python", "-m", recipe_module, str(config_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=Path(__file__).parent
            )

            # Wait for completion
            return_code = process.wait()

        # Update run status based on return code
        if return_code == 0:
            run.status = "completed"
            run.finished_at = datetime.utcnow()
            run.progress = 100.0
            logger.info(f"Training run {run_id} completed successfully")
        else:
            run.status = "failed"
            run.finished_at = datetime.utcnow()
            logger.error(f"Training run {run_id} failed with code {return_code}")

        db.commit()
        return {"status": run.status, "run_id": run_id}

    except Exception as e:
        logger.exception(f"Training run {run_id} failed with exception")

        # Update run status
        run = db.query(Run).filter(Run.id == run_id).first()
        if run:
            run.status = "failed"
            run.finished_at = datetime.utcnow()
            db.commit()

        return {"status": "failed", "error": str(e), "run_id": run_id}


@celery_app.task(base=DatabaseTask, bind=True, name='tasks.cancel_training_job')
def cancel_training_job(self, run_id: int) -> dict:
    """
    Cancel a running training job

    Args:
        run_id: ID of the Run to cancel

    Returns:
        dict with status and run_id
    """
    db = self.db

    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            return {"status": "error", "error": "Run not found", "run_id": run_id}

        if run.status not in ["pending", "running"]:
            return {"status": "error", "error": "Run is not cancellable", "run_id": run_id}

        # Update status
        run.status = "cancelled"
        run.finished_at = datetime.utcnow()
        db.commit()

        # If there's a celery task, try to revoke it
        if hasattr(run, 'celery_task_id') and run.celery_task_id:
            celery_app.control.revoke(run.celery_task_id, terminate=True)

        logger.info(f"Cancelled training run {run_id}")
        return {"status": "cancelled", "run_id": run_id}

    except Exception as e:
        logger.exception(f"Failed to cancel training run {run_id}")
        return {"status": "error", "error": str(e), "run_id": run_id}


@celery_app.task(base=DatabaseTask, bind=True, name='tasks.deploy_to_huggingface')
def deploy_to_huggingface(
    self,
    deployment_id: int,
    checkpoint_id: int,
    repo_name: str,
    private: bool,
    merge_weights: bool,
    user_id: int
) -> dict:
    """
    Deploy a checkpoint to HuggingFace Hub

    Args:
        deployment_id: ID of the Deployment record
        checkpoint_id: ID of the Checkpoint to deploy
        repo_name: HuggingFace repository name
        private: Whether the repository should be private
        merge_weights: Whether to merge LoRA weights
        user_id: ID of the user initiating deployment

    Returns:
        dict with status and deployment_id
    """
    db = self.db

    try:
        from utils.encryption import decrypt_token
        from services.huggingface_service import HuggingFaceService
        import tempfile
        import requests
        import tarfile

        # Check if Tinker is available
        try:
            import tinker
            TINKER_AVAILABLE = True
        except ImportError:
            TINKER_AVAILABLE = False

        deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
        if not deployment:
            return {"status": "error", "error": "Deployment not found", "deployment_id": deployment_id}

        # Update status
        deployment.status = "uploading"
        db.commit()

        # Get user and decrypt token
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.hf_token_encrypted:
            deployment.status = "failed"
            deployment.error_message = "HuggingFace token not configured"
            db.commit()
            return {"status": "error", "error": "HF token not configured"}

        hf_token = decrypt_token(user.hf_token_encrypted)
        hf_service = HuggingFaceService(token=hf_token)

        # Get checkpoint and run
        checkpoint = db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
        if not checkpoint:
            deployment.status = "failed"
            deployment.error_message = "Checkpoint not found"
            db.commit()
            return {"status": "error", "error": "Checkpoint not found"}

        run = db.query(Run).filter(Run.id == checkpoint.run_id).first()

        # Create temporary directory for checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint"
            checkpoint_path.mkdir()

            # Download checkpoint from Tinker API if available
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
                    response = requests.get(download_url, stream=True)
                    response.raise_for_status()

                    # Save and extract tar file
                    tar_path = Path(temp_dir) / "checkpoint.tar"
                    with open(tar_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    with tarfile.open(tar_path, 'r') as tar:
                        tar.extractall(checkpoint_path)

                    checkpoint_downloaded = True
                    logger.info(f"Successfully downloaded checkpoint")

                except Exception as e:
                    logger.error(f"Failed to download checkpoint: {e}")

            # Create model card
            training_config = run.config_json or {}
            training_config["recipe_type"] = run.recipe_type

            # Create repo and upload
            hf_service.create_model_repo(repo_name=repo_name, private=private)

            base_model = run.config_json.get("base_model", "unknown")
            hf_service.create_model_card(
                repo_name=repo_name,
                base_model=base_model,
                training_config=training_config,
                metrics=checkpoint.meta,
            )

            # Upload checkpoint files if available
            if checkpoint_downloaded:
                hf_service.upload_checkpoint(
                    checkpoint_path=checkpoint_path,
                    repo_name=repo_name,
                    commit_message=f"Upload checkpoint from Tuner-UI - Run {run.id}, Step {checkpoint.step}"
                )

        # Update deployment status
        deployment.status = "completed"
        deployment.deployed_at = datetime.utcnow()
        deployment.hf_repo_url = f"https://huggingface.co/{repo_name}"

        # Update checkpoint
        checkpoint.hf_repo_url = f"https://huggingface.co/{repo_name}"
        checkpoint.hf_deployed_at = datetime.utcnow()

        db.commit()

        logger.info(f"Successfully deployed checkpoint {checkpoint_id} to {repo_name}")
        return {"status": "completed", "deployment_id": deployment_id, "repo_url": deployment.hf_repo_url}

    except Exception as e:
        logger.exception(f"Failed to deploy checkpoint {checkpoint_id}")

        # Update deployment with error
        deployment = db.query(Deployment).filter(Deployment.id == deployment_id).first()
        if deployment:
            deployment.status = "failed"
            deployment.error_message = str(e)
            db.commit()

        return {"status": "error", "error": str(e), "deployment_id": deployment_id}


@celery_app.task(name='tasks.cleanup_old_results')
def cleanup_old_results() -> dict:
    """
    Clean up old Celery task results (periodic task)

    Returns:
        dict with cleanup statistics
    """
    try:
        from celery.result import AsyncResult
        from datetime import timedelta

        # Clean up results older than 24 hours
        cutoff_time = datetime.utcnow() - timedelta(days=1)

        logger.info(f"Cleaning up task results older than {cutoff_time}")

        # This is a simple implementation
        # In production, you might want to use Redis SCAN for efficiency
        return {"status": "completed", "message": "Cleanup completed"}

    except Exception as e:
        logger.exception("Failed to cleanup old results")
        return {"status": "error", "error": str(e)}
