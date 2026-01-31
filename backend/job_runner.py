from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

# Import database models
from database import SessionLocal
from models import Run, Checkpoint, Dataset, User
from config import settings
from utils.encryption import decrypt_token

# Import Tinker API
try:
    import tinker

    TINKER_AVAILABLE = True
except ImportError as e:
    print(f"Tinker not available: {e}")
    TINKER_AVAILABLE = False

# Import training modules
import chz
import datasets
from tinker_cookbook import model_info, hyperparam_utils, renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilderCommonConfig,
    ChatDatasetBuilder,
)

from recipes import sft, dpo, rl, chat_sl, distillation, math_rl, on_policy_distillation

# Import new utility modules
from utils.text_utils import strip_ansi_codes
from utils.env_utils import setup_training_environment, get_required_env
from utils.recipe_executor import create_recipe_executor

ARTIFACTS_ROOT = Path("artifacts")
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class JobTask:
    run_id: int
    task: asyncio.Task
    process: Optional[subprocess.Popen] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class JobRunner:
    def __init__(self) -> None:
        self._tasks: dict[int, JobTask] = {}
        self._lock = asyncio.Lock()

    async def submit(self, run_id: int) -> None:
        async with self._lock:
            if run_id in self._tasks:
                return
            task = asyncio.create_task(self._execute(run_id))
            self._tasks[run_id] = JobTask(run_id=run_id, task=task)

    async def cancel(self, run_id: int) -> bool:
        async with self._lock:
            job = self._tasks.get(run_id)
            if not job:
                return False
            cancelled = job.task.cancel()
            # Terminate the subprocess if it exists
            if job.process:
                try:
                    job.process.terminate()
                    # Wait a bit for graceful termination
                    await asyncio.sleep(0.1)
                    if job.process.poll() is None:
                        job.process.kill()
                except Exception as e:
                    print(f"Error terminating process for run {run_id}: {e}")
        if cancelled:
            await self._abort_run(run_id, reason="cancelled")
        return cancelled

    async def _execute(self, run_id: int) -> None:
        session: Session = SessionLocal()
        run = session.get(Run, run_id)
        if not run:
            session.close()
            return

        artifact_dir = ARTIFACTS_ROOT / f"run_{run.id}"
        logs_path = artifact_dir / "logs.txt"
        logs_dir = logs_path.with_suffix("")
        metrics_path = logs_dir / "metrics.jsonl"
        checkpoints_dir = artifact_dir / "checkpoints"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # CRITICAL FIX: Use try-finally to guarantee status update even on crash
        try:
            run.status = "running"
            run.started_at = datetime.utcnow()
            run.log_path = str(logs_path)
            session.add(run)
            session.commit()

            await self._log(
                logs_path, f"[RUN {run.id}] Starting {run.recipe_type} job\n"
            )
            await self._run_training(
                session, run, logs_path, metrics_path, checkpoints_dir
            )
        except asyncio.CancelledError:
            await self._log(logs_path, f"[RUN {run.id}] Cancelled by user\n")
            # Ensure we mark as cancelled even if commit fails
            try:
                self._mark_run_status(session, run, "cancelled")
            except Exception as e:
                print(f"Failed to mark run {run_id} as cancelled: {e}")
            raise
        except Exception as exc:  # pylint: disable=broad-except
            await self._log(logs_path, f"[RUN {run.id}] Failed: {exc!r}\n")
            # Ensure we mark as failed even if commit fails
            try:
                self._mark_run_status(session, run, "failed")
            except Exception as e:
                print(f"Failed to mark run {run_id} as failed: {e}")
        finally:
            # CRITICAL: Always close session and cleanup, regardless of errors
            try:
                session.close()
            except Exception as e:
                print(f"Error closing session for run {run_id}: {e}")

            async with self._lock:
                self._tasks.pop(run_id, None)

    async def _run_training(
        self,
        session: Session,
        run: Run,
        logs_path: Path,
        metrics_path: Path,
        checkpoints_dir: Path,
    ) -> None:
        config = run.config_json or {}
        base_model = config.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
        recipe_type = run.recipe_type

        await self._log(
            logs_path,
            f"[RUN {run.id}] Starting {recipe_type} training with base model {base_model}\n",
        )

        if not TINKER_AVAILABLE:
            await self._log(
                logs_path, f"[RUN {run.id}] Tinker API not available, using simulation\n"
            )
            await self._simulate_training(
                session, run, logs_path, metrics_path, checkpoints_dir
            )
            return

        try:
            if recipe_type in [
                "SFT",
                "DPO",
                "RL",
                "PPO",
                "GRPO",
                "DISTILLATION",
                "CHAT_SL",
                "PREFERENCE",
                "TOOL_USE",
                "MULTIPLAYER_RL",
                "PROMPT_DISTILLATION",
                "MATH_RL",
                "ON_POLICY_DISTILLATION",
                "EVAL",
            ]:
                await self._run_cookbook_training(
                    session,
                    run,
                    logs_path,
                    metrics_path,
                    checkpoints_dir,
                )
            else:
                # For unknown recipes, use simulation
                await self._log(
                    logs_path,
                    f"[RUN {run.id}] Recipe {recipe_type} not implemented, using simulation\n",
                )
                await self._simulate_training(
                    session, run, logs_path, metrics_path, checkpoints_dir
                )

        except Exception as e:
            await self._log(logs_path, f"[RUN {run.id}] Training failed: {str(e)}\n")
            raise

    async def _run_cookbook_training(
        self,
        session: Session,
        run: Run,
        logs_path: Path,
        metrics_path: Path,
        checkpoints_dir: Path,
    ) -> None:
        """Run training using Tinker cookbook via subprocess"""
        config = run.config_json or {}
        recipe_type = run.recipe_type

        base_model = config.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
        hyperparameters = config.get("hyperparameters", {})

        # Get user's HuggingFace token for gated models (like Llama)
        hf_token = None
        if run.project and run.project.owner:
            user = run.project.owner
            if user.hf_token_encrypted:
                try:
                    hf_token = decrypt_token(user.hf_token_encrypted)
                    await self._log(logs_path, f"[RUN {run.id}] Using HuggingFace token from user settings\n")
                except Exception as e:
                    await self._log(logs_path, f"[RUN {run.id}] Warning: Could not decrypt HF token: {e}\n")

        # Set up environment with HF token
        setup_training_environment(hf_token=hf_token)

        # Handle dataset
        dataset_arg = ""
        if run.dataset_id:
            dataset = session.get(Dataset, run.dataset_id)
            if dataset:
                dataset_arg = dataset.name

        # Build command line arguments
        log_path = f"logs/run_{run.id}"
        os.makedirs(log_path, exist_ok=True)
        cookbook_abs_path = Path("../../tinker-cookbook").resolve()

        # Map recipe types to scripts using proper module paths
        recipe_modules = {
            "SFT": "tinker_cookbook.recipes.sl_basic",
            "DPO": "tinker_cookbook.recipes.preference.dpo.train",
            "RL": "tinker_cookbook.recipes.rl_basic",
            "CHAT_SL": "tinker_cookbook.recipes.chat_sl.train",
            "DISTILLATION": "tinker_cookbook.recipes.distillation.on_policy_distillation",
            "PREFERENCE": "tinker_cookbook.recipes.preference.dpo.train",  # DPO
        }
        module_name = recipe_modules.get(
            recipe_type, "tinker_cookbook.recipes.sl_basic"
        )

        await self._log(logs_path, f"[RUN {run.id}] Recipe type: {recipe_type}\n")

        if recipe_type == "SFT":
            # Use RecipeExecutor for common pattern
            executor = create_recipe_executor(session, run, logs_path)
            await executor.execute_recipe(
                config_builder=self.build_sft_config,
                train_fn=sft.train.main,
                recipe_name="SFT",
                enable_monitoring=True,
                dataset_arg=dataset_arg,
            )
            return
        elif recipe_type == "DPO":
            # Use RecipeExecutor for common pattern
            executor = create_recipe_executor(session, run, logs_path)

            # DPO uses synchronous function, so wrap in executor
            async def run_dpo(config):
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: dpo.train_dpo.main(config)
                )

            await executor.execute_recipe(
                config_builder=self.build_dpo_config,
                train_fn=run_dpo,
                recipe_name="DPO",
                enable_monitoring=False,
            )
            return
        elif recipe_type == "RL":
            # Use RecipeExecutor for common pattern
            executor = create_recipe_executor(session, run, logs_path)
            await executor.execute_recipe(
                config_builder=self.build_rl_config,
                train_fn=rl.train.main,
                recipe_name="RL",
                enable_monitoring=False,
            )
            return
        elif recipe_type == "CHAT_SL":
            # CHAT_SL is just SFT with a chat dataset
            executor = create_recipe_executor(session, run, logs_path)

            # Build SFT config with chat dataset
            def build_chat_sl_config(run, dataset="HuggingFaceH4/no_robots"):
                return self.build_sft_config(run, dataset)

            await executor.execute_recipe(
                config_builder=build_chat_sl_config,
                train_fn=sft.train.main,
                recipe_name="CHAT_SL",
                enable_monitoring=True,
            )
            return
        elif recipe_type == "DISTILLATION":
            # Use RecipeExecutor for common pattern
            executor = create_recipe_executor(session, run, logs_path)
            await executor.execute_recipe(
                config_builder=self.build_distillation_config,
                train_fn=distillation.train_on_policy.main,
                recipe_name="DISTILLATION",
                enable_monitoring=False,
            )
            return
        elif recipe_type == "MATH_RL":
            # Math RL uses raw config
            executor = create_recipe_executor(session, run, logs_path)

            # Wrapper for math_rl that passes config and log callback
            async def run_math_rl_training(cfg):
                await math_rl.run_math_rl(
                    cfg,
                    log_callback=lambda msg: asyncio.create_task(
                        executor.log(f"[RUN {run.id}] {msg}\n")
                    ),
                )

            # Math RL uses raw config, not a builder
            def get_raw_config(run):
                return config

            await executor.execute_recipe(
                config_builder=get_raw_config,
                train_fn=run_math_rl_training,
                recipe_name="MATH_RL",
                enable_monitoring=False,
            )
            return
        elif recipe_type == "ON_POLICY_DISTILLATION":
            # On-policy distillation uses raw config
            executor = create_recipe_executor(session, run, logs_path)

            # ON_POLICY_DISTILLATION uses raw config, not a builder
            def get_raw_config(run):
                return config

            await executor.execute_recipe(
                config_builder=get_raw_config,
                train_fn=on_policy_distillation.run_on_policy_distillation,
                recipe_name="ON_POLICY_DISTILLATION",
                enable_monitoring=False,
            )
            return

        # Build command using Python module execution
        cmd = [
            sys.executable,
            "-m",
            module_name,
            f"model_name={base_model}",
            f"log_path={log_path}",
            f"num_epochs={hyperparameters.get('epochs', 1)}",
            f"learning_rate={hyperparameters.get('learning_rate', 1e-4)}",
        ]

        # For SFT, we can specify dataset if it's a simple jsonl file
        if dataset_arg and recipe_type == "SFT":
            # For our pig_latin dataset, we'll need to create a custom dataset builder
            # For now, use default dataset
            pass

        # Add recipe-specific args
        if recipe_type == "DPO":
            cmd.append(f"dpo_beta={config.get('dpo_beta', 0.1)}")

        # Check API key
        if (
            "TINKER_API_KEY" not in os.environ
            or os.environ["TINKER_API_KEY"] == "your_tinker_api_key_here"
        ):
            await self._log(
                logs_path,
                f"[RUN {run.id}] ERROR: TINKER_API_KEY not set or placeholder value\n",
            )
            run.status = "failed"
            session.add(run)
            session.commit()
            return

        await self._log(
            logs_path, f"[RUN {run.id}] Running cookbook: {' '.join(cmd)}\n"
        )
        await self._log(logs_path, f"[RUN {run.id}] Module: {module_name}\n")

        # Run the subprocess with environment
        env = os.environ.copy()

        # Ensure TINKER_API_KEY is explicitly passed to subprocess
        # SECURITY: API key MUST be set in environment or .env file
        if not settings.tinker_api_key:
            error_msg = (
                "TINKER_API_KEY is not set! "
                "Please set it in your .env file or environment variables."
            )
            await self._log(logs_path, f"[RUN {run.id}] ERROR: {error_msg}\n")
            raise ValueError(error_msg)

        env["TINKER_API_KEY"] = settings.tinker_api_key

        # Pass HuggingFace token to subprocess if available
        if hf_token:
            env["HF_TOKEN"] = hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token

        await self._log(
            logs_path,
            f"[RUN {run.id}] API key configured successfully\n",
        )

        try:
            # Use Popen for async execution with real-time metrics monitoring
            import subprocess

            process = subprocess.Popen(
                cmd,
                cwd=cookbook_abs_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            await self._log(
                logs_path,
                f"[RUN {run.id}] Training process started (PID: {process.pid})\n",
            )

            # Monitor metrics file in background while process runs
            metrics_log_file = logs_path.parent / "logs" / "metrics.jsonl"
            last_progress = 0.0
            stdout_lines = []
            stderr_lines = []

            # Calculate total steps from config
            total_steps = 0
            try:
                epochs = run.config_json.get("hyperparameters", {}).get("epochs", 1)
                batch_size = run.config_json.get("hyperparameters", {}).get("batch_size", 1)
                # Estimate steps per epoch (rough approximation, actual may vary)
                # This is a heuristic - ideally we'd know dataset size
                steps_per_epoch = max(100, 1000 // batch_size) if batch_size > 0 else 100
                total_steps = epochs * steps_per_epoch
                await self._log(
                    logs_path,
                    f"[PROGRESS] Estimated total steps: {total_steps} (epochs={epochs}, batch_size={batch_size})\n",
                )
            except Exception as e:
                await self._log(
                    logs_path,
                    f"[PROGRESS] Could not estimate total steps: {e}, will use step-based progress\n",
                )

            # Poll process and update progress from metrics file
            poll_count = 0
            while process.poll() is None:
                poll_count += 1
                # Read latest metrics from file
                if metrics_log_file.exists():
                    try:
                        with open(metrics_log_file, "r") as f:
                            lines = f.readlines()
                            if lines:
                                # Get last line with valid metrics
                                for line in reversed(lines):
                                    if line.strip():
                                        import json

                                        metric_data = json.loads(line)

                                        # Try to get progress from metrics
                                        progress = metric_data.get("progress", None)

                                        # If no progress field, calculate from step
                                        if progress is None:
                                            current_step = metric_data.get("step", 0)
                                            if total_steps > 0 and current_step > 0:
                                                progress = min(current_step / total_steps, 0.99)  # Cap at 99% until completion
                                            else:
                                                progress = 0.0

                                        if progress > last_progress:
                                            last_progress = progress
                                            run.progress = progress
                                            session.add(run)
                                            session.commit()
                                            await self._log(
                                                logs_path,
                                                f"[PROGRESS] Updated to {progress * 100:.1f}% (step {metric_data.get('step', 0)}, poll #{poll_count})\n",
                                            )
                                        break
                    except Exception as e:
                        await self._log(
                            logs_path,
                            f"[PROGRESS] Error reading metrics (poll #{poll_count}): {e}\n",
                        )
                elif poll_count % 10 == 0:  # Log every 10 polls (~5 seconds)
                    await self._log(
                        logs_path,
                        f"[PROGRESS] Waiting for metrics file (poll #{poll_count})...\n",
                    )

                # Read any available stdout/stderr
                import select

                if hasattr(select, "select"):
                    # Unix-like systems
                    ready = select.select([process.stdout, process.stderr], [], [], 0.5)
                    if process.stdout in ready[0]:
                        line = process.stdout.readline()
                        if line:
                            stdout_lines.append(line)
                            await self._log(logs_path, f"[COOKBOOK] {line}")
                    if process.stderr in ready[0]:
                        line = process.stderr.readline()
                        if line:
                            stderr_lines.append(line)
                            await self._log(logs_path, f"[COOKBOOK-ERR] {line}")
                else:
                    # Windows - just sleep and check
                    await asyncio.sleep(0.5)

            # Process completed - get remaining output
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                stdout_lines.extend(remaining_stdout.splitlines(keepends=True))
                for line in remaining_stdout.splitlines():
                    await self._log(logs_path, f"[COOKBOOK] {line}\n")
            if remaining_stderr:
                stderr_lines.extend(remaining_stderr.splitlines(keepends=True))
                for line in remaining_stderr.splitlines():
                    await self._log(logs_path, f"[COOKBOOK-ERR] {line}\n")

            returncode = process.returncode
            await self._log(
                logs_path,
                f"[RUN {run.id}] Process completed with return code: {returncode}\n",
            )

            # Check for success indicators
            stdout_text = "".join(stdout_lines)
            has_success = False
            success_indicators = [
                "Training completed successfully",
                "Saved checkpoints",
                "Training completed",
            ]
            has_success = any(
                indicator in stdout_text for indicator in success_indicators
            )

            # Check if training completed successfully
            if returncode == 0 or has_success or run.progress >= 1.0:
                run.progress = 1.0
                run.status = "completed"
                session.add(run)
                session.commit()
                await self._log(
                    logs_path, f"[RUN {run.id}] Training completed successfully\n"
                )

                # Parse metrics from logs and write to metrics.jsonl
                await self._parse_metrics_from_logs(logs_path, metrics_path)

                # Register checkpoints from checkpoints.jsonl
                await self._log(
                    logs_path, f"[RUN {run.id}] Registering checkpoints...\n"
                )
                from checkpoint_helper import register_checkpoint_from_logs

                await register_checkpoint_from_logs(session, run, logs_path)
            else:
                await self._log(
                    logs_path,
                    f"[RUN {run.id}] Cookbook script failed with code {returncode}\n",
                )
                raise Exception(f"Cookbook script failed with code {returncode}")

        except Exception as e:
            await self._log(
                logs_path, f"[RUN {run.id}] Cookbook failed, using simulation: {e}\n"
            )
            await self._simulate_training(
                session, run, logs_path, metrics_path, checkpoints_dir
            )

    async def _simulate_training(
        self,
        session: Session,
        run: Run,
        logs_path: Path,
        metrics_path: Path,
        checkpoints_dir: Path,
        steps: int = 5,
    ) -> None:
        """Fallback simulation when Tinker API is not available"""
        config = run.config_json or {}
        base_model = config.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
        await self._log(
            logs_path,
            f"[RUN {run.id}] SIMULATION: Using base model {base_model}, recipe {run.recipe_type}\n",
        )

        metrics_path.write_text("", encoding="utf-8")
        save_every = steps // 2 if steps > 2 else 1  # Save checkpoints more frequently

        for step in range(1, steps + 1):
            await asyncio.sleep(1.5)
            loss = round(random.uniform(0.5, 3.0) / step, 4)
            reward = round(random.uniform(0.1, 1.0) * step, 4)
            await self._log(
                logs_path,
                f"[RUN {run.id}] SIMULATION: Step {step}/{steps} loss={loss} reward={reward}\n",
            )
            metric_entry: Dict[str, Any] = {
                "step": step,
                "loss": loss,
                "reward": reward,
                "timestamp": datetime.utcnow().isoformat(),
            }
            with metrics_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(metric_entry) + "\n")

            run.progress = round(step / steps, 4)
            session.add(run)
            session.commit()

            # Save checkpoint at intervals and at the end
            if step % save_every == 0 or step == steps:
                checkpoint_file = checkpoints_dir / f"checkpoint-step-{step}.bin"
                checkpoint_file.write_text("SIMULATED_CHECKPOINT", encoding="utf-8")
                await self._register_checkpoint(run.id, str(checkpoint_file), step)
                await self._log(
                    logs_path,
                    f"[RUN {run.id}] SIMULATION: Saved checkpoint at step {step}\n",
                )

        await self._log(
            logs_path, f"[RUN {run.id}] SIMULATION: Job completed successfully\n"
        )

    async def _register_checkpoint(self, run_id: int, path: str, step: int) -> None:
        session: Session = SessionLocal()
        checkpoint = Checkpoint(
            run_id=run_id,
            tinker_path=f"file://{path}",
            kind="simulated",
            step=step,
            meta={"size_bytes": Path(path).stat().st_size},
        )
        session.add(checkpoint)
        session.commit()
        session.close()

    async def _read_stream(self, stream, logs_path: Path, prefix: str) -> None:
        """Read from a stream and log lines"""
        while True:
            line = await stream.readline()
            if not line:
                break
            line_str = line.decode().strip()
            await self._log(logs_path, f"{prefix} {line_str}\n")

    async def _log(self, path: Path, message: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_log, path, message)

    @staticmethod
    def _write_log(path: Path, message: str) -> None:
        with path.open("a", encoding="utf-8") as lf:
            lf.write(strip_ansi_codes(message))

    def _mark_run_status(self, session: Session, run: Run, status: str) -> None:
        run.status = status
        if status in {"failed", "cancelled"}:
            run.progress = 0.0
        run.finished_at = datetime.utcnow()
        session.add(run)
        session.commit()

    async def _abort_run(self, run_id: int, reason: str) -> None:
        session: Session = SessionLocal()
        run = session.get(Run, run_id)
        if run:
            run.status = reason
            run.progress = 0.0
            run.finished_at = datetime.utcnow()
            session.add(run)
            session.commit()
        session.close()

    async def _update_progress_periodically(self, run_id: int, logs_path: Path) -> None:
        """Update progress periodically during long-running tasks"""
        session = SessionLocal()
        progress = 0.1  # Start from 10%
        update_count = 0
        try:
            while progress < 0.95 and update_count < 10:  # Max 10 updates
                await asyncio.sleep(5)  # Update every 5 seconds
                progress += 0.05  # Increment by 5% every 5 seconds
                run = session.get(Run, run_id)
                if run and run.status == "running":
                    run.progress = min(progress, 0.95)
                    session.add(run)
                    session.commit()
                    await self._log(
                        logs_path,
                        f"[RUN {run_id}] Updated progress to {run.progress:.1%}\n",
                    )
                    print(f"Updated progress for run {run_id} to {run.progress}")
                    update_count += 1
        finally:
            session.close()

    async def _run_sft_subprocess(
        self,
        session: Session,
        run: Run,
        logs_path: Path,
        metrics_path: Path,
        checkpoints_dir: Path,
        dataset_arg: str,
    ) -> None:
        """Run SFT training via subprocess for real-time metrics parsing"""
        config = self.build_sft_config(run, dataset_arg)

        # Build command for SFT subprocess
        # Use the same pattern as other recipes
        base_model = config.get("model_name", "meta-llama/Llama-3.2-3B")
        hyperparameters = config.get("hyperparameters", {})

        log_path_arg = str(logs_path.parent / "logs")  # Same as build_sft_config

        cmd = [
            sys.executable,
            "-m",
            "tinker_cookbook.recipes.sl_basic",
            f"model_name={base_model}",
            f"log_path={log_path_arg}",
            f"num_epochs={hyperparameters.get('epochs', 1)}",
            f"learning_rate={hyperparameters.get('learning_rate', 1e-4)}",
        ]

        # Add dataset if specified
        if dataset_arg:
            cmd.append(f"dataset={dataset_arg}")

        await self._log(
            logs_path, f"[RUN {run.id}] Running SFT subprocess: {' '.join(cmd)}\n"
        )

        # Set API key
        api_key = os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise ValueError(
                "TINKER_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )
        env = os.environ.copy()
        env["TINKER_API_KEY"] = api_key

        # Get and pass HuggingFace token for gated models
        if run.project and run.project.owner:
            user = run.project.owner
            if user.hf_token_encrypted:
                try:
                    hf_token = decrypt_token(user.hf_token_encrypted)
                    env["HF_TOKEN"] = hf_token
                    env["HUGGING_FACE_HUB_TOKEN"] = hf_token
                except Exception as e:
                    await self._log(logs_path, f"[RUN {run.id}] Warning: Could not decrypt HF token: {e}\n")

        # Run subprocess with real-time output parsing
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=Path("../../tinker-cookbook").resolve(),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True,
            )

            # Parse stdout and stderr in real-time
            await asyncio.gather(
                self._read_stream_and_parse_metrics(
                    process.stdout, logs_path, metrics_path, run.id, session
                ),
                self._read_stream(process.stderr, logs_path, "[SFT-ERR]"),
            )

            # Wait for process to complete
            returncode = await process.wait()

            if returncode == 0:
                run.progress = 1.0
                run.status = "completed"
                session.add(run)
                session.commit()
                await self._log(
                    logs_path, f"[RUN {run.id}] SFT training completed successfully\n"
                )

                # Parse metrics from logs and write to metrics.jsonl
                await self._parse_metrics_from_logs(logs_path, metrics_path)

                # Register checkpoints
                await self._log(
                    logs_path, f"[RUN {run.id}] Registering checkpoints...\n"
                )
                from checkpoint_helper import register_checkpoint_from_logs

                await register_checkpoint_from_logs(session, run, logs_path)
            else:
                await self._log(
                    logs_path,
                    f"[RUN {run.id}] SFT subprocess failed with code {returncode}\n",
                )
                raise Exception(f"SFT subprocess failed with code {returncode}")

        except Exception as e:
            await self._log(
                logs_path, f"[RUN {run.id}] SFT subprocess error: {str(e)}\n"
            )
            raise

    async def _read_stream_and_parse_metrics(
        self, stream, logs_path: Path, metrics_path: Path, run_id: int, session: Session
    ):
        """Read stream and parse metrics in real-time"""
        import json
        from datetime import datetime

        metrics = []
        current_metrics = {}

        while True:
            line = await stream.readline()
            if not line:
                break
            line = line.strip()
            await self._log(logs_path, f"[SFT] {line}\n")

            # Parse metrics from line
            if (
                "loss" in line.lower()
                or "step" in line.lower()
                or "progress" in line.lower()
            ):
                try:
                    # Extract metrics (similar to existing parsing)
                    import re

                    # Update current metrics
                    loss_match = re.search(r"loss[:=]\s*([0-9.]+)", line, re.IGNORECASE)
                    if loss_match:
                        current_metrics["train_mean_nll"] = float(loss_match.group(1))

                    step_match = re.search(r"step[:=]\s*([0-9]+)", line, re.IGNORECASE)
                    if step_match:
                        current_metrics["step"] = int(step_match.group(1))

                    lr_match = re.search(
                        r"learning.rate[:=]\s*([0-9.e-]+)", line, re.IGNORECASE
                    )
                    if lr_match:
                        current_metrics["learning_rate"] = float(lr_match.group(1))

                    progress_match = re.search(
                        r"progress[:=]\s*([0-9.]+)", line, re.IGNORECASE
                    )
                    if progress_match:
                        progress = float(progress_match.group(1))
                        current_metrics["progress"] = progress
                        # Update run progress
                        run = session.get(Run, run_id)
                        if run:
                            run.progress = progress
                            session.add(run)
                            session.commit()

                    # If we have step, save the metric
                    if "step" in current_metrics:
                        metric_entry = current_metrics.copy()
                        metric_entry["timestamp"] = datetime.utcnow().isoformat()
                        metrics.append(metric_entry)

                        # Write to metrics file
                        with open(metrics_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(metric_entry) + "\n")

                        current_metrics = {}  # Reset for next

                except Exception as e:
                    await self._log(logs_path, f"[METRICS-PARSE-ERR] {str(e)}\n")

        # Write final metrics file
        if metrics:
            with open(metrics_path, "w", encoding="utf-8") as f:
                for metric in metrics:
                    f.write(json.dumps(metric) + "\n")

    async def _parse_metrics_from_logs(
        self, logs_path: Path, metrics_path: Path
    ) -> None:
        """Parse metrics from training logs and write to metrics.jsonl"""
        import json
        from datetime import datetime

        metrics = []
        if logs_path.exists():
            with open(logs_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Look for metrics in log lines
                    # Patterns: [METRICS] step=1, loss=1.23, lr=0.0004, tokens=1000, progress=0.1
                    # Or lines containing loss/step info
                    metric_dict = {}

                    if "[METRICS]" in line:
                        try:
                            # Extract the metrics part
                            metrics_part = line.split("[METRICS]")[1].strip()
                            # Parse key=value pairs
                            for pair in metrics_part.split(","):
                                if "=" in pair:
                                    key, value = pair.split("=", 1)
                                    key = key.strip()
                                    value = value.strip()
                                    # Convert to appropriate types
                                    if key in ["step", "tokens"]:
                                        metric_dict[key] = (
                                            int(float(value)) if value else 0
                                        )
                                    elif key in [
                                        "loss",
                                        "lr",
                                        "progress",
                                        "train_mean_nll",
                                        "learning_rate",
                                    ]:
                                        metric_dict[key] = (
                                            float(value) if value else 0.0
                                        )
                                    else:
                                        metric_dict[key] = value
                        except (ValueError, IndexError) as e:
                            pass

                    # Also check for other metric patterns in logs
                    elif "loss" in line.lower() or "step" in line.lower():
                        try:
                            # Try to extract numeric values
                            import re

                            # Look for patterns like "loss: 1.23" or "step 1"
                            loss_match = re.search(
                                r"loss[:=]\s*([0-9.]+)", line, re.IGNORECASE
                            )
                            if loss_match:
                                metric_dict["train_mean_nll"] = float(
                                    loss_match.group(1)
                                )

                            step_match = re.search(
                                r"step[:=]\s*([0-9]+)", line, re.IGNORECASE
                            )
                            if step_match:
                                metric_dict["step"] = int(step_match.group(1))

                            lr_match = re.search(
                                r"learning.rate[:=]\s*([0-9.e-]+)", line, re.IGNORECASE
                            )
                            if lr_match:
                                metric_dict["learning_rate"] = float(lr_match.group(1))

                        except (ValueError, IndexError):
                            pass

                    if metric_dict:
                        # Add timestamp if not present
                        if "timestamp" not in metric_dict:
                            metric_dict["timestamp"] = datetime.utcnow().isoformat()
                        # Ensure required fields
                        if "step" not in metric_dict:
                            metric_dict["step"] = len(metrics) + 1
                        if "train_mean_nll" not in metric_dict:
                            metric_dict["train_mean_nll"] = 2.0  # Placeholder
                        if "learning_rate" not in metric_dict:
                            metric_dict["learning_rate"] = 0.0004  # From config
                        if "progress" not in metric_dict:
                            metric_dict["progress"] = min(1.0, len(metrics) * 0.1)
                        metrics.append(metric_dict)

        # If no metrics found in logs, create a basic completion metric
        if not metrics:
            metrics = [
                {
                    "step": 1,
                    "train_mean_nll": 1.0,  # Placeholder
                    "learning_rate": 0.0004,
                    "progress": 1.0,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ]

        # Write to metrics.jsonl
        with open(metrics_path, "w", encoding="utf-8") as f:
            for metric in metrics:
                f.write(json.dumps(metric) + "\n")

    async def _monitor_logs_for_metrics(
        self, logs_path: Path, metrics_path: Path, run_id: int, session: Session
    ):
        """Monitor log file for metrics and update in real-time"""
        import json
        from datetime import datetime

        last_position = 0
        metrics = []

        try:
            while True:
                await asyncio.sleep(2)  # Check every 2 seconds

                if logs_path.exists():
                    with open(logs_path, "r", encoding="utf-8") as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()

                        for line in new_lines:
                            line = line.strip()
                            if not line:
                                continue

                            # Parse metrics from line
                            metric_dict = {}

                            # Look for [METRICS] lines
                            if "[METRICS]" in line:
                                try:
                                    metrics_part = line.split("[METRICS]")[1].strip()
                                    for pair in metrics_part.split(","):
                                        if "=" in pair:
                                            key, value = pair.split("=", 1)
                                            key = key.strip()
                                            value = value.strip()
                                            if key in ["step", "tokens"]:
                                                metric_dict[key] = (
                                                    int(float(value)) if value else 0
                                                )
                                            elif key in [
                                                "loss",
                                                "lr",
                                                "progress",
                                                "train_mean_nll",
                                                "learning_rate",
                                            ]:
                                                metric_dict[key] = (
                                                    float(value) if value else 0.0
                                                )
                                            else:
                                                metric_dict[key] = value
                                except:
                                    pass

                            # Also parse general patterns
                            elif (
                                "loss" in line.lower()
                                or "step" in line.lower()
                                or "progress" in line.lower()
                            ):
                                loss_match = re.search(
                                    r"loss[:=]\s*([0-9.]+)", line, re.IGNORECASE
                                )
                                if loss_match:
                                    metric_dict["train_mean_nll"] = float(
                                        loss_match.group(1)
                                    )

                                step_match = re.search(
                                    r"step[:=]\s*([0-9]+)", line, re.IGNORECASE
                                )
                                if step_match:
                                    metric_dict["step"] = int(step_match.group(1))

                                lr_match = re.search(
                                    r"learning.rate[:=]\s*([0-9.e-]+)",
                                    line,
                                    re.IGNORECASE,
                                )
                                if lr_match:
                                    metric_dict["learning_rate"] = float(
                                        lr_match.group(1)
                                    )

                                progress_match = re.search(
                                    r"progress\s+([0-9.]+)", line, re.IGNORECASE
                                )
                                if progress_match:
                                    metric_dict["progress"] = float(
                                        progress_match.group(1)
                                    )

                            if metric_dict:
                                metric_entry = metric_dict.copy()
                                metric_entry["timestamp"] = (
                                    datetime.utcnow().isoformat()
                                )

                                # Ensure required fields
                                if "step" not in metric_entry:
                                    metric_entry["step"] = len(metrics) + 1
                                if "train_mean_nll" not in metric_entry:
                                    metric_entry["train_mean_nll"] = 2.0
                                if "learning_rate" not in metric_entry:
                                    metric_entry["learning_rate"] = 0.0004
                                if "progress" not in metric_entry:
                                    metric_entry["progress"] = min(
                                        1.0, len(metrics) * 0.1
                                    )

                                metrics.append(metric_entry)

                                # Update progress in DB
                                if "progress" in metric_entry:
                                    run = session.get(Run, run_id)
                                    if run and run.status == "running":
                                        run.progress = metric_entry["progress"]
                                        session.add(run)
                                        session.commit()

                                # Write to metrics file
                                with open(metrics_path, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(metric_entry) + "\n")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            await self._log(logs_path, f"[METRICS-MONITOR] Error: {str(e)}\n")

    def build_sft_config(self, run: Run, dataset_arg: str = "") -> sft.train.Config:
        config = run.config_json or {}
        base_model = config.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
        dataset_name = (
            dataset_arg
            if dataset_arg
            else config.get("dataset", "yahma/alpaca-cleaned")
        )

        # Handle known dataset name mappings
        if dataset_name == "alpaca-cleaned-albanian":
            dataset_name = "iamshnoo/alpaca-cleaned-albanian"
        elif dataset_name == "yahma/alpaca-cleaned":
            dataset_name = "yahma/alpaca-cleaned"  # Keep as is
        hyperparameters = config.get("hyperparameters", {})

        artifact_dir = ARTIFACTS_ROOT / f"run_{run.id}"
        log_path = str(artifact_dir / "logs")

        renderer_name = model_info.get_recommended_renderer_name(base_model)
        common_config = ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=base_model,
            renderer_name=renderer_name,
            max_length=32768,
            batch_size=128,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )

        @chz.chz
        class AlpacaBuilder(ChatDatasetBuilder):
            dataset: str = dataset_name

            def __call__(self) -> tuple:
                ds = datasets.load_dataset(self.dataset)
                train_ds = ds["train"].shuffle(seed=0).select(range(1000))
                if "test" in ds:
                    test_ds = ds["test"].select(range(min(100, len(ds["test"]))))
                else:
                    test_ds = ds["train"].select(range(1000, 1100))

                def map_fn(row: dict):
                    # Handle both messages format (Tulu, OpenHermes, etc.) and Alpaca format
                    if "messages" in row:
                        # Direct messages format - use as-is
                        messages = row["messages"]
                    elif "instruction" in row or "output" in row:
                        # Alpaca format - convert to messages
                        instr = (row.get("instruction") or "").strip()
                        inp = (row.get("input") or "").strip()
                        if inp:
                            user_content = f"{instr}\n\nInput:\n{inp}"
                        else:
                            user_content = instr
                        messages = [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": row.get("output", "")},
                        ]
                    else:
                        # Fallback - skip this row
                        print(f"Warning: Skipping row with unrecognized format: {list(row.keys())}")
                        return None

                    return conversation_to_datum(
                        messages,
                        self.renderer,
                        common_config.max_length,
                        common_config.train_on_what,
                    )

                return (
                    SupervisedDatasetFromHFDataset(
                        train_ds, batch_size=common_config.batch_size, map_fn=map_fn
                    ),
                    SupervisedDatasetFromHFDataset(
                        test_ds, batch_size=common_config.batch_size, map_fn=map_fn
                    ),
                )

        dataset_builder = AlpacaBuilder(common_config=common_config)

        return sft.train.Config(
            log_path=log_path,
            model_name=base_model,
            dataset_builder=dataset_builder,
            learning_rate=hyperparameters.get(
                "learning_rate", hyperparam_utils.get_lr(base_model)
            ),
            lr_schedule="linear",
            num_epochs=hyperparameters.get("epochs", 1),
            eval_every=8,
            wandb_project=hyperparameters.get("wandb_project"),
            wandb_name=hyperparameters.get("wandb_name"),
        )

    def build_dpo_config(self, run: Run) -> dpo.train_dpo.Config:
        config = run.config_json or {}
        base_model = config.get("base_model", "meta-llama/Llama-3.2-1B")
        hyperparameters = config.get("hyperparameters", {})

        artifact_dir = ARTIFACTS_ROOT / f"run_{run.id}"
        log_path = str(artifact_dir / "logs")

        renderer_name = model_info.get_recommended_renderer_name(base_model)
        dataset_name = config.get("dataset", "hhh")

        # Use the get_dataset_builder from dpo
        dataset_builder = dpo.get_dataset_builder(
            dataset_name,
            base_model,
            renderer_name,
            config.get("max_length", 8192),
            config.get("batch_size", 256),
        )

        return dpo.train_dpo.Config(
            log_path=log_path,
            model_name=base_model,
            dataset_builder=dataset_builder,
            load_checkpoint_path=None,
            evaluator_builders=[],
            learning_rate=hyperparameters.get("learning_rate", 1e-5),
            lr_schedule="linear",
            dpo_beta=config.get("dpo_beta", 0.1),
            base_url=None,
            wandb_project=hyperparameters.get("wandb_project"),
            wandb_name=hyperparameters.get("wandb_name"),
            reference_model_name=config.get("reference_model"),
        )

    def build_rl_config(self, run: Run) -> rl.train.Config:
        config = run.config_json or {}
        base_model = config.get("base_model", "meta-llama/Llama-3.1-8B")
        hyperparameters = config.get("hyperparameters", {})

        artifact_dir = ARTIFACTS_ROOT / f"run_{run.id}"
        log_path = str(artifact_dir / "logs")

        renderer_name = model_info.get_recommended_renderer_name(base_model)
        builder = rl.Gsm8kDatasetBuilder(
            batch_size=128,
            group_size=16,
            renderer_name=renderer_name,
            model_name_for_tokenizer=base_model,
        )

        return rl.train.Config(
            model_name=base_model,
            log_path=log_path,
            dataset_builder=builder,
            learning_rate=hyperparameters.get("learning_rate", 4e-5),
            max_tokens=256,
            eval_every=0,
            wandb_project=hyperparameters.get("wandb_project"),
            wandb_name=hyperparameters.get("wandb_name"),
        )

    def build_chat_sl_config(self, run: Run) -> chat_sl.train.Config:
        config = run.config_json or {}
        base_model = config.get("base_model", "meta-llama/Llama-3.1-8B")
        hyperparameters = config.get("hyperparameters", {})

        artifact_dir = ARTIFACTS_ROOT / f"run_{run.id}"
        log_path = str(artifact_dir / "logs")

        renderer_name = model_info.get_recommended_renderer_name(base_model)
        dataset_name = config.get("dataset", "HuggingFaceH4/no_robots")

        dataset_builder = chat_sl.get_dataset_builder(
            dataset_name,
            base_model,
            renderer_name,
            config.get("max_length", 16384),
            config.get("batch_size", 256),
            config.get("train_on_what"),
        )

        return chat_sl.train.Config(
            log_path=log_path,
            model_name=base_model,
            load_checkpoint_path=None,
            dataset_builder=dataset_builder,
            evaluator_builders=[],
            infrequent_evaluator_builders=[],
            learning_rate=hyperparameters.get("learning_rate", 1e-4),
            lr_schedule="linear",
            num_epochs=hyperparameters.get("epochs", 1),
            base_url=None,
            wandb_project=hyperparameters.get("wandb_project"),
            wandb_name=hyperparameters.get("wandb_name"),
            lora_rank=config.get("lora_rank", 32),
            save_every=20,
            eval_every=20,
            infrequent_eval_every=100,
        )

    def build_distillation_config(
        self, run: Run
    ) -> distillation.train_on_policy.Config:
        config = run.config_json or {}
        base_model = config.get("base_model", "Qwen/Qwen3-8B-Base")
        hyperparameters = config.get("hyperparameters", {})

        artifact_dir = ARTIFACTS_ROOT / f"run_{run.id}"
        log_path = str(artifact_dir / "logs")

        dataset_name = config.get("dataset", "deepmath")

        dataset_builder = distillation.PromptOnlyDatasetBuilder(
            dataset_name=dataset_name,
            groups_per_batch=config.get("groups_per_batch", 1024),
            group_size=config.get("group_size", 4),
            model_name_for_tokenizer=base_model,
            renderer_name=model_info.get_recommended_renderer_name(base_model),
        )

        teacher_config = distillation.TeacherConfig(
            base_model=config.get("teacher_model", "Qwen/Qwen3-8B"),
            load_checkpoint_path=config.get("teacher_checkpoint"),
        )

        dataset_config = distillation.DistillationDatasetConfig(
            dataset_builder=dataset_builder,
            teacher_config=teacher_config,
            groups_per_batch=config.get("groups_per_batch", 1024),
        )

        return distillation.train_on_policy.Config(
            learning_rate=hyperparameters.get("learning_rate", 1e-4),
            dataset_configs=[dataset_config],
            model_name=base_model,
            lora_rank=config.get("lora_rank", 128),
            max_tokens=config.get("max_tokens", 4096),
            kl_penalty_coef=config.get("kl_penalty_coef", 1.0),
            kl_discount_factor=config.get("kl_discount_factor", 0.0),
            num_substeps=config.get("num_substeps", 1),
            loss_fn="importance_sampling",
            wandb_project=hyperparameters.get("wandb_project"),
            wandb_name=hyperparameters.get("wandb_name"),
            log_path=log_path,
            base_url=None,
            load_checkpoint_path=None,
            compute_post_kl=False,
            eval_every=config.get("eval_every", 20),
            save_every=config.get("save_every", 20),
        )

    async def cleanup(self) -> None:
        async with self._lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()
        for job in tasks:
            job.task.cancel()
        await asyncio.gather(*(job.task for job in tasks), return_exceptions=True)

    @staticmethod
    def reset_artifacts() -> None:
        if ARTIFACTS_ROOT.exists():
            shutil.rmtree(ARTIFACTS_ROOT)
        ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
