"""
Recipe Executor
===============

Common execution logic for training recipes.

This module consolidates the duplicated code across different recipe types,
reducing ~450 lines of duplication to a single reusable pattern.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional
from sqlalchemy.orm import Session

from utils.env_utils import setup_training_environment
from models import Run

try:
    from checkpoint_helper import register_checkpoint_from_logs
except Exception:  # pragma: no cover - optional helper in local mode
    async def register_checkpoint_from_logs(*_args, **_kwargs) -> None:
        return None


class RecipeExecutor:
    """
    Manages the common execution pattern for all training recipes.

    This class consolidates the duplicated code that was repeated
    7 times across SFT, DPO, RL, CHAT_SL, DISTILLATION, etc.
    """

    def __init__(self, session: Session, run: Run, logs_path: Path):
        """
        Initialize recipe executor.

        Args:
            session: Database session
            run: Training run object
            logs_path: Path to log file
        """
        self.session = session
        self.run = run
        self.logs_path = logs_path

    async def log(self, message: str) -> None:
        """
        Write a log message.

        Args:
            message: Message to log
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_log, message)

    def _write_log(self, message: str) -> None:
        """
        Write log message to file (synchronous).

        Args:
            message: Message to write
        """
        from utils.text_utils import strip_ansi_codes

        with self.logs_path.open("a", encoding="utf-8") as lf:
            lf.write(strip_ansi_codes(message))

    async def execute_recipe(
        self,
        config_builder: Callable[[Run], Any],
        train_fn: Callable[[Any], Awaitable[None]],
        recipe_name: str,
        enable_monitoring: bool = False,
        dataset_arg: Optional[str] = None,
    ) -> None:
        """
        Execute a training recipe with common error handling and lifecycle.

        This method consolidates the pattern that was repeated 7 times:
        1. Setup environment (API key)
        2. Build config
        3. Start monitoring (optional)
        4. Run training
        5. Mark as completed
        6. Register checkpoints

        Args:
            config_builder: Function to build recipe config
            train_fn: Async training function to execute
            recipe_name: Name of the recipe (for logging)
            enable_monitoring: Whether to enable real-time metrics monitoring
            dataset_arg: Optional dataset argument for config builder

        Raises:
            Exception: If training fails
        """
        try:
            # Step 1: Setup environment
            await self.log(
                f"[RUN {self.run.id}] Starting {recipe_name} training\n"
            )
            setup_training_environment()

            # Step 2: Build configuration
            if dataset_arg is not None:
                config = config_builder(self.run, dataset_arg)
            else:
                config = config_builder(self.run)

            await self.log(
                f"[RUN {self.run.id}] Configuration built successfully\n"
            )

            # Step 3: Start monitoring if enabled
            monitor_task = None
            if enable_monitoring:
                metrics_path = self.logs_path.parent / "logs" / "metrics.jsonl"
                monitor_task = asyncio.create_task(
                    self._monitor_logs_for_metrics(metrics_path)
                )

            # Step 4: Execute training
            await self.log(
                f"[RUN {self.run.id}] Executing {recipe_name} training...\n"
            )

            try:
                await train_fn(config)
            finally:
                # Cancel monitoring if it was started
                if monitor_task:
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass

            # Step 5: Mark as completed
            self.run.progress = 1.0
            self.run.status = "completed"
            self.run.finished_at = datetime.utcnow()
            self.session.add(self.run)
            self.session.commit()

            await self.log(
                f"[RUN {self.run.id}] Training completed successfully\n"
            )

            # Step 6: Register checkpoints
            await self.log(
                f"[RUN {self.run.id}] Registering checkpoints...\n"
            )
            await self._register_checkpoints()

            await self.log(
                f"[RUN {self.run.id}] {recipe_name} training finished successfully\n"
            )

        except Exception as e:
            await self.log(
                f"[RUN {self.run.id}] Training failed: {str(e)}\n"
            )
            import traceback
            await self.log(
                f"[RUN {self.run.id}] Traceback: {traceback.format_exc()}\n"
            )
            raise

    async def _register_checkpoints(self) -> None:
        """
        Register checkpoints from training logs.

        This uses the checkpoint_helper module to parse checkpoints.jsonl
        and register them in the database.
        """
        try:
            await register_checkpoint_from_logs(
                self.session, self.run, self.logs_path
            )
        except Exception as e:
            await self.log(
                f"[RUN {self.run.id}] Checkpoint registration failed: {str(e)}\n"
            )
            # Don't fail the training if checkpoint registration fails

    async def _monitor_logs_for_metrics(self, metrics_path: Path) -> None:
        """
        Monitor log file for metrics and update progress in real-time.

        Args:
            metrics_path: Path to metrics.jsonl file
        """
        import json

        last_position = 0
        metrics = []

        try:
            while True:
                await asyncio.sleep(2)  # Check every 2 seconds

                if self.logs_path.exists():
                    with open(self.logs_path, "r", encoding="utf-8") as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()

                        for line in new_lines:
                            line = line.strip()
                            if not line:
                                continue

                            # Parse metrics from line
                            metric_dict = self._parse_metric_line(line)

                            if metric_dict:
                                metric_entry = metric_dict.copy()
                                metric_entry["timestamp"] = (
                                    datetime.utcnow().isoformat()
                                )

                                # Ensure required fields
                                if "step" not in metric_entry:
                                    metric_entry["step"] = len(metrics) + 1

                                metrics.append(metric_entry)

                                # Update progress in DB
                                if "progress" in metric_entry:
                                    self.run.progress = metric_entry["progress"]
                                    self.session.add(self.run)
                                    self.session.commit()

                                # Write to metrics file
                                if metrics_path:
                                    metrics_path.parent.mkdir(
                                        parents=True, exist_ok=True
                                    )
                                    with open(
                                        metrics_path, "a", encoding="utf-8"
                                    ) as f:
                                        f.write(json.dumps(metric_entry) + "\n")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            await self.log(f"[METRICS-MONITOR] Error: {str(e)}\n")

    def _parse_metric_line(self, line: str) -> dict:
        """
        Parse a log line for metrics.

        Args:
            line: Log line to parse

        Returns:
            Dictionary of parsed metrics, or empty dict if no metrics found
        """
        import re

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
                            metric_dict[key] = int(float(value)) if value else 0
                        elif key in [
                            "loss",
                            "lr",
                            "progress",
                            "train_mean_nll",
                            "learning_rate",
                        ]:
                            metric_dict[key] = float(value) if value else 0.0
                        else:
                            metric_dict[key] = value
            except Exception:
                pass

        # Also parse general patterns
        elif "loss" in line.lower() or "step" in line.lower():
            loss_match = re.search(r"loss[:=]\s*([0-9.]+)", line, re.IGNORECASE)
            if loss_match:
                metric_dict["train_mean_nll"] = float(loss_match.group(1))

            step_match = re.search(r"step[:=]\s*([0-9]+)", line, re.IGNORECASE)
            if step_match:
                metric_dict["step"] = int(step_match.group(1))

            lr_match = re.search(
                r"learning.rate[:=]\s*([0-9.e-]+)", line, re.IGNORECASE
            )
            if lr_match:
                metric_dict["learning_rate"] = float(lr_match.group(1))

            progress_match = re.search(
                r"progress\s+([0-9.]+)", line, re.IGNORECASE
            )
            if progress_match:
                metric_dict["progress"] = float(progress_match.group(1))

        return metric_dict


def create_recipe_executor(session: Session, run: Run, logs_path: Path) -> RecipeExecutor:
    """
    Factory function to create a RecipeExecutor.

    Args:
        session: Database session
        run: Training run object
        logs_path: Path to log file

    Returns:
        RecipeExecutor instance
    """
    return RecipeExecutor(session, run, logs_path)
