"""
Enhanced logging configuration for Tuner-UI using Loguru
Provides structured logging with rotation, retention, and compression
"""
from loguru import logger
import sys
from pathlib import Path
from typing import Optional


def setup_production_logging(log_level: str = "INFO", log_dir: Optional[str] = None) -> None:
    """
    Configure Loguru for production logging

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
    """
    # Remove default handler
    logger.remove()

    log_directory = Path(log_dir) if log_dir else Path("logs")
    log_directory.mkdir(exist_ok=True, parents=True)

    # Console handler with colors (for development and container logs)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler for all logs
    logger.add(
        log_directory / "tuner_ui_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress old logs
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",  # Log everything to file
        enqueue=True,  # Thread-safe
    )

    # Error file handler (errors only)
    logger.add(
        log_directory / "errors_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="90 days",  # Keep error logs longer
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        level="ERROR",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    # JSON file handler for structured logging (optional, useful for log aggregation)
    logger.add(
        log_directory / "tuner_ui_{time:YYYY-MM-DD}.json",
        rotation="00:00",
        retention="7 days",
        compression="zip",
        serialize=True,  # JSON format
        level="INFO",
        enqueue=True,
    )

    logger.info(f"Logging initialized - Level: {log_level}, Directory: {log_directory}")


def get_logger(name: str = __name__):
    """
    Get a logger instance with the specified name

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Convenience function to log API requests
def log_api_request(method: str, path: str, status_code: int, duration_ms: float, user_id: Optional[int] = None):
    """
    Log API request with structured data

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        user_id: Optional user ID
    """
    logger.info(
        f"API Request",
        extra={
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "user_id": user_id,
        }
    )


# Convenience function to log training events
def log_training_event(run_id: int, event: str, details: Optional[dict] = None):
    """
    Log training run event with structured data

    Args:
        run_id: Training run ID
        event: Event name (started, completed, failed, etc.)
        details: Optional additional details
    """
    logger.info(
        f"Training Event: {event}",
        extra={
            "run_id": run_id,
            "event": event,
            **(details or {}),
        }
    )
