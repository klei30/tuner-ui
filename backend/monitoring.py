"""
Monitoring and observability utilities for Tuner-UI
Includes health checks, Sentry integration, and Prometheus metrics
"""
from datetime import datetime
from typing import Dict, Any
import os

from sqlalchemy import text
from sqlalchemy.orm import Session


def check_database_health(db: Session) -> Dict[str, Any]:
    """
    Check database connectivity and health

    Args:
        db: Database session

    Returns:
        dict with status and optional error
    """
    try:
        db.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_redis_health() -> Dict[str, Any]:
    """
    Check Redis connectivity and health

    Returns:
        dict with status and optional error
    """
    try:
        import redis
        from config import settings

        redis_client = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        redis_client.ping()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_celery_workers() -> Dict[str, Any]:
    """
    Check Celery worker availability

    Returns:
        dict with status, worker count, and optional error
    """
    try:
        from celery_app import celery_app

        inspect = celery_app.control.inspect(timeout=2.0)
        stats = inspect.stats()

        if stats:
            worker_count = len(stats)
            return {"status": "healthy", "workers": worker_count, "worker_names": list(stats.keys())}
        else:
            return {"status": "unhealthy", "error": "No workers available", "workers": 0}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "workers": 0}


def check_tinker_api() -> Dict[str, Any]:
    """
    Check Tinker API availability

    Returns:
        dict with status and optional error
    """
    try:
        import tinker

        service_client = tinker.ServiceClient()
        capabilities = service_client.get_server_capabilities()

        return {
            "status": "healthy",
            "models": len(capabilities.supported_models) if hasattr(capabilities, 'supported_models') else 0,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def get_detailed_health_status(db: Session) -> Dict[str, Any]:
    """
    Get comprehensive health check status for all dependencies

    Args:
        db: Database session

    Returns:
        dict with overall status and component statuses
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }

    # Check database
    db_health = check_database_health(db)
    health["components"]["database"] = db_health
    if db_health["status"] != "healthy":
        health["status"] = "unhealthy"

    # Check Redis
    redis_health = check_redis_health()
    health["components"]["redis"] = redis_health
    if redis_health["status"] != "healthy":
        health["status"] = "degraded" if health["status"] == "healthy" else "unhealthy"

    # Check Celery workers
    celery_health = check_celery_workers()
    health["components"]["celery"] = celery_health
    if celery_health["status"] != "healthy":
        health["status"] = "degraded" if health["status"] == "healthy" else "unhealthy"

    # Check Tinker API (optional - degraded if unavailable)
    try:
        tinker_health = check_tinker_api()
        health["components"]["tinker"] = tinker_health
        if tinker_health["status"] != "healthy":
            health["status"] = "degraded" if health["status"] == "healthy" else health["status"]
    except:
        health["components"]["tinker"] = {"status": "unavailable"}

    return health


def setup_sentry():
    """
    Initialize Sentry error tracking if DSN is configured
    """
    from config import settings

    if not settings.sentry_dsn:
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        from sentry_sdk.integrations.celery import CeleryIntegration

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            integrations=[
                FastApiIntegration(),
                SqlalchemyIntegration(),
                CeleryIntegration(),
            ],
            traces_sample_rate=0.1,  # 10% of transactions for performance monitoring
            environment=settings.environment,
            release=f"tuner-ui@{settings.app_version}",
            # Set error sampling
            sample_rate=1.0,  # Capture 100% of errors
            # Performance monitoring
            enable_tracing=True,
        )

        print(f"✓ Sentry initialized for environment: {settings.environment}")
    except Exception as e:
        print(f"✗ Failed to initialize Sentry: {e}")


def setup_prometheus_metrics(app):
    """
    Initialize Prometheus metrics instrumentation

    Args:
        app: FastAPI application instance
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/metrics", "/health", "/nginx-health"],
            env_var_name="ENABLE_METRICS",
            inprogress_name="http_requests_inprogress",
            inprogress_labels=True,
        )

        instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

        print("✓ Prometheus metrics enabled at /metrics")
    except ImportError:
        print("✗ Prometheus instrumentator not available (optional)")
    except Exception as e:
        print(f"✗ Failed to setup Prometheus: {e}")
