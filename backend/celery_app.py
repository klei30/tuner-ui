"""
Celery application configuration for Tuner-UI
Handles distributed background tasks for training jobs
"""
from celery import Celery
import os

# Read Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "tuner_ui",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"]  # Import tasks module
)

# Configure Celery
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_extended=True,

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_track_started=True,
    task_time_limit=86400,  # 24 hours max
    task_soft_time_limit=82800,  # 23 hours soft limit

    # Worker configuration
    worker_prefetch_multiplier=1,  # Only fetch 1 task at a time
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks
    worker_disable_rate_limits=True,

    # Task routing
    task_routes={
        'tasks.run_training_job': {'queue': 'training'},
        'tasks.deploy_to_huggingface': {'queue': 'deployment'},
        'tasks.cancel_training_job': {'queue': 'control'},
    },

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        'master_name': 'mymaster'
    },

    # Broker settings
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
)

# Optional: Configure periodic tasks (Celery Beat)
celery_app.conf.beat_schedule = {
    # Example: Clean up old task results every day
    'cleanup-task-results': {
        'task': 'tasks.cleanup_old_results',
        'schedule': 86400.0,  # 24 hours
    },
}

if __name__ == '__main__':
    celery_app.start()
