"""Add celery_task_id to runs table

Revision ID: 002
Revises: 001
Create Date: 2026-01-25 00:01:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add celery_task_id column to runs table
    op.add_column('runs', sa.Column('celery_task_id', sa.String(), nullable=True))
    op.create_index(op.f('ix_runs_celery_task_id'), 'runs', ['celery_task_id'], unique=False)


def downgrade() -> None:
    # Remove celery_task_id column from runs table
    op.drop_index(op.f('ix_runs_celery_task_id'), table_name='runs')
    op.drop_column('runs', 'celery_task_id')
