"""Add test_telemetry table for ML-powered test selection

Revision ID: 001_test_telemetry
Revises: 
Create Date: 2025-10-06 15:00:00.000000

Phase 3 Week 7 Day 5: ML Test Selection Foundation
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


# revision identifiers, used by Alembic.
revision = '001_test_telemetry'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create test_telemetry table for collecting test execution data."""
    op.create_table(
        'test_telemetry',
        sa.Column('id', sa.String(16), primary_key=True, nullable=False,
                  comment='Unique ID: hash(test_name + commit_sha + timestamp)'),
        sa.Column('test_name', sa.String(500), nullable=False, index=True,
                  comment='Full pytest node ID (e.g., tests/test_api.py::test_health)'),
        sa.Column('test_file', sa.String(500), nullable=False,
                  comment='Relative path to test file'),
        
        # Execution results
        sa.Column('duration_ms', sa.Float, nullable=False,
                  comment='Test execution time in milliseconds'),
        sa.Column('passed', sa.Boolean, nullable=False, index=True,
                  comment='Whether test passed (True) or failed (False)'),
        sa.Column('error_message', sa.Text, nullable=True,
                  comment='Error message if test failed'),
        
        # Source control context
        sa.Column('commit_sha', sa.String(40), nullable=False, index=True,
                  comment='Git commit SHA'),
        sa.Column('branch', sa.String(200), nullable=False,
                  comment='Git branch name'),
        sa.Column('changed_files', JSON, nullable=False,
                  comment='List of files changed in this commit'),
        
        # Features for ML model
        sa.Column('lines_added', sa.Integer, default=0,
                  comment='Lines added in changed files'),
        sa.Column('lines_deleted', sa.Integer, default=0,
                  comment='Lines deleted in changed files'),
        sa.Column('files_changed', sa.Integer, default=0,
                  comment='Number of files changed'),
        sa.Column('complexity_delta', sa.Float, default=0.0,
                  comment='Change in cyclomatic complexity'),
        
        # Historical features (computed)
        sa.Column('recent_failure_rate', sa.Float, default=0.0,
                  comment='Failure rate over last 30 days'),
        sa.Column('avg_duration', sa.Float, default=0.0,
                  comment='Average duration from history'),
        sa.Column('days_since_last_change', sa.Integer, default=0,
                  comment='Days since test file was last modified'),
        
        # Metadata
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), nullable=False,
                  comment='Timestamp when record was created'),
        
        # Indexes for performance
        sa.Index('idx_test_telemetry_test_name_created', 'test_name', 'created_at'),
        sa.Index('idx_test_telemetry_commit_sha', 'commit_sha'),
        sa.Index('idx_test_telemetry_passed_created', 'passed', 'created_at'),
    )
    
    print("✅ Created test_telemetry table for ML-powered test selection")


def downgrade() -> None:
    """Drop test_telemetry table."""
    op.drop_table('test_telemetry')
    print("✅ Dropped test_telemetry table")
