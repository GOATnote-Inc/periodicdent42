"""Add BETE-NET runs table

Revision ID: 002_bete_runs
Revises: 001_add_test_telemetry
Create Date: 2025-10-08 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision = '002_bete_runs'
down_revision = '001_add_test_telemetry'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create bete_runs table for BETE-NET predictions."""
    op.create_table(
        'bete_runs',
        sa.Column('run_id', UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('input_hash', sa.String(64), nullable=False, index=True),
        sa.Column('mp_id', sa.String(32), nullable=True, index=True),
        sa.Column('structure_formula', sa.String(128), nullable=False),
        sa.Column('tc_kelvin', sa.Float, nullable=False),
        sa.Column('tc_std', sa.Float, nullable=False),
        sa.Column('lambda_ep', sa.Float, nullable=False),
        sa.Column('lambda_std', sa.Float, nullable=False),
        sa.Column('omega_log', sa.Float, nullable=False),
        sa.Column('omega_log_std', sa.Float, nullable=False),
        sa.Column('mu_star', sa.Float, nullable=False, server_default='0.10'),
        sa.Column('uncertainty', sa.Float, nullable=True, comment='Overall prediction uncertainty'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now(), nullable=False),
        sa.Column('evidence_path', sa.Text, nullable=True, comment='Path to evidence pack ZIP'),
        sa.Column('user_id', sa.String(128), nullable=True, comment='User or API key'),
        sa.Column('run_type', sa.String(32), nullable=False, server_default='single', comment='single or batch'),
        sa.Column('parent_batch_id', UUID(as_uuid=True), nullable=True, comment='Parent batch run_id'),
        sa.Column('metadata', sa.JSON, nullable=True, comment='Additional metadata (alpha2F data, etc.)'),
    )

    # Create indexes for common queries
    op.create_index('idx_bete_tc_desc', 'bete_runs', ['tc_kelvin'], postgresql_using='btree', postgresql_order_by='desc')
    op.create_index('idx_bete_created', 'bete_runs', ['created_at'], postgresql_using='btree', postgresql_order_by='desc')
    op.create_index('idx_bete_formula', 'bete_runs', ['structure_formula'], postgresql_using='btree')

    # Create materialized view for top superconductors
    op.execute("""
        CREATE MATERIALIZED VIEW top_superconductors AS
        SELECT 
            structure_formula,
            mp_id,
            AVG(tc_kelvin) as avg_tc,
            STDDEV(tc_kelvin) as std_tc,
            AVG(lambda_ep) as avg_lambda,
            COUNT(*) as n_predictions,
            MAX(created_at) as last_predicted
        FROM bete_runs
        WHERE tc_kelvin > 1.0  -- Only viable superconductors
        GROUP BY structure_formula, mp_id
        HAVING COUNT(*) >= 1
        ORDER BY avg_tc DESC;

        CREATE INDEX idx_top_sc_tc ON top_superconductors(avg_tc DESC);
    """)


def downgrade() -> None:
    """Drop BETE-NET tables."""
    op.execute("DROP MATERIALIZED VIEW IF EXISTS top_superconductors;")
    op.drop_table('bete_runs')

