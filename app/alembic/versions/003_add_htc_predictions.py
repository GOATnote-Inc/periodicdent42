"""Add HTC predictions table

Revision ID: 003_add_htc_predictions
Revises: 002_add_bete_runs
Create Date: 2025-10-10 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003_htc_predictions'
down_revision = '002_bete_runs'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create htc_predictions table for HTC optimization framework."""
    op.create_table(
        'htc_predictions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('composition', sa.String(), nullable=False),
        sa.Column('reduced_formula', sa.String(), nullable=False),
        sa.Column('structure_info', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        
        # Critical temperature
        sa.Column('tc_predicted', sa.Float(), nullable=False),
        sa.Column('tc_lower_95ci', sa.Float(), nullable=False),
        sa.Column('tc_upper_95ci', sa.Float(), nullable=False),
        sa.Column('tc_uncertainty', sa.Float(), nullable=False),
        
        # Pressure
        sa.Column('pressure_required_gpa', sa.Float(), nullable=False),
        sa.Column('pressure_uncertainty_gpa', sa.Float(), nullable=True, server_default='0.0'),
        
        # Electron-phonon coupling
        sa.Column('lambda_ep', sa.Float(), nullable=False),
        sa.Column('omega_log', sa.Float(), nullable=False),
        sa.Column('mu_star', sa.Float(), nullable=True, server_default='0.13'),
        sa.Column('xi_parameter', sa.Float(), nullable=False),
        
        # Stability
        sa.Column('phonon_stable', sa.String(), nullable=True, server_default='true'),
        sa.Column('thermo_stable', sa.String(), nullable=True, server_default='true'),
        sa.Column('hull_distance_eV', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('imaginary_modes_count', sa.Integer(), nullable=True, server_default='0'),
        
        # Metadata
        sa.Column('prediction_method', sa.String(), nullable=True, server_default='McMillan-Allen-Dynes'),
        sa.Column('confidence_level', sa.String(), nullable=True, server_default='medium'),
        sa.Column('extrapolation_warning', sa.String(), nullable=True, server_default='false'),
        
        # Tracking
        sa.Column('experiment_id', sa.String(), nullable=True),
        sa.Column('created_by', sa.String(), nullable=True, server_default='anonymous'),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for common queries
    op.create_index(
        'ix_htc_predictions_composition',
        'htc_predictions',
        ['composition'],
        unique=False
    )
    op.create_index(
        'ix_htc_predictions_xi_parameter',
        'htc_predictions',
        ['xi_parameter'],
        unique=False
    )
    op.create_index(
        'ix_htc_predictions_experiment_id',
        'htc_predictions',
        ['experiment_id'],
        unique=False
    )
    op.create_index(
        'ix_htc_predictions_created_at',
        'htc_predictions',
        ['created_at'],
        unique=False
    )


def downgrade() -> None:
    """Drop htc_predictions table and indexes."""
    op.drop_index('ix_htc_predictions_created_at', table_name='htc_predictions')
    op.drop_index('ix_htc_predictions_experiment_id', table_name='htc_predictions')
    op.drop_index('ix_htc_predictions_xi_parameter', table_name='htc_predictions')
    op.drop_index('ix_htc_predictions_composition', table_name='htc_predictions')
    op.drop_table('htc_predictions')

