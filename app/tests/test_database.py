"""
Integration tests for Cloud SQL database service.

These tests require a working PostgreSQL database (local or Cloud SQL Proxy).
Set database credentials in .env or environment variables.
"""

import pytest
from datetime import datetime
from src.services.db import (
    init_database, get_session, create_experiment, update_experiment,
    log_ai_query, get_experiments, get_optimization_runs,
    ExperimentStatus, OptimizationMethod,
    Experiment, OptimizationRun, AIQuery,
    Base, _engine
)

# Skip all tests if database not configured
pytest_skip_if_no_db = pytest.mark.skipif(
    not hasattr(_engine, 'url') if _engine else True,
    reason="Database not configured"
)


@pytest.fixture(scope="module")
def db_session():
    """Initialize database for testing."""
    init_database()
    session = get_session()
    if session is None:
        pytest.skip("Database not available")
    
    yield session
    
    # Cleanup: Drop test data (optional)
    session.close()


class TestDatabaseConnection:
    """Test database connection and initialization."""
    
    def test_init_database(self, db_session):
        """Test database initialization."""
        assert db_session is not None
        assert db_session.is_active
    
    def test_tables_created(self, db_session):
        """Test that all tables are created."""
        from sqlalchemy import inspect
        
        inspector = inspect(db_session.bind)
        tables = inspector.get_table_names()
        
        expected_tables = [
            "experiments",
            "optimization_runs",
            "ai_queries",
            "noise_estimates",
            "routing_decisions"
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not found"


class TestExperimentCRUD:
    """Test experiment CRUD operations."""
    
    def test_create_experiment(self, db_session):
        """Test creating an experiment."""
        experiment = create_experiment(
            experiment_id="test-exp-1",
            parameters={"temperature": 300, "pressure": 1.0},
            config={"method": "xrd"},
            created_by="test_user"
        )
        
        assert experiment is not None
        assert experiment.id == "test-exp-1"
        assert experiment.parameters == {"temperature": 300, "pressure": 1.0}
        assert experiment.status == ExperimentStatus.PENDING
        assert experiment.created_at is not None
    
    def test_update_experiment(self, db_session):
        """Test updating an experiment with results."""
        # Create experiment first
        create_experiment(
            experiment_id="test-exp-2",
            parameters={"temperature": 350},
            created_by="test_user"
        )
        
        # Update with results
        success = update_experiment(
            experiment_id="test-exp-2",
            status=ExperimentStatus.COMPLETED,
            result_value=42.5,
            result_data={"peaks": [1.2, 3.4, 5.6]},
            result_uri="gs://bucket/test.json"
        )
        
        assert success is True
        
        # Verify update
        experiments = get_experiments(limit=100)
        exp = next((e for e in experiments if e.id == "test-exp-2"), None)
        assert exp is not None
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.result_value == 42.5
        assert exp.result_data == {"peaks": [1.2, 3.4, 5.6]}
        assert exp.completed_at is not None
    
    def test_query_experiments(self, db_session):
        """Test querying experiments."""
        # Create multiple experiments
        for i in range(3):
            create_experiment(
                experiment_id=f"test-exp-query-{i}",
                parameters={"value": i},
                created_by="test_user"
            )
        
        # Query all
        experiments = get_experiments(limit=10)
        assert len(experiments) >= 3
        
        # Query by status
        experiments = get_experiments(status=ExperimentStatus.PENDING, limit=10)
        assert all(e.status == ExperimentStatus.PENDING for e in experiments)


class TestAIQueryLogging:
    """Test AI query logging."""
    
    def test_log_ai_query(self, db_session):
        """Test logging an AI query."""
        success = log_ai_query(
            query_id="test-query-1",
            query_text="Design next experiment",
            context={"previous_experiments": []},
            flash_response={
                "response": "Test response",
                "latency_ms": 1234,
                "tokens": 100
            },
            pro_response=None,
            selected_model="flash",
            created_by="test_user"
        )
        
        assert success is True
        
        # Verify by querying
        session = get_session()
        query = session.query(AIQuery).filter(AIQuery.id == "test-query-1").first()
        assert query is not None
        assert query.query_text == "Design next experiment"
        assert query.selected_model == "flash"
        assert query.flash_latency_ms == 1234
        assert query.flash_tokens == 100
        session.close()


class TestOptimizationRuns:
    """Test optimization run tracking."""
    
    def test_create_optimization_run(self, db_session):
        """Test creating an optimization run."""
        run = OptimizationRun(
            id="test-run-1",
            name="Test Run",
            method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
            objective="maximize",
            search_space={"temperature": [200, 400]},
            status=ExperimentStatus.PENDING,
            created_by="test_user"
        )
        
        session = get_session()
        session.add(run)
        session.commit()
        session.close()
        
        # Verify
        runs = get_optimization_runs(limit=10)
        test_run = next((r for r in runs if r.id == "test-run-1"), None)
        assert test_run is not None
        assert test_run.method == OptimizationMethod.BAYESIAN_OPTIMIZATION
        assert test_run.objective == "maximize"
    
    def test_experiments_linked_to_run(self, db_session):
        """Test linking experiments to optimization run."""
        # Create run
        run = OptimizationRun(
            id="test-run-2",
            name="Test Run 2",
            method=OptimizationMethod.REINFORCEMENT_LEARNING,
            objective="minimize",
            search_space={"x": [0, 1]},
            status=ExperimentStatus.RUNNING,
            created_by="test_user"
        )
        session = get_session()
        session.add(run)
        session.commit()
        session.close()
        
        # Create experiments in this run
        for i in range(3):
            create_experiment(
                experiment_id=f"test-exp-run2-{i}",
                parameters={"x": i * 0.1},
                optimization_run_id="test-run-2",
                created_by="test_user"
            )
        
        # Query experiments for this run
        experiments = get_experiments(optimization_run_id="test-run-2", limit=10)
        assert len(experiments) >= 3
        assert all(e.optimization_run_id == "test-run-2" for e in experiments)


class TestDatabaseEnums:
    """Test enum types."""
    
    def test_experiment_status_enum(self, db_session):
        """Test ExperimentStatus enum."""
        assert ExperimentStatus.PENDING == "pending"
        assert ExperimentStatus.RUNNING == "running"
        assert ExperimentStatus.COMPLETED == "completed"
        assert ExperimentStatus.FAILED == "failed"
        assert ExperimentStatus.CANCELLED == "cancelled"
    
    def test_optimization_method_enum(self, db_session):
        """Test OptimizationMethod enum."""
        assert OptimizationMethod.BAYESIAN_OPTIMIZATION == "bayesian_optimization"
        assert OptimizationMethod.REINFORCEMENT_LEARNING == "reinforcement_learning"
        assert OptimizationMethod.RANDOM_SEARCH == "random_search"
        assert OptimizationMethod.GRID_SEARCH == "grid_search"
        assert OptimizationMethod.ADAPTIVE_ROUTER == "adaptive_router"
        assert OptimizationMethod.HYBRID == "hybrid"


class TestDatabaseIndexes:
    """Test that indexes are created."""
    
    def test_indexes_exist(self, db_session):
        """Test that expected indexes are created."""
        from sqlalchemy import inspect
        
        inspector = inspect(db_session.bind)
        
        # Check experiments indexes
        exp_indexes = [idx['name'] for idx in inspector.get_indexes('experiments')]
        assert 'idx_experiments_status' in exp_indexes
        assert 'idx_experiments_created_at' in exp_indexes
        
        # Check optimization_runs indexes
        run_indexes = [idx['name'] for idx in inspector.get_indexes('optimization_runs')]
        assert 'idx_optimization_runs_method' in run_indexes
        assert 'idx_optimization_runs_status' in run_indexes


class TestTransactions:
    """Test transaction handling."""
    
    def test_rollback_on_error(self, db_session):
        """Test that failed transactions are rolled back."""
        session = get_session()
        
        try:
            # Try to create duplicate experiment (should fail on primary key)
            exp1 = Experiment(
                id="test-duplicate",
                parameters={},
                status=ExperimentStatus.PENDING
            )
            session.add(exp1)
            session.commit()
            
            # Try to add duplicate
            exp2 = Experiment(
                id="test-duplicate",  # Same ID
                parameters={},
                status=ExperimentStatus.PENDING
            )
            session.add(exp2)
            session.commit()
            
            # Should not reach here
            assert False, "Expected IntegrityError"
        
        except Exception:
            # Rollback should happen automatically
            session.rollback()
            
            # Verify first experiment still exists
            exp = session.query(Experiment).filter(Experiment.id == "test-duplicate").first()
            assert exp is not None
        
        finally:
            session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

