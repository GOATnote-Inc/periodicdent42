"""
Tests for metadata API endpoints (experiments, optimization_runs, ai_queries).
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime

from src.api.main import app
from src.services.db import (
    Experiment, OptimizationRun, AIQuery,
    ExperimentStatus, OptimizationMethod
)


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_experiment():
    """Mock experiment object."""
    exp = Mock(spec=Experiment)
    exp.id = "exp-123"
    exp.name = "Test Experiment"
    exp.description = "Test description"
    exp.status = ExperimentStatus.COMPLETED
    exp.parameters = {"temp": 500, "pressure": 1.0}
    exp.config = {"method": "BO"}
    exp.result_value = 0.85
    exp.result_data = {"metric": "accuracy", "value": 0.85}
    exp.result_uri = "gs://bucket/exp-123.json"
    exp.created_at = datetime(2025, 10, 1, 10, 0, 0)
    exp.started_at = datetime(2025, 10, 1, 10, 5, 0)
    exp.completed_at = datetime(2025, 10, 1, 10, 15, 0)
    exp.created_by = "test-user"
    exp.optimization_run_id = "run-456"
    exp.optimization_run = None
    return exp


@pytest.fixture
def mock_optimization_run():
    """Mock optimization run object."""
    run = Mock(spec=OptimizationRun)
    run.id = "run-456"
    run.name = "Test Run"
    run.description = "Test run description"
    run.method = OptimizationMethod.BAYESIAN_OPTIMIZATION
    run.objective = "maximize"
    run.search_space = {"temp": [200, 800], "pressure": [0.5, 2.0]}
    run.config = {"acq_func": "EI"}
    run.num_experiments = 10
    run.best_value = 0.92
    run.best_experiment_id = "exp-456"
    run.status = ExperimentStatus.COMPLETED
    run.created_at = datetime(2025, 10, 1, 9, 0, 0)
    run.started_at = datetime(2025, 10, 1, 9, 5, 0)
    run.completed_at = datetime(2025, 10, 1, 11, 0, 0)
    run.created_by = "test-user"
    return run


@pytest.fixture
def mock_ai_query():
    """Mock AI query object."""
    query = Mock(spec=AIQuery)
    query.id = "query-789"
    query.query_text = "What is the best temperature for perovskite synthesis?"
    query.selected_model = "flash"
    query.flash_latency_ms = 250.5
    query.pro_latency_ms = 1200.8
    query.flash_tokens = 150
    query.pro_tokens = 180
    query.estimated_cost_usd = 0.0012
    query.created_at = datetime(2025, 10, 1, 12, 0, 0)
    query.created_by = "test-user"
    query.experiment_id = "exp-123"
    return query


class TestExperimentsEndpoint:
    """Tests for GET /api/experiments endpoint."""
    
    @patch("src.api.main.get_session")
    def test_list_experiments_success(self, mock_get_session, client, mock_experiment):
        """Test successful experiment listing."""
        # Mock database session
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_experiment]
        mock_session.query.return_value = mock_query
        mock_get_session.return_value = mock_session
        
        response = client.get("/api/experiments")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["count"] == 1
        assert len(data["experiments"]) == 1
        
        exp_data = data["experiments"][0]
        assert exp_data["id"] == "exp-123"
        assert exp_data["name"] == "Test Experiment"
        assert exp_data["status"] == "completed"
        assert exp_data["result_value"] == 0.85
        
        mock_session.close.assert_called_once()
    
    @patch("src.api.main.get_session")
    def test_list_experiments_with_filters(self, mock_get_session, client, mock_experiment):
        """Test experiment listing with filters."""
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_experiment]
        mock_session.query.return_value = mock_query
        mock_get_session.return_value = mock_session
        
        response = client.get("/api/experiments?status=completed&limit=50")
        
        assert response.status_code == 200
        assert response.json()["count"] == 1
    
    @patch("src.api.main.get_session")
    def test_list_experiments_invalid_status(self, mock_get_session, client):
        """Test experiment listing with invalid status."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        response = client.get("/api/experiments?status=invalid")
        
        assert response.status_code == 400
        assert "Invalid status" in response.json()["error"]
    
    @patch("src.api.main.get_session")
    def test_list_experiments_db_unavailable(self, mock_get_session, client):
        """Test experiment listing when database is unavailable."""
        mock_get_session.return_value = None
        
        response = client.get("/api/experiments")
        
        assert response.status_code == 503
        assert response.json()["error"] == "Database not available"


class TestExperimentDetailEndpoint:
    """Tests for GET /api/experiments/{id} endpoint."""
    
    @patch("src.api.main.get_session")
    def test_get_experiment_success(self, mock_get_session, client, mock_experiment):
        """Test successful experiment retrieval."""
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_experiment
        mock_session.query.return_value = mock_query
        mock_get_session.return_value = mock_session
        
        response = client.get("/api/experiments/exp-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["experiment"]["id"] == "exp-123"
        assert data["experiment"]["name"] == "Test Experiment"
        
        mock_session.close.assert_called_once()
    
    @patch("src.api.main.get_session")
    def test_get_experiment_not_found(self, mock_get_session, client):
        """Test experiment retrieval when not found."""
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None
        mock_session.query.return_value = mock_query
        mock_get_session.return_value = mock_session
        
        response = client.get("/api/experiments/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["error"]


class TestOptimizationRunsEndpoint:
    """Tests for GET /api/optimization_runs endpoint."""
    
    @patch("src.api.main.get_session")
    def test_list_runs_success(self, mock_get_session, client, mock_optimization_run):
        """Test successful optimization run listing."""
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_optimization_run]
        mock_session.query.return_value = mock_query
        mock_get_session.return_value = mock_session
        
        response = client.get("/api/optimization_runs")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["count"] == 1
        
        run_data = data["runs"][0]
        assert run_data["id"] == "run-456"
        assert run_data["method"] == "bayesian_optimization"
        assert run_data["best_value"] == 0.92
        
        mock_session.close.assert_called_once()
    
    @patch("src.api.main.get_session")
    def test_list_runs_with_method_filter(self, mock_get_session, client, mock_optimization_run):
        """Test run listing with method filter."""
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_optimization_run]
        mock_session.query.return_value = mock_query
        mock_get_session.return_value = mock_session
        
        response = client.get("/api/optimization_runs?method=bayesian_optimization")
        
        assert response.status_code == 200
        assert response.json()["count"] == 1


class TestAIQueriesEndpoint:
    """Tests for GET /api/ai_queries endpoint."""
    
    @patch("src.api.main.get_session")
    def test_list_queries_with_cost_analysis(self, mock_get_session, client, mock_ai_query):
        """Test AI query listing with cost analysis."""
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_ai_query, mock_ai_query]
        mock_session.query.return_value = mock_query
        mock_get_session.return_value = mock_session
        
        response = client.get("/api/ai_queries?include_cost_analysis=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["count"] == 2
        
        # Check cost analysis
        assert "cost_analysis" in data
        cost = data["cost_analysis"]
        assert cost["total_queries"] == 2
        assert cost["flash_queries"] == 2
        assert cost["pro_queries"] == 0
        assert cost["total_flash_tokens"] == 300
        assert cost["estimated_total_cost_usd"] > 0
        
        mock_session.close.assert_called_once()
    
    @patch("src.api.main.get_session")
    def test_list_queries_without_cost_analysis(self, mock_get_session, client, mock_ai_query):
        """Test AI query listing without cost analysis."""
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_ai_query]
        mock_session.query.return_value = mock_query
        mock_get_session.return_value = mock_session
        
        response = client.get("/api/ai_queries?include_cost_analysis=false")
        
        assert response.status_code == 200
        data = response.json()
        assert "cost_analysis" not in data
    
    @patch("src.api.main.get_session")
    def test_list_queries_invalid_model(self, mock_get_session, client):
        """Test AI query listing with invalid model filter."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        response = client.get("/api/ai_queries?selected_model=invalid")
        
        assert response.status_code == 400
        assert "Invalid model" in response.json()["error"]

