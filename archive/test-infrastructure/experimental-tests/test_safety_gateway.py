"""
Tests for safety gateway integration.

Validates that experiments are properly checked against safety policies
before entering the queue.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch
from datetime import datetime

try:  # pragma: no cover - optional dependency
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("structlog not available", allow_module_level=True)

# Mock safety_kernel module before importing gateway
sys.modules['safety_kernel'] = MagicMock()

from src.safety.gateway import (
    SafetyGateway,
    SafetyCheckResult,
    SafetyVerdict,
    get_safety_gateway,
    validate_experiment_safe
)
from configs.data_schema import Experiment, Protocol, Sample


@pytest.fixture
def mock_rust_kernel():
    """Mock Rust safety kernel for testing."""
    kernel = MagicMock()
    kernel.check_experiment = MagicMock(return_value=None)  # No violations by default
    return kernel


@pytest.fixture
def sample_experiment():
    """Create a valid test experiment."""
    protocol = Protocol(
        instrument_id="xrd-001",
        parameters={
            "temperature": 100.0,
            "pressure": 1.0,
            "duration": 2.0
        },
        duration_estimate_hours=2.0
    )
    
    return Experiment(
        sample_id="sample-001",
        protocol=protocol,
        created_by="test-user",
        priority=5
    )


@pytest.fixture
def unsafe_experiment():
    """Create an experiment that violates temperature limits."""
    protocol = Protocol(
        instrument_id="xrd-001",
        parameters={
            "temperature": 200.0,  # Exceeds 150°C limit
            "pressure": 1.0,
            "duration": 2.0
        },
        duration_estimate_hours=2.0
    )
    
    return Experiment(
        sample_id="sample-002",
        protocol=protocol,
        created_by="test-user",
        priority=5
    )


class TestSafetyGateway:
    """Test safety gateway functionality."""
    
    def test_gateway_initialization_without_rust_kernel(self, tmp_path):
        """Test that gateway fails safe when Rust kernel unavailable."""
        # Create temporary policy file
        policy_path = tmp_path / "test_policies.yaml"
        policy_path.write_text("policies: []")
        
        # Initialize gateway (Rust kernel is mocked but set to None)
        with patch('src.safety.gateway.RustSafetyKernel', None):
            gateway = SafetyGateway(policy_path=str(policy_path))
            
            assert gateway.rust_kernel is None
            assert not gateway.policies_loaded
    
    def test_safety_check_rejects_when_kernel_unavailable(self, sample_experiment):
        """Test fail-safe behavior when safety kernel unavailable."""
        with patch('src.safety.gateway.RustSafetyKernel', None):
            gateway = SafetyGateway()
            result = gateway.check_experiment(sample_experiment)
            
            assert result.verdict == SafetyVerdict.REJECTED
            assert result.rejected
            assert "Safety kernel not initialized" in result.policy_violations[0]
    
    def test_safe_experiment_approved(self, sample_experiment, mock_rust_kernel):
        """Test that safe experiments are approved."""
        with patch('src.safety.gateway.RustSafetyKernel', return_value=mock_rust_kernel):
            gateway = SafetyGateway()
            gateway.rust_kernel = mock_rust_kernel
            gateway.policies_loaded = True
            
            result = gateway.check_experiment(sample_experiment)
            
            assert result.verdict == SafetyVerdict.APPROVED
            assert result.approved
            assert len(result.policy_violations) == 0
    
    def test_unsafe_experiment_rejected(self, unsafe_experiment, mock_rust_kernel):
        """Test that unsafe experiments are rejected."""
        # Mock Rust kernel to return a violation
        mock_rust_kernel.check_experiment.return_value = (
            "Policy 'Maximum temperature limit' violated: "
            "temperature = 200.00 violates <= 150.00 (action: Shutdown)"
        )
        
        with patch('src.safety.gateway.RustSafetyKernel', return_value=mock_rust_kernel):
            gateway = SafetyGateway()
            gateway.rust_kernel = mock_rust_kernel
            gateway.policies_loaded = True
            
            result = gateway.check_experiment(unsafe_experiment)
            
            assert result.verdict == SafetyVerdict.REJECTED
            assert result.rejected
            assert not result.approved
            assert len(result.policy_violations) > 0
            assert "temperature" in result.policy_violations[0].lower()
    
    def test_low_confidence_requires_approval(self, sample_experiment, mock_rust_kernel):
        """Test that low confidence experiments require human approval."""
        # Add low confidence to protocol
        sample_experiment.protocol.parameters["confidence"] = 0.5
        
        # Mock violation for low confidence
        mock_rust_kernel.check_experiment.return_value = (
            "Policy 'Low confidence requires approval' violated: "
            "confidence = 0.50 violates >= 0.80 (action: PauseForApproval)"
        )
        
        with patch('src.safety.gateway.RustSafetyKernel', return_value=mock_rust_kernel):
            gateway = SafetyGateway()
            gateway.rust_kernel = mock_rust_kernel
            gateway.policies_loaded = True
            
            result = gateway.check_experiment(sample_experiment)
            
            assert result.verdict == SafetyVerdict.REQUIRES_APPROVAL
            assert result.requires_human_approval
            assert not result.approved
    
    def test_reagent_incompatibility_detection(self, sample_experiment, mock_rust_kernel):
        """Test detection of incompatible reagents."""
        # Add incompatible reagents to metadata
        sample_experiment.metadata["reagents"] = ["sodium", "water"]
        
        with patch('src.safety.gateway.RustSafetyKernel', return_value=mock_rust_kernel):
            gateway = SafetyGateway()
            gateway.rust_kernel = mock_rust_kernel
            gateway.policies_loaded = True
            gateway.reagent_incompatibilities = [
                {
                    "pair": ["sodium", "water"],
                    "reason": "Violent exothermic reaction"
                }
            ]
            
            result = gateway.check_experiment(sample_experiment)
            
            assert result.verdict == SafetyVerdict.REJECTED
            assert result.rejected
            assert any("sodium" in v.lower() and "water" in v.lower() 
                      for v in result.policy_violations)
    
    def test_numeric_parameter_extraction(self, mock_rust_kernel):
        """Test extraction of numeric parameters from protocol."""
        protocol = Protocol(
            instrument_id="test-001",
            parameters={
                "temperature": 100.0,
                "pressure": "2.5 bar",  # String with unit
                "duration": "5",  # Numeric string
                "sample_name": "TestSample",  # Non-numeric
                "mode": "fast"  # Non-numeric
            },
            duration_estimate_hours=1.0
        )
        
        with patch('src.safety.gateway.RustSafetyKernel', return_value=mock_rust_kernel):
            gateway = SafetyGateway()
            gateway.rust_kernel = mock_rust_kernel
            gateway.policies_loaded = True
            
            params = gateway._extract_numeric_parameters(protocol)
            
            assert params["temperature"] == 100.0
            assert params["pressure"] == 2.5
            assert params["duration"] == 5.0
            assert "sample_name" not in params
            assert "mode" not in params
    
    def test_safety_check_exception_handling(self, sample_experiment, mock_rust_kernel):
        """Test that exceptions in safety check result in rejection (fail-safe)."""
        # Mock kernel to raise exception
        mock_rust_kernel.check_experiment.side_effect = RuntimeError("Kernel error")
        
        with patch('src.safety.gateway.RustSafetyKernel', return_value=mock_rust_kernel):
            gateway = SafetyGateway()
            gateway.rust_kernel = mock_rust_kernel
            gateway.policies_loaded = True
            
            result = gateway.check_experiment(sample_experiment)
            
            assert result.verdict == SafetyVerdict.REJECTED
            assert result.rejected
            assert "Safety check failed" in result.policy_violations[0]
    
    def test_validate_experiment_safe_convenience_function(self, sample_experiment, mock_rust_kernel):
        """Test convenience function for safety validation."""
        with patch('src.safety.gateway.RustSafetyKernel', return_value=mock_rust_kernel):
            with patch('src.safety.gateway.get_safety_gateway') as mock_get_gateway:
                mock_gateway = MagicMock()
                mock_gateway.check_experiment.return_value = SafetyCheckResult(
                    verdict=SafetyVerdict.APPROVED,
                    policy_violations=[],
                    warnings=[],
                    reason="All checks passed"
                )
                mock_get_gateway.return_value = mock_gateway
                
                is_safe, reason = validate_experiment_safe(sample_experiment)
                
                assert is_safe
                assert "All checks passed" in reason


class TestExperimentOSIntegration:
    """Test safety gateway integration with ExperimentOS."""
    
    @pytest.mark.asyncio
    async def test_safe_experiment_queued(self, sample_experiment, mock_rust_kernel):
        """Test that safe experiments are successfully queued."""
        from src.experiment_os.core import ExperimentOS
        
        with patch('src.safety.gateway.RustSafetyKernel', return_value=mock_rust_kernel):
            os_system = ExperimentOS(enable_safety_gateway=True)
            
            # Mock the safety gateway to approve
            os_system.safety_gateway = MagicMock()
            os_system.safety_gateway.check_experiment.return_value = SafetyCheckResult(
                verdict=SafetyVerdict.APPROVED,
                policy_violations=[],
                warnings=[],
                reason="All checks passed"
            )
            
            result = await os_system.submit_experiment(sample_experiment)
            
            assert result["status"] == "queued"
            assert result["experiment_id"] == sample_experiment.id
            assert len(result["violations"]) == 0
    
    @pytest.mark.asyncio
    async def test_unsafe_experiment_rejected(self, unsafe_experiment, mock_rust_kernel):
        """Test that unsafe experiments raise ValueError and are not queued."""
        from src.experiment_os.core import ExperimentOS
        
        with patch('src.safety.gateway.RustSafetyKernel', return_value=mock_rust_kernel):
            os_system = ExperimentOS(enable_safety_gateway=True)
            
            # Mock the safety gateway to reject
            os_system.safety_gateway = MagicMock()
            os_system.safety_gateway.check_experiment.return_value = SafetyCheckResult(
                verdict=SafetyVerdict.REJECTED,
                policy_violations=["Temperature exceeds limit"],
                warnings=[],
                reason="Critical safety violation"
            )
            
            with pytest.raises(ValueError) as exc_info:
                await os_system.submit_experiment(unsafe_experiment)
            
            assert "rejected by safety gateway" in str(exc_info.value).lower()
            
            # Verify experiment was NOT queued
            assert len(os_system.queue.queue) == 0
    
    @pytest.mark.asyncio
    async def test_approval_required_experiment(self, sample_experiment, mock_rust_kernel):
        """Test that experiments requiring approval return correct status."""
        from src.experiment_os.core import ExperimentOS
        
        with patch('src.safety.gateway.RustSafetyKernel', return_value=mock_rust_kernel):
            os_system = ExperimentOS(enable_safety_gateway=True)
            
            # Mock the safety gateway to require approval
            os_system.safety_gateway = MagicMock()
            os_system.safety_gateway.check_experiment.return_value = SafetyCheckResult(
                verdict=SafetyVerdict.REQUIRES_APPROVAL,
                policy_violations=["Low confidence"],
                warnings=[],
                reason="Human approval required"
            )
            
            result = await os_system.submit_experiment(sample_experiment)
            
            assert result["status"] == "requires_approval"
            assert result["experiment_id"] == sample_experiment.id
            assert len(result["violations"]) > 0
            
            # Verify experiment was NOT auto-queued
            assert len(os_system.queue.queue) == 0
    
    @pytest.mark.asyncio
    async def test_safety_gateway_disabled(self, sample_experiment):
        """Test that experiments can be submitted when safety gateway disabled (testing only)."""
        from src.experiment_os.core import ExperimentOS
        
        os_system = ExperimentOS(enable_safety_gateway=False)
        
        result = await os_system.submit_experiment(sample_experiment)
        
        assert result["status"] == "queued"
        assert result["experiment_id"] == sample_experiment.id
        # Verify experiment WAS queued (no safety check)
        assert len(os_system.queue.queue) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

