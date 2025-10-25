"""
Safety Gateway: Python wrapper for Rust safety kernel.

This module provides the critical safety gate between experiment submission
and execution. It enforces YAML-defined policies via the Rust safety kernel
before any experiment enters the queue.

Moat: TRUST - Fail-safe by default, zero tolerance for policy violations.
"""

import os
import structlog
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import Rust safety kernel
try:
    from safety_kernel import SafetyKernel as RustSafetyKernel
except ImportError:
    # Fallback for testing without compiled Rust
    RustSafetyKernel = None

from configs.data_schema import Experiment, Protocol


logger = structlog.get_logger()


class SafetyVerdict(Enum):
    """Safety check result."""
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_APPROVAL = "requires_approval"
    WARNING = "warning"


@dataclass
class SafetyCheckResult:
    """Result of safety validation."""
    verdict: SafetyVerdict
    policy_violations: List[str]
    warnings: List[str]
    reason: Optional[str] = None
    
    @property
    def approved(self) -> bool:
        """True if experiment can proceed without human intervention."""
        return self.verdict in [SafetyVerdict.APPROVED, SafetyVerdict.WARNING]
    
    @property
    def requires_human_approval(self) -> bool:
        """True if human must approve before queueing."""
        return self.verdict == SafetyVerdict.REQUIRES_APPROVAL
    
    @property
    def rejected(self) -> bool:
        """True if experiment is rejected and cannot proceed."""
        return self.verdict == SafetyVerdict.REJECTED


class SafetyGateway:
    """
    Python-side safety gateway that invokes Rust safety kernel.
    
    This is the MANDATORY gate between experiment submission and queue.
    NO experiment bypasses this check.
    
    Architecture:
        User/AI → submit_experiment() → SafetyGateway.check()
                                              ↓
                                       [Rust SafetyKernel]
                                              ↓
                                    [YAML Policy Evaluation]
                                              ↓
                                    Approved / Rejected / Approval Needed
                                              ↓
                            queue.enqueue() OR reject OR pause
    """
    
    def __init__(self, policy_path: Optional[str] = None):
        """
        Initialize safety gateway with policies.
        
        Args:
            policy_path: Path to safety_policies.yaml. Defaults to configs/safety_policies.yaml
        """
        if policy_path is None:
            policy_path = str(Path(__file__).parent.parent.parent / "configs" / "safety_policies.yaml")
        
        self.policy_path = policy_path
        self.rust_kernel = None
        self.policies_loaded = False
        self.reagent_incompatibilities = []
        
        self._initialize_kernel()
        
        logger.info(
            "safety_gateway_initialized",
            policy_path=policy_path,
            kernel_available=self.rust_kernel is not None
        )
    
    def _initialize_kernel(self):
        """Load Rust safety kernel and YAML policies."""
        if RustSafetyKernel is None:
            logger.warning("safety_kernel_not_compiled", 
                         message="Rust safety kernel not available - safety checks DISABLED")
            return
        
        try:
            self.rust_kernel = RustSafetyKernel()
            
            # Load YAML policies
            if os.path.exists(self.policy_path):
                with open(self.policy_path, 'r') as f:
                    yaml_content = f.read()
                
                self.rust_kernel.load_policies_from_yaml(yaml_content)
                self.policies_loaded = True
                
                # Load reagent incompatibilities (Python-side check)
                import yaml
                policy_data = yaml.safe_load(yaml_content)
                self.reagent_incompatibilities = policy_data.get("reagent_incompatibilities", [])
                
                logger.info(
                    "safety_policies_loaded",
                    policy_count=len(policy_data.get("policies", [])),
                    incompatibility_count=len(self.reagent_incompatibilities)
                )
            else:
                logger.error("safety_policy_file_not_found", path=self.policy_path)
        
        except Exception as e:
            logger.exception("safety_kernel_initialization_failed", error=str(e))
    
    def check_experiment(self, experiment: Experiment) -> SafetyCheckResult:
        """
        Validate experiment against all safety policies.
        
        This is the CRITICAL safety gate. Called BEFORE queueing.
        
        Args:
            experiment: Experiment to validate
            
        Returns:
            SafetyCheckResult with verdict and violations
        """
        violations = []
        warnings = []
        
        # If kernel not available, REJECT by default (fail-safe)
        if self.rust_kernel is None or not self.policies_loaded:
            return SafetyCheckResult(
                verdict=SafetyVerdict.REJECTED,
                policy_violations=["Safety kernel not initialized - cannot validate"],
                warnings=[],
                reason="CRITICAL: Safety system unavailable"
            )
        
        try:
            # 1. Check protocol parameters against Rust kernel policies
            protocol_params = self._extract_numeric_parameters(experiment.protocol)
            instrument_id = experiment.protocol.instrument_id
            
            rust_violation = self.rust_kernel.check_experiment(
                instrument_id,
                protocol_params
            )
            
            if rust_violation is not None:
                violations.append(rust_violation)
            
            # 2. Check reagent incompatibilities (Python-side)
            reagent_violation = self._check_reagent_compatibility(experiment)
            if reagent_violation:
                violations.append(reagent_violation)
            
            # 3. Determine verdict
            if violations:
                # Check if any violation triggers shutdown or rejection
                if any("Shutdown" in v or "Reject" in v for v in violations):
                    verdict = SafetyVerdict.REJECTED
                    reason = f"{len(violations)} critical policy violation(s)"
                elif any("PauseForApproval" in v for v in violations):
                    verdict = SafetyVerdict.REQUIRES_APPROVAL
                    reason = "Human approval required before execution"
                else:
                    verdict = SafetyVerdict.WARNING
                    warnings = violations.copy()
                    violations = []
                    reason = "Non-critical warnings present"
            else:
                verdict = SafetyVerdict.APPROVED
                reason = "All safety checks passed"
            
            # Log safety check
            logger.info(
                "safety_check_completed",
                experiment_id=experiment.id,
                verdict=verdict.value,
                violations=len(violations),
                warnings=len(warnings)
            )
            
            return SafetyCheckResult(
                verdict=verdict,
                policy_violations=violations,
                warnings=warnings,
                reason=reason
            )
        
        except Exception as e:
            # ANY exception in safety check → REJECT (fail-safe)
            logger.exception("safety_check_error", experiment_id=experiment.id, error=str(e))
            return SafetyCheckResult(
                verdict=SafetyVerdict.REJECTED,
                policy_violations=[f"Safety check failed: {str(e)}"],
                warnings=[],
                reason="CRITICAL: Safety validation error"
            )
    
    def _extract_numeric_parameters(self, protocol: Protocol) -> Dict[str, float]:
        """
        Extract numeric parameters from protocol for Rust kernel.
        
        Rust kernel expects Dict[str, float] for rule evaluation.
        """
        numeric_params = {}
        
        for key, value in protocol.parameters.items():
            # Try to convert to float
            try:
                if isinstance(value, (int, float)):
                    numeric_params[key] = float(value)
                elif isinstance(value, str):
                    # Try parsing numeric strings like "100.5" or "25 C"
                    numeric_str = ''.join(c for c in value if c.isdigit() or c == '.' or c == '-')
                    if numeric_str:
                        numeric_params[key] = float(numeric_str)
            except (ValueError, TypeError):
                # Skip non-numeric parameters
                pass
        
        return numeric_params
    
    def _check_reagent_compatibility(self, experiment: Experiment) -> Optional[str]:
        """
        Check for incompatible reagent pairs.
        
        This is a Python-side check for reagent safety that complements
        the Rust kernel's numeric parameter checks.
        """
        # Extract reagents from protocol metadata or parameters
        reagents = []
        
        if "reagents" in experiment.metadata:
            reagents = experiment.metadata["reagents"]
        elif "reagent" in experiment.protocol.parameters:
            reagents = [experiment.protocol.parameters["reagent"]]
        
        if not reagents:
            return None
        
        # Check all incompatible pairs
        for incompatibility in self.reagent_incompatibilities:
            pair = incompatibility["pair"]
            reason = incompatibility["reason"]
            
            # Check if both reagents in the pair are present
            pair_lower = [p.lower() for p in pair]
            reagents_lower = [r.lower() for r in reagents]
            
            if all(any(p in r for r in reagents_lower) for p in pair_lower):
                return f"Incompatible reagents detected: {pair[0]} + {pair[1]} - {reason} (action: Reject)"
        
        return None


# Global singleton instance
_gateway_instance: Optional[SafetyGateway] = None


def get_safety_gateway() -> SafetyGateway:
    """Get or create global safety gateway instance."""
    global _gateway_instance
    
    if _gateway_instance is None:
        _gateway_instance = SafetyGateway()
    
    return _gateway_instance


def validate_experiment_safe(experiment: Experiment) -> Tuple[bool, str]:
    """
    Convenience function for quick safety validation.
    
    Args:
        experiment: Experiment to validate
        
    Returns:
        Tuple of (is_safe, reason)
    """
    gateway = get_safety_gateway()
    result = gateway.check_experiment(experiment)
    
    return result.approved, result.reason or "Unknown"

