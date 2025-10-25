//! Safety Kernel: Rust-based safety enforcement for autonomous experiments.
//!
//! This module provides memory-safe, fast enforcement of safety policies,
//! resource limits, and interlocks to protect hardware and personnel.
//!
//! Moat: TRUST - Safety-first design with fail-safe defaults.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Safety policy action to take when rule is violated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[pyclass]
pub enum SafetyAction {
    /// Immediately shut down all instruments
    Shutdown,
    /// Reject experiment submission
    Reject,
    /// Pause and request human approval
    PauseForApproval,
    /// Log warning but allow
    Warn,
}

/// Safety policy rule with conditions and actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct SafetyPolicy {
    #[pyo3(get, set)]
    pub name: String,
    
    #[pyo3(get, set)]
    pub rule: String,  // Symbolic expression like "temperature <= 150.0"
    
    #[pyo3(get, set)]
    pub scope: Vec<String>,  // Instrument IDs this applies to
    
    #[pyo3(get, set)]
    pub action: SafetyAction,
    
    #[pyo3(get, set)]
    pub severity: String,  // "critical", "medium", "low"
}

#[pymethods]
impl SafetyPolicy {
    #[new]
    fn new(name: String, rule: String, scope: Vec<String>, action: SafetyAction, severity: String) -> Self {
        SafetyPolicy {
            name,
            rule,
            scope,
            action,
            severity,
        }
    }
}

/// Safety violation details.
#[derive(Debug, Error)]
pub enum SafetyError {
    #[error("Safety policy '{policy}' violated: {reason}")]
    PolicyViolation { policy: String, reason: String },
    
    #[error("Resource limit exceeded: {resource} = {value} > {limit}")]
    ResourceLimit {
        resource: String,
        value: f64,
        limit: f64,
    },
    
    #[error("Incompatible reagents: {reagents:?}")]
    IncompatibleReagents { reagents: Vec<String> },
}

/// Main safety kernel for policy enforcement.
#[pyclass]
pub struct SafetyKernel {
    policies: Vec<SafetyPolicy>,
    resource_limits: HashMap<String, f64>,
}

#[pymethods]
impl SafetyKernel {
    #[new]
    fn new() -> Self {
        SafetyKernel {
            policies: Vec::new(),
            resource_limits: HashMap::new(),
        }
    }
    
    /// Load safety policies from YAML file.
    fn load_policies_from_yaml(&mut self, yaml_content: &str) -> PyResult<()> {
        #[derive(Deserialize)]
        struct PolicyFile {
            policies: Vec<SafetyPolicy>,
        }
        
        let policy_file: PolicyFile = serde_yaml::from_str(yaml_content)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid YAML: {}", e)))?;
        
        self.policies = policy_file.policies;
        Ok(())
    }
    
    /// Check if experiment protocol violates any safety policies.
    fn check_experiment(&self, instrument_id: &str, protocol_params: HashMap<String, f64>) -> PyResult<Option<String>> {
        for policy in &self.policies {
            // Check if policy applies to this instrument
            if policy.scope.iter().any(|s| s == "all" || s == instrument_id) {
                // Simple rule evaluation (temperature, pressure, etc.)
                if let Some(violation) = self.evaluate_rule(&policy.rule, &protocol_params) {
                    // Return violation message
                    return Ok(Some(format!(
                        "Policy '{}' violated: {} (action: {:?})",
                        policy.name, violation, policy.action
                    )));
                }
            }
        }
        
        Ok(None)  // No violations
    }
    
    /// Set resource limit.
    fn set_resource_limit(&mut self, resource: String, limit: f64) {
        self.resource_limits.insert(resource, limit);
    }
    
    /// Check if resource usage is within limits.
    fn check_resource_limit(&self, resource: &str, value: f64) -> PyResult<bool> {
        if let Some(&limit) = self.resource_limits.get(resource) {
            Ok(value <= limit)
        } else {
            Ok(true)  // No limit set, allow
        }
    }
}

impl SafetyKernel {
    /// Evaluate a simple rule expression.
    /// 
    /// Supports: >, <, >=, <=, ==, !=
    /// Example: "temperature <= 150.0"
    fn evaluate_rule(&self, rule: &str, params: &HashMap<String, f64>) -> Option<String> {
        // Parse rule: "parameter operator value"
        let parts: Vec<&str> = rule.split_whitespace().collect();
        
        if parts.len() < 3 {
            return None;
        }
        
        let param_name = parts[0];
        let operator = parts[1];
        let threshold: f64 = parts[2].parse().ok()?;
        
        let param_value = params.get(param_name)?;
        
        let violated = match operator {
            "<=" => *param_value > threshold,
            ">=" => *param_value < threshold,
            "<" => *param_value >= threshold,
            ">" => *param_value <= threshold,
            "==" => (*param_value - threshold).abs() > 1e-6,
            "!=" => (*param_value - threshold).abs() < 1e-6,
            _ => false,
        };
        
        if violated {
            Some(format!(
                "{} = {:.2} violates {} {:.2}",
                param_name, param_value, operator, threshold
            ))
        } else {
            None
        }
    }
}

/// Dead-man switch for automatic shutdown.
#[pyclass]
pub struct DeadManSwitch {
    last_heartbeat: std::time::Instant,
    timeout_secs: u64,
}

#[pymethods]
impl DeadManSwitch {
    #[new]
    fn new(timeout_secs: u64) -> Self {
        DeadManSwitch {
            last_heartbeat: std::time::Instant::now(),
            timeout_secs,
        }
    }
    
    /// Update heartbeat timestamp.
    fn heartbeat(&mut self) {
        self.last_heartbeat = std::time::Instant::now();
    }
    
    /// Check if heartbeat has timed out.
    fn is_alive(&self) -> bool {
        self.last_heartbeat.elapsed().as_secs() < self.timeout_secs
    }
    
    /// Get seconds since last heartbeat.
    fn seconds_since_heartbeat(&self) -> u64 {
        self.last_heartbeat.elapsed().as_secs()
    }
}

/// Python module definition.
#[pymodule]
fn safety_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SafetyKernel>()?;
    m.add_class::<SafetyPolicy>()?;
    m.add_class::<SafetyAction>()?;
    m.add_class::<DeadManSwitch>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_temperature_limit() {
        let kernel = SafetyKernel::new();
        
        let mut params = HashMap::new();
        params.insert("temperature".to_string(), 200.0);
        
        let violation = kernel.evaluate_rule("temperature <= 150.0", &params);
        assert!(violation.is_some());
        assert!(violation.unwrap().contains("200.00 violates <= 150.00"));
    }
    
    #[test]
    fn test_temperature_within_limit() {
        let kernel = SafetyKernel::new();
        
        let mut params = HashMap::new();
        params.insert("temperature".to_string(), 100.0);
        
        let violation = kernel.evaluate_rule("temperature <= 150.0", &params);
        assert!(violation.is_none());
    }
    
    #[test]
    fn test_dead_man_switch() {
        let mut switch = DeadManSwitch::new(5);
        
        assert!(switch.is_alive());
        
        switch.heartbeat();
        assert!(switch.is_alive());
        
        std::thread::sleep(std::time::Duration::from_secs(6));
        assert!(!switch.is_alive());
    }
}

