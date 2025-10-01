#![deny(clippy::all)]

pub mod data;
pub mod errors;
pub mod features;
pub mod plan;
pub mod qc;
pub mod tracing;

use chrono::{DateTime, Utc};
use plan::planner::{DeterministicPlanner, Planner};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info_span;
use uuid::Uuid;

/// Unique identifier type alias for domain objects.
pub type Id = Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sample {
    pub id: Id,
    pub composition: IndexMap<String, f64>,
    pub provenance: String,
    pub batch: String,
}

use indexmap::IndexMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Constraint {
    pub name: String,
    pub description: String,
    pub satisfied: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Step {
    pub description: String,
    pub duration_minutes: u32,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Recipe {
    pub steps: Vec<Step>,
    pub constraints: Vec<Constraint>,
    pub safety_interlocks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RationaleTraceEntry {
    pub option: String,
    pub score: f64,
    pub why: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Plan {
    pub id: Id,
    pub objective: Objective,
    pub candidate_recipes: Vec<Recipe>,
    pub rationale_trace: Vec<RationaleTraceEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Run {
    pub id: Id,
    pub instrument_id: String,
    pub params: IndexMap<String, String>,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub artifacts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QCCheck {
    pub id: Id,
    pub checklist_items: Vec<String>,
    pub outcome: QCOutcome,
    pub notes: String,
    pub negatives_captured: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QCOutcome {
    Pass,
    Fail,
    NeedsReview,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Objective {
    pub description: String,
    pub target_metrics: IndexMap<String, f64>,
}

/// High-level façade for the core planner that binds deterministic seeding with telemetry.
#[derive(Clone)]
pub struct CorePlanner {
    planner: Arc<DeterministicPlanner>,
}

impl CorePlanner {
    pub fn new(seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self {
            planner: Arc::new(DeterministicPlanner::new(rng)),
        }
    }

    pub fn plan(&self, objective: Objective) -> Result<Plan, errors::CoreError> {
        let span = info_span!("plan.generate", objective = %objective.description);
        let _guard = span.enter();
        self.planner.plan(objective)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_contains_rationale() {
        let planner = CorePlanner::new(42);
        let objective = Objective {
            description: "Test objective".to_string(),
            target_metrics: IndexMap::from([(String::from("yield"), 0.9)]),
        };
        let plan = planner.plan(objective).expect("plan generation succeeds");
        assert!(!plan.rationale_trace.is_empty());
    }
}
