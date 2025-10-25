use crate::{errors::CoreError, Objective, Plan, RationaleTraceEntry, Recipe, Step};
use itertools::Itertools;
use parking_lot::Mutex;
use rand::{rngs::StdRng, Rng};
use tracing::info_span;
use uuid::Uuid;

pub trait Planner: Send + Sync {
    fn plan(&self, objective: Objective) -> Result<Plan, CoreError>;
}

#[derive(Debug)]
pub struct DeterministicPlanner {
    rng: Mutex<StdRng>,
}

impl DeterministicPlanner {
    pub fn new(rng: StdRng) -> Self {
        Self {
            rng: parking_lot::Mutex::new(rng),
        }
    }

    fn score_options(&self, objective: &Objective) -> Vec<RationaleTraceEntry> {
        let mut rng = self.rng.lock();
        let mut entries = objective
            .target_metrics
            .iter()
            .map(|(metric, target)| {
                let jitter: f64 = rng.gen::<f64>() / 10.0;
                let score = target + jitter;
                RationaleTraceEntry {
                    option: format!("Optimize {metric}"),
                    score,
                    why: format!("Target {target:.3} + jitter {jitter:.3}"),
                }
            })
            .collect_vec();
        entries.sort_by(|a, b| b.score.total_cmp(&a.score));
        entries
    }

    fn synthesize_recipe(entry: &RationaleTraceEntry) -> Recipe {
        let step = Step {
            description: format!("Adjust parameter for {}", entry.option),
            duration_minutes: 15,
            rationale: entry.why.clone(),
        };
        Recipe {
            steps: vec![step],
            constraints: vec![crate::Constraint {
                name: "safety-window".to_string(),
                description: "Operate within validated bounds".to_string(),
                satisfied: true,
            }],
            safety_interlocks: vec!["interlock-1".to_string()],
        }
    }
}

impl Planner for DeterministicPlanner {
    fn plan(&self, objective: Objective) -> Result<Plan, CoreError> {
        let span = info_span!("planner.evaluate", objective = %objective.description);
        let _guard = span.enter();
        if objective.target_metrics.is_empty() {
            return Err(CoreError::InvalidObjective("missing target metrics".into()));
        }
        let rationale = self.score_options(&objective);
        let candidate_recipes = rationale.iter().map(Self::synthesize_recipe).collect();
        Ok(Plan {
            id: Uuid::new_v4(),
            objective,
            candidate_recipes,
            rationale_trace: rationale,
        })
    }
}
