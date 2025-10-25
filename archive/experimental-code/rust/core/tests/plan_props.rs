use indexmap::IndexMap;
use periodic_core::{plan::planner::DeterministicPlanner, Objective};
use proptest::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

proptest! {
    #[test]
    fn rationale_is_never_empty(description in "[a-zA-Z0-9 ]{1,16}") {
        let mut metrics = IndexMap::new();
        metrics.insert("yield".to_string(), 0.8);
        let planner = DeterministicPlanner::new(StdRng::seed_from_u64(1));
        let objective = Objective { description, target_metrics: metrics };
        let plan = planner.plan(objective).expect("plan");
        prop_assert!(!plan.rationale_trace.is_empty());
    }
}
