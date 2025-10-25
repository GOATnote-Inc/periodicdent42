#![no_main]
use indexmap::IndexMap;
use libfuzzer_sys::fuzz_target;
use periodic_core::{plan::planner::DeterministicPlanner, Objective};
use rand::{rngs::StdRng, SeedableRng};

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let seed = u64::from_le_bytes({
        let mut bytes = [0u8; 8];
        for (i, b) in data.iter().take(8).enumerate() {
            bytes[i] = *b;
        }
        bytes
    });
    let mut metrics = IndexMap::new();
    metrics.insert("yield".to_string(), (data[0] as f64) / 255.0);
    let planner = DeterministicPlanner::new(StdRng::seed_from_u64(seed));
    let objective = Objective {
        description: format!("fuzz-{}", data.len()),
        target_metrics: metrics,
    };
    let _ = planner.plan(objective);
});
