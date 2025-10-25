#![no_main]
use indexmap::IndexMap;
use libfuzzer_sys::fuzz_target;
use periodic_core::Objective;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let description = String::from_utf8_lossy(data).into_owned();
    let mut metrics = IndexMap::new();
    metrics.insert("yield".to_string(), (data[0] as f64) / 255.0);
    let objective = Objective {
        description,
        target_metrics: metrics,
    };
    assert!(!objective.description.is_empty());
});
