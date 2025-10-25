use indexmap::IndexMap;
use periodic_core::{CorePlanner, Objective};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct PlannerHandle {
    inner: CorePlanner,
}

#[wasm_bindgen]
impl PlannerHandle {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64) -> PlannerHandle {
        #[cfg(feature = "console")]
        console_error_panic_hook::set_once();
        PlannerHandle {
            inner: CorePlanner::new(seed),
        }
    }

    #[wasm_bindgen]
    pub fn plan(&self, objective: JsValue) -> Result<JsValue, JsValue> {
        let value: serde_json::Value = objective.into_serde().map_err(|e| e.to_string())?;
        let description = value
            .get("description")
            .and_then(|d| d.as_str())
            .unwrap_or_default()
            .to_string();
        let mut metrics = IndexMap::new();
        if let Some(entries) = value.get("metrics").and_then(|m| m.as_array()) {
            for entry in entries {
                if let (Some(name), Some(target)) = (entry.get("name"), entry.get("target")) {
                    if let (Some(name), Some(target)) = (name.as_str(), target.as_f64()) {
                        metrics.insert(name.to_string(), target);
                    }
                }
            }
        }
        let plan = self
            .inner
            .plan(Objective {
                description,
                target_metrics: metrics,
            })
            .map_err(|e| e.to_string())?;
        JsValue::from_serde(&plan).map_err(|e| e.to_string().into())
    }
}
