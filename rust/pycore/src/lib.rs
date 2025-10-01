use indexmap::IndexMap;
use periodic_core::{CorePlanner, Objective};
use pyo3::prelude::*;
use std::sync::OnceLock;
use uuid::Uuid;

static PLANNER: OnceLock<CorePlanner> = OnceLock::new();

fn planner() -> &'static CorePlanner {
    PLANNER.get_or_init(|| CorePlanner::new(1337))
}

#[pyfunction]
fn plan(objective: &PyAny) -> PyResult<PyObject> {
    let description: String = objective.get_item("description")?.extract()?;
    let metrics_py = objective.get_item("metrics")?;
    let mut metrics = IndexMap::new();
    for item in metrics_py.iter()? {
        let item = item?;
        let name: String = item.get_item("name")?.extract()?;
        let target: f64 = item.get_item("target")?.extract()?;
        metrics.insert(name, target);
    }
    let plan = planner()
        .plan(Objective {
            description,
            target_metrics: metrics,
        })
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
    Python::with_gil(|py| Ok(serde_json::to_value(&plan)?.into_py(py)))
}

#[pyfunction]
fn qc_check(items: Vec<String>, notes: Option<String>) -> PyResult<PyObject> {
    let checklist = periodic_core::qc::checklist::Checklist {
        id: Uuid::new_v4(),
        name: "pycore".into(),
        items: items
            .into_iter()
            .map(|item| periodic_core::qc::checklist::ChecklistItem {
                description: item,
                required: true,
                satisfied: true,
            })
            .collect(),
    };
    let qc = checklist
        .run(notes.unwrap_or_default(), vec![])
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
    Python::with_gil(|py| Ok(serde_json::to_value(&qc)?.into_py(py)))
}

#[pymodule]
fn pycore(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(qc_check, m)?)?;
    Ok(())
}
