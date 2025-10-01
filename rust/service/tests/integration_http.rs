use opentelemetry_prometheus::exporter;
use service::{build_http_router, AppState};
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
async fn plan_endpoint_returns_plan() {
    let _ = tracing_subscriber::fmt::try_init();
    let exporter = exporter().init();
    let state = AppState::new(Arc::new(periodic_core::CorePlanner::new(1)), exporter);
    let app = build_http_router(state);
    let body = serde_json::json!({
        "objective": {
            "description": "test",
            "metrics": [{"name": "yield", "target": 0.8}]
        }
    });
    let response = app
        .oneshot(
            http::Request::post("/v1/plan")
                .header(http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(response.status().is_success());
}
