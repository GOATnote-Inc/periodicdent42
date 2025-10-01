use opentelemetry_prometheus::exporter;
use service::{grpc::ExperimentGrpc, AppState};
use std::sync::Arc;
use tonic::Request;

#[tokio::test]
async fn grpc_plan_returns_plan() {
    let exporter = exporter().init();
    let state = AppState::new(Arc::new(periodic_core::CorePlanner::new(2)), exporter);
    let grpc = ExperimentGrpc::new(state);
    let request = tonic::Request::new(service::grpc::proto::PlanRequest {
        objective: Some(service::grpc::proto::Objective {
            description: "test".into(),
            metrics: vec![service::grpc::proto::ObjectiveMetric {
                name: "yield".into(),
                target: 0.9,
            }],
        }),
    });
    let response = grpc.plan(request).await.unwrap().into_inner();
    assert_eq!(response.objective.unwrap().description, "test");
}
