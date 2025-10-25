pub mod config;
pub mod grpc;
pub mod http;
pub mod repo;
pub mod telemetry;

use axum::{routing::get, Router};
use opentelemetry_prometheus::PrometheusExporter;
use periodic_core::CorePlanner;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::info;

pub type SharedPlanner = Arc<CorePlanner>;

#[derive(Clone)]
pub struct AppState {
    pub planner: SharedPlanner,
    pub repository: repo::InMemoryRepository,
    pub exporter: Arc<PrometheusExporter>,
}

impl AppState {
    pub fn new(planner: SharedPlanner, exporter: PrometheusExporter) -> Self {
        Self {
            planner,
            repository: repo::InMemoryRepository::default(),
            exporter: Arc::new(exporter),
        }
    }
}

pub fn build_http_router(state: AppState) -> Router {
    Router::new()
        .merge(http::build_routes(state.clone()))
        .route("/healthz", get(http::healthz))
        .route("/readyz", get(http::readyz))
        .route("/metrics", get(http::metrics_handler))
        .with_state(state)
}

pub async fn serve_grpc(state: AppState, addr: std::net::SocketAddr) -> anyhow::Result<()> {
    let svc = grpc::ExperimentGrpc::new(state);
    info!("starting grpc", %addr);
    Server::builder()
        .add_service(grpc::ExperimentServiceServer::new(svc))
        .serve(addr)
        .await?;
    Ok(())
}
