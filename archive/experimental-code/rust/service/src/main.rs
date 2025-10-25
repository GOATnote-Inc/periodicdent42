use clap::Parser;
use service::{
    build_http_router,
    config::{Cli, Settings},
    serve_grpc, telemetry, AppState,
};
use std::sync::Arc;
use tokio::signal;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let settings = Settings::load(&cli).unwrap_or_default();

    let exporter = telemetry::init_tracing("periodic-service", &settings.telemetry);
    let planner = periodic_core::CorePlanner::new(42);
    let state = AppState::new(Arc::new(planner), exporter);

    let http_addr: std::net::SocketAddr = settings.http.addr.parse()?;
    let grpc_addr: std::net::SocketAddr = settings.grpc.addr.parse()?;

    let app = build_http_router(state.clone());
    let http_server = axum::Server::bind(&http_addr).serve(app.into_make_service());

    let grpc_state = state.clone();
    let grpc_future = serve_grpc(grpc_state, grpc_addr);

    info!("service starting", %http_addr, %grpc_addr);

    tokio::select! {
        result = http_server => {
            if let Err(err) = result {
                warn!(?err, "http server failed");
            }
        }
        result = grpc_future => {
            if let Err(err) = result {
                warn!(?err, "grpc server failed");
            }
        }
        _ = shutdown_signal() => {
            info!("shutdown signal received");
        }
    }

    telemetry::shutdown();
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install CTRL+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("install signal")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
