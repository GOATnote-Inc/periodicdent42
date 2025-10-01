use opentelemetry::global;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_prometheus::PrometheusExporter;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::config::TelemetryConfig;

pub fn init_tracing(service_name: &str, telemetry: &TelemetryConfig) -> PrometheusExporter {
    let exporter = opentelemetry_prometheus::exporter().init();

    let tracer = if let Some(endpoint) = &telemetry.otlp_endpoint {
        let otlp_exporter = opentelemetry_otlp::new_exporter()
            .tonic()
            .with_endpoint(endpoint);
        let trace_exporter = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(otlp_exporter)
            .install_batch(opentelemetry::runtime::Tokio)
            .expect("otlp");
        trace_exporter
    } else {
        opentelemetry::sdk::trace::TracerProvider::builder()
            .with_simple_exporter(opentelemetry_stdout::SpanExporter::default())
            .build()
            .tracer(service_name)
    };

    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer().json())
        .with(otel_layer)
        .init();

    exporter
}

pub fn shutdown() {
    global::shutdown_tracer_provider();
}
