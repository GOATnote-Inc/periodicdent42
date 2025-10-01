use clap::Parser;
use secrecy::SecretString;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct TelemetryConfig {
    pub otlp_endpoint: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpConfig {
    pub addr: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GrpcConfig {
    pub addr: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub url: Option<SecretString>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Settings {
    pub http: HttpConfig,
    pub grpc: GrpcConfig,
    pub telemetry: TelemetryConfig,
    pub database: DatabaseConfig,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            http: HttpConfig {
                addr: "0.0.0.0:8080".into(),
            },
            grpc: GrpcConfig {
                addr: "0.0.0.0:50051".into(),
            },
            telemetry: TelemetryConfig {
                otlp_endpoint: None,
            },
            database: DatabaseConfig { url: None },
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Experiment service")]
pub struct Cli {
    #[arg(long, env = "SERVICE_CONFIG", default_value = "configs/service.yaml")]
    pub config: PathBuf,
}

impl Settings {
    pub fn load(cli: &Cli) -> anyhow::Result<Self> {
        let mut settings = config::Config::builder()
            .add_source(config::File::with_name(&cli.config.display().to_string()).required(false))
            .add_source(config::Environment::with_prefix("SERVICE").separator("__"))
            .build()?;
        settings.try_deserialize().map_err(Into::into)
    }
}
