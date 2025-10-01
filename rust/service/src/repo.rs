use anyhow::Result;
use chrono::Utc;
use futures::executor;
use indexmap::IndexMap;
use periodic_core::Run;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;
use uuid::Uuid;

#[async_trait::async_trait]
pub trait RunRepository: Send + Sync {
    async fn record_run(&self, run: Run) -> Result<()>;
    async fn latest(&self) -> Result<Option<Run>>;
}

#[derive(Default, Clone)]
pub struct InMemoryRepository {
    inner: Arc<RwLock<HashMap<Uuid, Run>>>,
}

impl InMemoryRepository {
    pub fn record_run(&self, id: Uuid, instrument: &str) -> Result<()> {
        let mut guard = executor::block_on(self.inner.write());
        guard.insert(
            id,
            Run {
                id,
                instrument_id: instrument.to_string(),
                params: IndexMap::new(),
                started_at: Utc::now(),
                finished_at: None,
                artifacts: vec![],
            },
        );
        Ok(())
    }
}

#[cfg(feature = "postgres")]
pub mod postgres {
    use super::*;
    use sqlx::PgPool;

    #[derive(Clone)]
    pub struct PostgresRepository {
        pool: PgPool,
    }

    impl PostgresRepository {
        pub fn new(pool: PgPool) -> Self {
            Self { pool }
        }

        pub async fn record_run(&self, id: Uuid, instrument: &str) -> Result<()> {
            sqlx::query!(
                "INSERT INTO runs (id, instrument_id) VALUES ($1, $2)",
                id,
                instrument
            )
            .execute(&self.pool)
            .await?;
            Ok(())
        }
    }
}
