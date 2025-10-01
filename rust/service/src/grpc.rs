use crate::AppState;
use indexmap::IndexMap;
use periodic_core::Objective;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::info_span;
use uuid::Uuid;

pub mod proto {
    tonic::include_proto!("periodic");
}

use proto::experiment_service_server::{ExperimentService, ExperimentServiceServer};
use proto::{PlanRequest, PlanResponse, QCRequest, QCResponse, RunRequest, RunStatus};

#[derive(Clone)]
pub struct ExperimentGrpc {
    state: AppState,
}

impl ExperimentGrpc {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl ExperimentService for ExperimentGrpc {
    type StartRunStream = ReceiverStream<Result<RunStatus, Status>>;

    async fn plan(&self, request: Request<PlanRequest>) -> Result<Response<PlanResponse>, Status> {
        let span = info_span!("grpc.plan");
        let _guard = span.enter();
        let payload = request.into_inner();
        let mut metrics = IndexMap::new();
        for metric in payload
            .objective
            .as_ref()
            .map(|o| o.metrics.clone())
            .unwrap_or_default()
        {
            metrics.insert(metric.name, metric.target);
        }
        let objective = Objective {
            description: payload
                .objective
                .as_ref()
                .map(|o| o.description.clone())
                .unwrap_or_default(),
            target_metrics: metrics,
        };
        let plan = self
            .state
            .planner
            .plan(objective)
            .map_err(|e| Status::internal(e.to_string()))?;
        let response = PlanResponse {
            id: plan.id.to_string(),
            objective: Some(proto::Objective {
                description: plan.objective.description,
                metrics: plan
                    .objective
                    .target_metrics
                    .into_iter()
                    .map(|(name, target)| proto::ObjectiveMetric { name, target })
                    .collect(),
            }),
            rationale: plan
                .rationale_trace
                .into_iter()
                .map(|entry| proto::RationaleEntry {
                    option: entry.option,
                    score: entry.score,
                    why: entry.why,
                })
                .collect(),
        };
        Ok(Response::new(response))
    }

    async fn start_run(
        &self,
        request: Request<RunRequest>,
    ) -> Result<Response<Self::StartRunStream>, Status> {
        let payload = request.into_inner();
        let id = Uuid::new_v4();
        self.state
            .repository
            .record_run(id, &payload.instrument_id)
            .map_err(|err| Status::internal(err.to_string()))?;
        let (tx, rx) = mpsc::channel(4);
        let status = RunStatus {
            run_id: id.to_string(),
            state: "started".into(),
            message: "Run accepted".into(),
        };
        tx.send(Ok(status)).await.unwrap();
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn submit_qc(&self, request: Request<QCRequest>) -> Result<Response<QCResponse>, Status> {
        let payload = request.into_inner();
        let checklist = periodic_core::qc::checklist::Checklist {
            id: Uuid::new_v4(),
            name: "grpc".to_string(),
            items: payload
                .items
                .into_iter()
                .map(|item| periodic_core::qc::checklist::ChecklistItem {
                    description: item,
                    required: true,
                    satisfied: true,
                })
                .collect(),
        };
        let qc = checklist
            .run(payload.notes, vec![])
            .map_err(|err| Status::internal(err.to_string()))?;
        Ok(Response::new(QCResponse {
            id: qc.id.to_string(),
            outcome: format!("{:?}", qc.outcome),
            negatives: qc.negatives_captured,
        }))
    }
}

pub use proto::experiment_service_server::ExperimentServiceServer;
