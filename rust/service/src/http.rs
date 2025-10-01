use crate::AppState;
use axum::{extract::State, response::IntoResponse, routing::post, Json, Router};
use periodic_core::Objective;
use prometheus::Encoder;
use serde::{Deserialize, Serialize};
use tracing::{info_span, Instrument};
use utoipa::{OpenApi, ToSchema};
use uuid::Uuid;

#[derive(OpenApi)]
#[openapi(
    paths(plan, submit_qc, start_run),
    components(schemas(PlanRequest, PlanResponse, ObjectiveDto, MetricDto, RationaleEntryDto, QCRequest, QCResponse)),
    tags((name = "periodic", description = "Planning APIs"))
)]
pub struct ApiDoc;

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct MetricDto {
    pub name: String,
    pub target: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ObjectiveDto {
    pub description: String,
    pub metrics: Vec<MetricDto>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct PlanRequest {
    pub objective: ObjectiveDto,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct PlanResponse {
    pub id: String,
    pub objective: ObjectiveDto,
    pub rationale: Vec<RationaleEntryDto>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct RationaleEntryDto {
    pub option: String,
    pub score: f64,
    pub why: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct RunRequest {
    pub instrument_id: String,
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct RunEventDto {
    pub run_id: String,
    pub state: String,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct QCRequest {
    pub items: Vec<String>,
    pub notes: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct QCResponse {
    pub id: String,
    pub outcome: String,
    pub negatives: Vec<String>,
}

pub fn build_routes(state: AppState) -> Router {
    Router::new()
        .route("/v1/plan", post(plan))
        .route("/v1/run", post(start_run))
        .route("/v1/qc", post(submit_qc))
        .route("/docs", axum::routing::get(serve_docs))
        .route("/docs/openapi.json", axum::routing::get(openapi_json))
        .with_state(state)
}

#[utoipa::path(post, path = "/v1/plan", request_body = PlanRequest, responses((status = 200, body = PlanResponse)))]
pub async fn plan(
    State(state): State<AppState>,
    Json(payload): Json<PlanRequest>,
) -> Result<Json<PlanResponse>, axum::http::StatusCode> {
    let span = info_span!("http.plan", objective = %payload.objective.description);
    async move {
        let planner = &state.planner;
        let objective = Objective {
            description: payload.objective.description.clone(),
            target_metrics: payload
                .objective
                .metrics
                .iter()
                .map(|m| (m.name.clone(), m.target))
                .collect(),
        };
        let plan = planner
            .plan(objective)
            .map_err(|_| axum::http::StatusCode::BAD_REQUEST)?;
        Ok(Json(PlanResponse {
            id: plan.id.to_string(),
            objective: payload.objective,
            rationale: plan
                .rationale_trace
                .into_iter()
                .map(|entry| RationaleEntryDto {
                    option: entry.option,
                    score: entry.score,
                    why: entry.why,
                })
                .collect(),
        }))
    }
    .instrument(span)
    .await
}

#[utoipa::path(post, path = "/v1/qc", request_body = QCRequest, responses((status = 200, body = QCResponse)))]
pub async fn submit_qc(
    State(state): State<AppState>,
    Json(payload): Json<QCRequest>,
) -> Result<Json<QCResponse>, axum::http::StatusCode> {
    let checklist = periodic_core::qc::checklist::Checklist {
        id: Uuid::new_v4(),
        name: "http".to_string(),
        items: payload
            .items
            .iter()
            .map(|item| periodic_core::qc::checklist::ChecklistItem {
                description: item.clone(),
                required: true,
                satisfied: true,
            })
            .collect(),
    };
    let qc = checklist
        .run(payload.notes.clone(), vec![])
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(QCResponse {
        id: qc.id.to_string(),
        outcome: format!("{:?}", qc.outcome),
        negatives: qc.negatives_captured,
    }))
}

#[utoipa::path(post, path = "/v1/run", request_body = RunRequest, responses((status = 200)))]
pub async fn start_run(
    State(state): State<AppState>,
    Json(payload): Json<RunRequest>,
) -> Result<Json<RunEventDto>, axum::http::StatusCode> {
    let id = Uuid::new_v4();
    state
        .repository
        .record_run(id, &payload.instrument_id)
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(RunEventDto {
        run_id: id.to_string(),
        state: "started".to_string(),
        message: "Run accepted".to_string(),
    }))
}

pub async fn serve_docs() -> impl IntoResponse {
    axum::response::Html(utoipa_swagger_ui::SwaggerUi::new("/docs/openapi.json").to_html())
}

pub async fn openapi_json() -> impl IntoResponse {
    Json(ApiDoc::openapi())
}

pub async fn healthz() -> &'static str {
    "ok"
}

pub async fn readyz() -> &'static str {
    "ready"
}

pub async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = state
        .exporter
        .registry()
        .gather()
        .into_iter()
        .map(prometheus::proto::MetricFamily::to_owned)
        .collect::<Vec<_>>();
    let mut buffer = vec![];
    let encoder = prometheus::TextEncoder::new();
    encoder
        .encode(&metrics, &mut buffer)
        .expect("encode metrics");
    (axum::http::StatusCode::OK, buffer)
}
