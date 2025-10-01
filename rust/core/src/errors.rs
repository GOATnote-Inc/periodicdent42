use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid objective: {0}")]
    InvalidObjective(String),
    #[error("planning failure: {0}")]
    PlanningFailure(String),
    #[error("data access failure: {0}")]
    DataAccess(String),
}

pub type CoreResult<T> = Result<T, CoreError>;
