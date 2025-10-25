use crate::{errors::CoreError, QCCheck, QCOutcome};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChecklistItem {
    pub description: String,
    pub required: bool,
    pub satisfied: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Checklist {
    pub id: Uuid,
    pub name: String,
    pub items: Vec<ChecklistItem>,
}

impl Checklist {
    pub fn run(
        self,
        notes: impl Into<String>,
        negatives: Vec<String>,
    ) -> Result<QCCheck, CoreError> {
        let outcome = if self.items.iter().all(|item| item.satisfied) {
            QCOutcome::Pass
        } else if self
            .items
            .iter()
            .any(|item| item.required && !item.satisfied)
        {
            QCOutcome::Fail
        } else {
            QCOutcome::NeedsReview
        };
        Ok(QCCheck {
            id: self.id,
            checklist_items: self
                .items
                .iter()
                .map(|item| {
                    format!(
                        "{} - {}",
                        item.description,
                        if item.satisfied {
                            "ok"
                        } else {
                            "needs attention"
                        }
                    )
                })
                .collect(),
            outcome,
            notes: notes.into(),
            negatives_captured: negatives,
        })
    }
}
