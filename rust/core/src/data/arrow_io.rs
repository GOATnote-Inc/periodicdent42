#![cfg(feature = "arrow")]

use arrow::array::{Float64Array, StringArray, StructArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_ipc::writer::FileWriter;
use std::io::Write;

use crate::{Plan, RationaleTraceEntry};

pub fn plan_to_record_batch(plan: &Plan) -> RecordBatch {
    let option = StringArray::from(
        plan.rationale_trace
            .iter()
            .map(|entry| entry.option.clone())
            .collect::<Vec<_>>(),
    );
    let score = Float64Array::from(
        plan.rationale_trace
            .iter()
            .map(|entry| entry.score)
            .collect::<Vec<_>>(),
    );
    let why = StringArray::from(
        plan.rationale_trace
            .iter()
            .map(|entry| entry.why.clone())
            .collect::<Vec<_>>(),
    );
    let struct_array = StructArray::from(vec![
        (
            Field::new("option", DataType::Utf8, false),
            Arc::new(option) as _,
        ),
        (
            Field::new("score", DataType::Float64, false),
            Arc::new(score) as _,
        ),
        (Field::new("why", DataType::Utf8, false), Arc::new(why) as _),
    ]);
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "rationale",
            DataType::Struct(struct_array.fields().to_vec()),
            false,
        )])),
        vec![Arc::new(struct_array)],
    )
    .expect("valid batch")
}

pub fn write_plan<W: Write>(plan: &Plan, writer: W) {
    let batch = plan_to_record_batch(plan);
    let mut writer = FileWriter::try_new(writer, batch.schema()).expect("writer");
    writer.write(&batch).expect("write batch");
    writer.finish().expect("finish writer");
}

use std::sync::Arc;
