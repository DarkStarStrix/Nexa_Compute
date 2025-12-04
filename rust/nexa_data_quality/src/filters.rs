use arrow::record_batch::RecordBatch;
use arrow::array::{Array, StringArray, BooleanArray};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize)]
pub struct FilterConfig {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub bad_words: Option<Vec<String>>,
}

#[derive(Debug)]
pub enum RejectReason {
    TooShort,
    TooLong,
    ContainsBadWord,
}

pub fn filter_record_batch(batch: &RecordBatch, config: &FilterConfig, text_col: &str) -> (RecordBatch, Vec<Option<RejectReason>>) {
    let text_array = batch.column_by_name(text_col)
        .expect("Text column not found")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Column is not string");

    let mut keep_indices = Vec::with_capacity(batch.num_rows());
    let mut reject_reasons = Vec::with_capacity(batch.num_rows());

    for i in 0..batch.num_rows() {
        if text_array.is_null(i) {
            reject_reasons.push(None); // Or reject?
            continue; // Skip nulls
        }

        let text = text_array.value(i);
        let mut reason = None;

        if let Some(min_len) = config.min_length {
            if text.len() < min_len {
                reason = Some(RejectReason::TooShort);
            }
        }

        if reason.is_none() {
            if let Some(max_len) = config.max_length {
                if text.len() > max_len {
                    reason = Some(RejectReason::TooLong);
                }
            }
        }

        if reason.is_none() {
            if let Some(bad_words) = &config.bad_words {
                for word in bad_words {
                    if text.contains(word) {
                        reason = Some(RejectReason::ContainsBadWord);
                        break;
                    }
                }
            }
        }

        if reason.is_none() {
            keep_indices.push(true);
            reject_reasons.push(None);
        } else {
            keep_indices.push(false);
            reject_reasons.push(reason);
        }
    }

    let boolean_array = BooleanArray::from(keep_indices);
    let filtered_batch = arrow::compute::filter_record_batch(batch, &boolean_array).unwrap();
    
    (filtered_batch, reject_reasons)
}

