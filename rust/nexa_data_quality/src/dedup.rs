use bloom::BloomFilter;
use arrow::record_batch::RecordBatch;
use arrow::array::{Array, StringArray, BooleanArray};
use std::sync::Mutex;

pub struct DedupState {
    filter: Mutex<BloomFilter>,
}

impl DedupState {
    pub fn new(capacity: usize, error_rate: f32) -> Self {
        // bloom crate API might differ, using generic idea
        Self {
            filter: Mutex::new(BloomFilter::with_rate(error_rate, capacity as u32)),
        }
    }

    pub fn check_and_add(&self, text: &str) -> bool {
        let mut filter = self.filter.lock().unwrap();
        if filter.contains(&text) {
            true // Duplicate
        } else {
            filter.insert(&text);
            false // New
        }
    }
}

pub fn deduplicate_record_batch(batch: &RecordBatch, text_col: &str, state: &DedupState) -> RecordBatch {
    let text_array = batch.column_by_name(text_col)
        .expect("Text column not found")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Column is not string");

    let mut keep_indices = Vec::with_capacity(batch.num_rows());

    for i in 0..batch.num_rows() {
        if text_array.is_null(i) {
            keep_indices.push(false);
            continue;
        }

        let text = text_array.value(i);
        let is_dupe = state.check_and_add(text);
        keep_indices.push(!is_dupe);
    }

    let boolean_array = BooleanArray::from(keep_indices);
    arrow::compute::filter_record_batch(batch, &boolean_array).unwrap()
}

