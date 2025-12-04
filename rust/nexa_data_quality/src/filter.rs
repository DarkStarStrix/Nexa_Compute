use arrow::record_batch::RecordBatch;
use arrow::array::{StringArray, BooleanArray};
use arrow::compute::filter_record_batch;
use pyo3.prelude::*;
use crate::config::{FilterConfig, FilterStats};
use bloom::BloomFilter;
use std::collections::{HashSet, HashMap};
use xxhash_rust::xxh3::xxh3_64;

pub struct DedupState {
    bloom: BloomFilter,
    seen_hashes: HashSet<u64>,
}

impl DedupState {
    pub fn new(capacity: usize, error_rate: f32) -> Self {
        DedupState {
            bloom: BloomFilter::with_rate(error_rate, capacity as u32),
            seen_hashes: HashSet::new(),
        }
    }
    
    pub fn check_and_insert(&mut self, text: &str) -> bool {
        let hash = xxh3_64(text.as_bytes());
        if self.seen_hashes.contains(&hash) {
            return true;
        }
        self.seen_hashes.insert(hash);
        self.bloom.insert(&hash);
        false
    }
}

// Changed signature to take Option<&mut DedupState>
pub fn filter_record_batch_internal(batch: &RecordBatch, config: &FilterConfig, dedup_state: Option<&mut DedupState>) -> PyResult<(RecordBatch, RecordBatch, FilterStats)> {
    let num_rows = batch.num_rows();
    let col_idx = batch.schema().index_of(&config.text_column).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let col = batch.column(col_idx);
    let string_col = col.as_any().downcast_ref::<StringArray>().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Column is not string"))?;
    
    let mut stats = FilterStats {
        total: num_rows,
        kept: 0,
        rejected_length: 0,
        rejected_pattern: 0,
        rejected_dedup: 0,
        length_histogram: HashMap::new(),
    };
    
    let mut keep_mask = Vec::with_capacity(num_rows);
    
    // We need to allow mutable access to dedup_state.
    // Since Option is Copy/Clone if content is, but &mut T is not.
    // We consume the Option<&mut T> which is fine.
    // However, we are inside a loop. We need to re-borrow?
    // Actually, we have the reference. We can just use it.
    // But we cannot move it into the loop if we need it across iterations.
    // Actually `dedup_state` is `Option<&mut DedupState>`.
    // We can't easily pattern match `Some(ref mut s)` inside loop on a `&mut`?
    // Let's just unwrap it outside if present.
    
    // Use a pointer or ref wrapper to avoid borrow checker issues with the Option
    // Ideally we just pass `&mut DedupState` and if it's None we don't call it.
    // But the function signature has Option.
    
    // Let's refactor to separate dedup logic.
    // Or just:
    // let mut dedup_ref = dedup_state; 
    // Wait, if I pass `dedup_state` which is `Option<&mut ...>`, I can match on it.
    
    // Workaround for "cannot borrow `*s` as mutable more than once" inside loop?
    // `dedup_state` is owned (it's an Option of a ref).
    // But `&mut T` is not Copy.
    // So we cannot iterate and use it?
    // Yes we can reborrow `&mut *s`.
    
    // Let's try:
    // match dedup_state { Some(s) => ... use s.check_and_insert ... }
    // This consumes s (the reference).
    // We need `as_deref_mut`? No, it's already `&mut`.
    // We need `reborrow` logic.
    
    // Simplified:
    // let mut dedup_wrapper = dedup_state; 
    // Inside loop:
    // if let Some(s) = &mut dedup_wrapper { ... }
    
    let mut dedup_wrapper = dedup_state;

    for i in 0..num_rows {
        if string_col.is_null(i) {
            keep_mask.push(false);
            stats.rejected_length += 1;
            continue;
        }
        
        let text = string_col.value(i);
        let len = text.len();
        
        let bucket = (len / 100) * 100;
        *stats.length_histogram.entry(bucket).or_insert(0) += 1;
        
        if let Some(min) = config.min_length {
            if len < min {
                keep_mask.push(false);
                stats.rejected_length += 1;
                continue;
            }
        }
        if let Some(max) = config.max_length {
            if len > max {
                keep_mask.push(false);
                stats.rejected_length += 1;
                continue;
            }
        }
        
        let mut bad = false;
        for pattern in &config.bad_patterns {
            if text.contains(pattern) {
                bad = true;
                break;
            }
        }
        if bad {
            keep_mask.push(false);
            stats.rejected_pattern += 1;
            continue;
        }
        
        if config.dedup_enabled {
            if let Some(state) = &mut dedup_wrapper {
                if state.check_and_insert(text) {
                    keep_mask.push(false);
                    stats.rejected_dedup += 1;
                    continue;
                }
            }
        }
        
        keep_mask.push(true);
        stats.kept += 1;
    }
    
    let boolean_array = BooleanArray::from(keep_mask.clone());
    let kept_batch = filter_record_batch(batch, &boolean_array).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let reject_mask: Vec<bool> = keep_mask.iter().map(|&b| !b).collect();
    let reject_array = BooleanArray::from(reject_mask);
    let rejected_batch = filter_record_batch(batch, &reject_array).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    Ok((kept_batch, rejected_batch, stats))
}
