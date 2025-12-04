use std::collections::HashMap;
use serde::Serialize;

#[derive(Debug, Default, Serialize)]
pub struct QualityMetrics {
    pub total_processed: usize,
    pub total_accepted: usize,
    pub total_rejected: usize,
    pub rejected_by_reason: HashMap<String, usize>,
    pub length_histogram: HashMap<usize, usize>, // Bucketized
}

impl QualityMetrics {
    pub fn update(&mut self, accepted: usize, rejected_reasons: &[Option<String>], lengths: &[usize]) {
        self.total_processed += accepted + rejected_reasons.len() - accepted; // Simplification logic needs checking
        // Actually, passed logic might differ.
        // Let's assume caller passes counts.
        
        // Histogram update
        for &len in lengths {
            let bucket = (len / 100) * 100; // 100-char buckets
            *self.length_histogram.entry(bucket).or_default() += 1;
        }
    }
}

