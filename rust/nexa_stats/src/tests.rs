use statrs::distribution::{ContinuousCDF, Normal};
use pyo3::prelude::*;

pub fn ks_statistic(data1: &[f64], data2: &[f64]) -> f64 {
    // Simplified KS statistic calculation
    // Real implementation would sort and find max diff of ECDFs
    // Using statrs or manual calculation
    
    let mut d1 = data1.to_vec();
    let mut d2 = data2.to_vec();
    d1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // Placeholder logic for brevity in this snippet, usually O(N+M) scan
    0.05 // Dummy result
}

pub fn population_stability_index(expected: &[f64], actual: &[f64], buckets: usize) -> f64 {
    // Bin data, calculate PSI = sum((actual_prop - expected_prop) * ln(actual_prop / expected_prop))
    0.1 // Dummy result
}

pub fn chi_square_test(observed: &[f64], expected: &[f64]) -> f64 {
    // sum((obs - exp)^2 / exp)
    let mut stat = 0.0;
    for (o, e) in observed.iter().zip(expected.iter()) {
        if *e > 0.0 {
            stat += (o - e).powi(2) / e;
        }
    }
    stat
}
