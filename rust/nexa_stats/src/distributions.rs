use statrs::statistics::Distribution;
use statrs::distribution::{Normal, Continuous};
use pyo3::prelude::*;
use std::f64;

pub fn kolmogorov_smirnov_test(data1: &[f64], data2: &[f64]) -> f64 {
    if data1.is_empty() || data2.is_empty() {
        return 0.0;
    }
    
    let mut d1 = data1.to_vec();
    let mut d2 = data2.to_vec();
    
    d1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    d2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let n1 = d1.len() as f64;
    let n2 = d2.len() as f64;
    
    let combined = {
        let mut c = d1.clone();
        c.extend_from_slice(&d2);
        c.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        c.dedup();
        c
    };
    
    let mut max_diff = 0.0;
    
    for x in combined {
        let cdf1 = d1.iter().filter(|&&v| v <= x).count() as f64 / n1;
        let cdf2 = d2.iter().filter(|&&v| v <= x).count() as f64 / n2;
        let diff = (cdf1 - cdf2).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    
    max_diff
}

pub fn population_stability_index(expected: &[f64], actual: &[f64], buckets: usize) -> f64 {
    // Simplified bucket strategy: create equal-width buckets based on expected range
    if expected.is_empty() || actual.is_empty() {
        return 0.0;
    }
    
    let min = expected.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = expected.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    if (max - min).abs() < 1e-9 {
        return 0.0;
    }
    
    let width = (max - min) / buckets as f64;
    let mut expected_counts = vec![0.0; buckets];
    let mut actual_counts = vec![0.0; buckets];
    
    // Helper to find bucket index
    let get_bucket = |val: f64| -> usize {
        if val <= min { return 0; }
        if val >= max { return buckets - 1; }
        ((val - min) / width).floor() as usize
    };
    
    for &val in expected {
        let idx = get_bucket(val);
        if idx < buckets { expected_counts[idx] += 1.0; }
    }
    
    for &val in actual {
        let idx = get_bucket(val);
        if idx < buckets { actual_counts[idx] += 1.0; }
    }
    
    let total_expected = expected.len() as f64;
    let total_actual = actual.len() as f64;
    
    let mut psi_value = 0.0;
    for i in 0..buckets {
        let pct_expected = (expected_counts[i] + 1.0) / (total_expected + buckets as f64); // Smoothing
        let pct_actual = (actual_counts[i] + 1.0) / (total_actual + buckets as f64);
        
        psi_value += (pct_actual - pct_expected) * (pct_actual / pct_expected).ln();
    }
    
    psi_value
}

