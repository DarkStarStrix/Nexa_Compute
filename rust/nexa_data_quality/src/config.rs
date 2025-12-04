use pyo3.prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct FilterConfig {
    #[pyo3(get, set)]
    pub min_length: Option<usize>,
    #[pyo3(get, set)]
    pub max_length: Option<usize>,
    #[pyo3(get, set)]
    pub bad_patterns: Vec<String>,
    #[pyo3(get, set)]
    pub text_column: String,
    #[pyo3(get, set)]
    pub dedup_enabled: bool,
}

#[pymethods]
impl FilterConfig {
    #[new]
    fn new(text_column: String, min_length: Option<usize>, max_length: Option<usize>, bad_patterns: Vec<String>, dedup_enabled: bool) -> Self {
        FilterConfig {
            text_column,
            min_length,
            max_length,
            bad_patterns,
            dedup_enabled,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct FilterStats {
    #[pyo3(get, set)]
    pub total: usize,
    #[pyo3(get, set)]
    pub kept: usize,
    #[pyo3(get, set)]
    pub rejected_length: usize,
    #[pyo3(get, set)]
    pub rejected_pattern: usize,
    #[pyo3(get, set)]
    pub rejected_dedup: usize,
    #[pyo3(get, set)]
    pub length_histogram: HashMap<usize, usize>, // bucket -> count
}

#[pymethods]
impl FilterStats {
    #[new]
    fn new() -> Self {
        FilterStats {
            total: 0,
            kept: 0,
            rejected_length: 0,
            rejected_pattern: 0,
            rejected_dedup: 0,
            length_histogram: HashMap::new(),
        }
    }
}

impl FilterStats {
    pub fn merge(&mut self, other: &FilterStats) {
        self.total += other.total;
        self.kept += other.kept;
        self.rejected_length += other.rejected_length;
        self.rejected_pattern += other.rejected_pattern;
        self.rejected_dedup += other.rejected_dedup;
        for (k, v) in &other.length_histogram {
            *self.length_histogram.entry(*k).or_insert(0) += v;
        }
    }
}
