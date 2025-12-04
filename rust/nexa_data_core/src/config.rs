use pyo3.prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct PreprocessingConfig {
    #[pyo3(get, set)]
    pub input_path: String,
    #[pyo3(get, set)]
    pub output_dir: String,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub file_type: String, // "csv" or "jsonl"
    #[pyo3(get, set)]
    pub schema: Option<String>,
}

#[pymethods]
impl PreprocessingConfig {
    #[new]
    fn new(input_path: String, output_dir: String, batch_size: usize, file_type: String, schema: Option<String>) -> Self {
        PreprocessingConfig {
            input_path,
            output_dir,
            batch_size,
            file_type,
            schema,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct SplitConfig {
    #[pyo3(get, set)]
    pub input_path: String, // Path to parquet file or directory
    #[pyo3(get, set)]
    pub output_dir: String,
    #[pyo3(get, set)]
    pub train_ratio: f64,
    #[pyo3(get, set)]
    pub val_ratio: f64,
    #[pyo3(get, set)]
    pub test_ratio: f64,
    #[pyo3(get, set)]
    pub seed: u64,
}

#[pymethods]
impl SplitConfig {
    #[new]
    fn new(input_path: String, output_dir: String, train_ratio: f64, val_ratio: f64, test_ratio: f64, seed: u64) -> Self {
        SplitConfig {
            input_path,
            output_dir,
            train_ratio,
            val_ratio,
            test_ratio,
            seed,
        }
    }
}
