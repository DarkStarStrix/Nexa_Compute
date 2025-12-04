use std::collections::HashMap;
use arrow::record_batch::RecordBatch;
use arrow::array::{Array, Float64Array, Int64Array};
use arrow::datatypes::DataType;
use serde::{Serialize, Deserialize};
use pyo3::prelude::*;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ColumnStats {
    pub count: usize,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub sum: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DatasetStats {
    pub total_rows: usize,
    pub columns: HashMap<String, ColumnStats>,
}

impl DatasetStats {
    pub fn update(&mut self, batch: &RecordBatch) {
        self.total_rows += batch.num_rows();
        
        for field in batch.schema().fields() {
            let col_name = field.name();
            let col_stats = self.columns.entry(col_name.clone()).or_default();
            
            let array = batch.column_by_name(col_name).unwrap();
            col_stats.count += array.len() - array.null_count();
            
            match array.data_type() {
                DataType::Float64 => {
                    let values = array.as_any().downcast_ref::<Float64Array>().unwrap();
                    for val in values.iter().flatten() {
                        col_stats.sum += val;
                        col_stats.min = Some(col_stats.min.map_or(*val, |m| m.min(*val)));
                        col_stats.max = Some(col_stats.max.map_or(*val, |m| m.max(*val)));
                    }
                },
                DataType::Int64 => {
                    let values = array.as_any().downcast_ref::<Int64Array>().unwrap();
                    for val in values.iter().flatten() {
                        let val_f = *val as f64;
                        col_stats.sum += val_f;
                        col_stats.min = Some(col_stats.min.map_or(val_f, |m| m.min(val_f)));
                        col_stats.max = Some(col_stats.max.map_or(val_f, |m| m.max(val_f)));
                    }
                },
                _ => {} // Ignore other types for min/max/sum for now
            }
        }
    }
}

#[pyfunction]
pub fn compute_stats_json(_file_path: String) -> PyResult<String> {
    // Placeholder: In real implementation, read parquet file, iterate batches, update stats
    let stats = DatasetStats::default();
    serde_json::to_string(&stats).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}
