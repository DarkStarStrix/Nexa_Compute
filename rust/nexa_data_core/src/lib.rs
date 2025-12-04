use pyo3::prelude::*;

#[cfg(not(target_os = "windows"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_os = "windows"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

mod transforms;
mod schema;
mod shuffle;
mod batching;
mod stats;

/// Nexa Data Core - Fast, deterministic data preprocessing
#[pymodule]
fn nexa_data_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(preprocess_corpus, m)?)?;
    m.add_function(wrap_pyfunction!(stream_records, m)?)?;
    m.add_function(wrap_pyfunction!(split_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(write_parquet_shards, m)?)?;
    m.add_function(wrap_pyfunction!(convert_csv_to_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::shuffle_and_split, m)?)?;
    m.add_function(wrap_pyfunction!(batching::parallel_process_files, m)?)?;
    m.add_function(wrap_pyfunction!(stats::compute_stats_json, m)?)?;
    Ok(())
}

#[pyfunction]
fn preprocess_corpus(_cfg: String) -> PyResult<String> {
    Ok("Not implemented".to_string())
}

#[pyfunction]
fn stream_records(_path: String, _batch_size: usize) -> PyResult<String> {
    Ok("Not implemented".to_string())
}

#[pyfunction]
fn split_dataset(_cfg: String) -> PyResult<String> {
    Ok("Not implemented".to_string())
}

#[pyfunction]
fn write_parquet_shards(_records: String, _cfg: String) -> PyResult<String> {
    Ok("Not implemented".to_string())
}

#[pyfunction]
fn convert_csv_to_parquet(input_path: String, output_path: String, batch_size: usize) -> PyResult<()> {
    transforms::convert_csv_to_parquet(&input_path, &output_path, batch_size)?;
    Ok(())
}
