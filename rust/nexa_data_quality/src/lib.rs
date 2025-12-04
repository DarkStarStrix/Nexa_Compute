use pyo3::prelude::*;

#[cfg(not(target_os = "windows"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_os = "windows"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

mod filters;
mod dedup;
mod metrics;

#[pymodule]
fn nexa_data_quality(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(filter_batch, m)?)?;
    m.add_function(wrap_pyfunction!(deduplicate_batch, m)?)?;
    Ok(())
}

#[pyfunction]
fn filter_batch(_input_path: String, _cfg: String) -> PyResult<String> {
    // Placeholder for actual implementation that reads parquet, applies filters, returns stats
    Ok("Not implemented".to_string())
}

#[pyfunction]
fn deduplicate_batch(_input_path: String) -> PyResult<String> {
    // Placeholder for actual dedup logic
    Ok("Not implemented".to_string())
}
