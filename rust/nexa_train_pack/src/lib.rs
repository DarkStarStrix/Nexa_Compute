use pyo3::prelude::*;

#[cfg(not(target_os = "windows"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_os = "windows"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

mod packing;
mod sampling;

#[pymodule]
fn nexa_train_pack(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pack_sequences, m)?)?;
    Ok(())
}

#[pyfunction]
fn pack_sequences(_shards: Vec<String>, _cfg: String) -> PyResult<String> {
    // Placeholder
    Ok("Not implemented".to_string())
}
