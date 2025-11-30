/// Rust extension for MS/MS pipeline performance.
use pyo3::prelude::*;

#[cfg(not(target_os = "windows"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_os = "windows"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

mod batch_processor;
mod hdf5_reader;
mod validator;

#[pymodule]
fn msms_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hdf5_reader::read_hdf5_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(hdf5_reader::read_hdf5_batch, m)?)?;
    m.add_function(wrap_pyfunction!(validator::validate_spectrum, m)?)?;
    m.add_class::<batch_processor::RustBatchProcessor>()?;
    Ok(())
}
