use pyo3::prelude::*;

#[cfg(not(target_os = "windows"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_os = "windows"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

mod tests;
mod hist;
mod metrics;

#[pymodule]
fn nexa_stats(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ks_test, m)?)?;
    m.add_function(wrap_pyfunction!(psi, m)?)?;
    m.add_function(wrap_pyfunction!(chi_square, m)?)?;
    m.add_function(wrap_pyfunction!(compute_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_reductions, m)?)?;
    Ok(())
}

#[pyfunction]
fn ks_test(ref_data: Vec<f64>, cur_data: Vec<f64>) -> PyResult<f64> {
    Ok(tests::ks_statistic(&ref_data, &cur_data))
}

#[pyfunction]
fn psi(ref_data: Vec<f64>, cur_data: Vec<f64>) -> PyResult<f64> {
    // Assuming raw data, internal bucketing?
    // Or passing pre-bucketed? API implied raw data in spec A2
    Ok(tests::population_stability_index(&ref_data, &cur_data, 10))
}

#[pyfunction]
fn chi_square(obs: Vec<f64>, exp: Vec<f64>) -> PyResult<f64> {
    Ok(tests::chi_square_test(&obs, &exp))
}

#[pyfunction]
fn compute_histogram(data: Vec<f64>, bins: usize) -> PyResult<String> {
    let hist = hist::compute(&data, bins);
    serde_json::to_string(&hist).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn compute_reductions(data: Vec<f64>) -> PyResult<String> {
    let m = metrics::reduce(&data);
    serde_json::to_string(&m).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}
