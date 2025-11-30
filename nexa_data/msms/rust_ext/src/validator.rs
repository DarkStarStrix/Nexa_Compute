use numpy::{PyArray1, PyArrayMethods};
/// Fast validation routines with SIMD optimizations.
use pyo3::prelude::*;

#[pyfunction]
pub fn validate_spectrum(
    _py: Python,
    mzs: &Bound<'_, PyArray1<f32>>,
    ints: &Bound<'_, PyArray1<f32>>,
    precursor_mz: f32,
) -> PyResult<bool> {
    // Fast validation using Rust
    // Safe to use as_slice when we know the array is contiguous and not being modified
    let mzs_slice = unsafe { mzs.as_slice()? };
    let ints_slice = unsafe { ints.as_slice()? };

    // Check shape consistency
    if mzs_slice.len() != ints_slice.len() {
        return Ok(false);
    }

    if mzs_slice.is_empty() {
        return Ok(false);
    }

    // Check for non-finite values (NaN, Inf)
    for &mz in mzs_slice {
        if !mz.is_finite() {
            return Ok(false);
        }
    }

    for &intensity in ints_slice {
        if !intensity.is_finite() {
            return Ok(false);
        }
    }

    // Check for negative m/z values
    for &mz in mzs_slice {
        if mz <= 0.0 {
            return Ok(false);
        }
    }

    // Check precursor_mz
    if !precursor_mz.is_finite() || precursor_mz <= 0.0 {
        return Ok(false);
    }

    Ok(true)
}
